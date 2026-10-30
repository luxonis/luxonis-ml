"""Helpers for the commands of ``luxonis_ml data``."""

import ast
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import rich.box
from rich import print as rprint
from rich.console import RenderableType, group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.typing import Params, TrackedAugmentations, check_type


def get_tracked_augmentations(
    metadata: Params,
) -> TrackedAugmentations | None:
    """Read the augmentation provenance the loader added to a sample.

    A record can store its own ``"augmentations"`` metadata, which the
    loader keeps instead. The marker type tells the two apart; the shape
    alone does not.

    Returns:
        The augmentations that the loader tracked for the sample, or
        ``None`` if the ``"augmentations"`` value is missing or is
        record metadata.

    """
    augmentations = metadata.get("augmentations")
    if isinstance(augmentations, TrackedAugmentations):
        return augmentations
    return None


def parse_split_ratio(
    value: str | None = None,
    train: float | None = None,
    val: float | None = None,
    test: float | None = None,
) -> dict[str, float | int] | None:
    r"""Parse a CLI split-ratio argument.

    Float lists represent ratios and must sum to :math:`1.0`. Integer
    lists represent absolute counts.

    Args:
        value: A string representation of a list
            of 3 values (train, val, test).
        train:
            Optional float or int for training split.
        val:
            Optional float or int for validation split.
        test:
            Optional float or int for test split.

    Returns:
        A dictionary with keys "train", "val", and "test" mapping to their
        respective ratios or counts, or None if no values were provided.

    Raises:
        ValueError: If the input format is invalid, if ratios don't sum to 1.0,
        or if there's a mix of counts and ratios.

    Example:
        >>> parse_split_ratio("[0.7, 0.2, 0.1]")
        {'train': 0.7, 'val': 0.2, 'test': 0.1}
        >>> parse_split_ratio(train=70, val=20, test=10)
        {'train': 70, 'val': 20, 'test': 10}
        >>> parse_split_ratio(train=0.7, val=0.2)
        {'train': 0.7, 'val': 0.2, 'test': 0.1}
        >>> parse_split_ratio(train=0.7, val=0.2, test=0.1)
        {'train': 0.7, 'val': 0.2, 'test': 0.1}
        >>> parse_split_ratio("[10, 27]")
        Traceback (most recent call last):
        ...
        ValueError: Split ratio must be a list of 3 values (train, val, test).
        >>> parse_split_ratio(train=0.7, val=0.2, test=0.2)
        Traceback (most recent call last):
        ...
        ValueError: Split ratios must sum to 1.0; use whole numbers for counts.

    """
    if value is not None and any(v is not None for v in (train, val, test)):
        raise ValueError(
            "Cannot specify split ratio both as a list and as separate "
            "train/val/test arguments."
        )
    if not any(v is not None for v in (value, train, val, test)):
        return None

    parsed = (
        _parse_split_list(value)
        if value is not None
        else _split_from_parts(train, val, test)
    )
    return _checked_split_values(parsed)


def _parse_split_list(value: str) -> dict[str, Any]:
    """Parse a ``"[train, val, test]"`` list literal into a split mapping."""
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid list syntax: {e}") from e
    if not isinstance(parsed, list) or len(parsed) != 3:
        raise ValueError(
            "Split ratio must be a list of 3 values (train, val, test)."
        )
    return dict(zip(["train", "val", "test"], parsed, strict=True))


def _split_from_parts(
    train: float | None, val: float | None, test: float | None
) -> dict[str, Any]:
    """Build a split mapping from the given parts.

    Whole numbers that sum above 1 are counts. Otherwise the parts are
    ratios, and the missing ones share the remainder to 1 equally.

    Returns:
        A mapping of ``"train"``, ``"val"`` and ``"test"`` to their
        values. Counts are integers, and a missing count is 0.

    Raises:
        ValueError: If the parts sum above 1 but are not all whole
            numbers.

    """
    defined = [v for v in (train, val, test) if v is not None]
    sum_ = sum(defined)
    is_count_input = all(float(v).is_integer() for v in defined)

    if sum_ > 1 + 1e-6 and not is_count_input:
        raise ValueError(
            "Split ratios must sum to 1.0; use whole numbers for counts."
        )

    if sum_ > 1 + 1e-6:
        return {
            "train": int(train or 0),
            "val": int(val or 0),
            "test": int(test or 0),
        }

    n_defined = len(defined)
    rem = (1 - sum_) / (3 - n_defined) if n_defined < 3 else 0
    return {
        "train": train if train is not None else rem,
        "val": val if val is not None else rem,
        "test": test if test is not None else rem,
    }


def _checked_split_values(parsed: dict[str, Any]) -> dict[str, float | int]:
    """Check that the split values are all counts or all ratios that sum to 1."""
    if not check_type(parsed, dict[str, float | int]):
        raise ValueError("Split ratio values must be numbers.")

    all_ints = all(isinstance(v, int) for v in parsed.values())
    all_floats = all(isinstance(v, float) for v in parsed.values())

    if not (all_ints or all_floats):
        raise ValueError(
            "Split ratio values must be all integers (counts) "
            "or all floats (ratios), not a mix."
        )

    if all_floats and abs(sum(parsed.values()) - 1.0) >= 1e-6:
        raise ValueError(
            f"Float ratios must sum up to 1.0, "
            f"but got {sum(parsed.values()):.2f}."
        )

    if all_ints:
        return {k: int(v) for k, v in parsed.items()}
    return {k: float(v) for k, v in parsed.items()}


def check_exists(name: str, bucket_storage: BucketStorage) -> None:
    """Ensure that a dataset exists in the selected storage.

    Args:
        name: Dataset name.
        bucket_storage: Storage backend to query.

    Raises:
        SystemExit: If the dataset does not exist.

    """
    if not LuxonisDataset.exists(name, bucket_storage=bucket_storage):
        rprint(f"[red]Dataset [magenta]'{name}'[red] does not exist.")
        raise SystemExit(1)


def get_dataset_info(dataset: LuxonisDataset) -> tuple[set[str], list[str]]:
    """Return the class names and the task names of a dataset.

    Args:
        dataset: Dataset to read.

    Returns:
        The set of class names over all tasks, and the list of task
        names.

    """
    all_classes = {
        c for classes in dataset.get_classes().values() for c in classes
    }
    return all_classes, dataset.get_task_names()


def print_info(dataset: LuxonisDataset) -> None:
    """Print a summary of a dataset to the console.

    The summary shows the name, version, bucket storage, team ID, the
    size and share of each split, and tables of the classes and the
    task types.

    Args:
        dataset: Dataset to summarize.

    """
    class_table = _per_task_table(
        "Classes", "Class Names", dataset.get_classes()
    )
    task_table = _per_task_table(
        "Tasks",
        "Task Types",
        {name: sorted(types) for name, types in dataset.get_tasks().items()},
    )

    splits = dataset.get_splits()
    source_names = dataset.get_source_names()
    total_groups = len(dataset)
    if source_names:
        total_groups /= len(source_names)

    @group()
    def get_panels() -> Iterator[RenderableType]:
        yield f"[magenta b]Name: [not b cyan]{dataset.identifier}"
        yield f"[magenta b]Version: [not b cyan]{dataset.version}"
        yield f"[magenta b]Bucket Storage: [not b cyan]{dataset._bucket_storage.value}"
        yield f"[magenta b]Team ID: [not b cyan]{dataset._team_id}"
        yield ""
        yield Panel.fit(
            _sizes_panel(splits, total_groups), title="Split Sizes"
        )
        yield class_table
        yield task_table

    rprint(Panel.fit(get_panels(), title="Dataset Info"))


def _per_task_table(
    title: str, value_header: str, rows: Mapping[str, Iterable[str]]
) -> Table:
    """Build a table with one row per task.

    The task name column appears only when some task has a name.

    Returns:
        The table, titled ``title``. Each row holds the values of one
        task, joined by commas.

    """
    has_names = any(rows)
    table = Table(
        title=title, box=rich.box.ROUNDED, row_styles=["yellow", "cyan"]
    )
    if has_names:
        table.add_column("Task Name", header_style="magenta i", max_width=30)
    table.add_column(value_header, header_style="magenta i", max_width=50)
    for task_name, values in rows.items():
        if has_names:
            table.add_row(task_name, ", ".join(values))
        else:
            table.add_row(", ".join(values))
    return table


@group()
def _sizes_panel(
    splits: dict[str, list[str]] | None, total_groups: float
) -> Iterator[RenderableType]:
    """Render the size and the share of each split, then the total."""
    if splits:
        for split, group_ids in splits.items():
            split_size = len(group_ids)
            percentage = (
                (split_size / total_groups * 100) if total_groups > 0 else 0
            )
            yield f"[magenta b]{split}: [not b cyan]{split_size:,} [dim]({percentage:.1f}%)[/dim]"
    else:
        yield "[red]No splits found"
    yield Rule()
    yield f"[magenta b]Total: [not b cyan]{int(total_groups)}"


def complete_dataset_name(incomplete: str) -> list[str]:
    """Return the local dataset names that start with a prefix.

    Args:
        incomplete: Prefix of the dataset name.

    Returns:
        Names of the local datasets that start with ``incomplete``.

    """
    datasets = LuxonisDataset.list_datasets()
    return [d for d in datasets if d.startswith(incomplete)]
