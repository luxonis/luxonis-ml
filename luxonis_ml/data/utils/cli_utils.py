import ast
from collections.abc import Iterator

import rich.box
from rich import print as rprint
from rich.console import RenderableType, group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.typing import check_type


def parse_split_ratio(
    value: str | None = None,
    train: float | None = None,
    val: float | None = None,
    test: float | None = None,
) -> dict[str, float | int] | None:
    """Parse split ratio argument.

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
        >>> parse_split_ratio([10, 27])
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

    if value is not None:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid list syntax: {e}") from e
        if not isinstance(parsed, list) or len(parsed) != 3:
            raise ValueError(
                "Split ratio must be a list of 3 values (train, val, test)."
            )
        parsed = dict(zip(["train", "val", "test"], parsed, strict=True))
    else:
        defined = [v for v in (train, val, test) if v is not None]
        sum_ = sum(defined)
        is_count_input = all(float(v).is_integer() for v in defined)

        if sum_ > 1 + 1e-6 and not is_count_input:
            raise ValueError(
                "Split ratios must sum to 1.0; use whole numbers for counts."
            )

        if sum_ > 1 + 1e-6:
            parsed = {
                "train": int(train or 0),
                "val": int(val or 0),
                "test": int(test or 0),
            }
        else:
            n_defined = sum(v is not None for v in (train, val, test))
            rem = (1 - sum_) / (3 - n_defined) if n_defined < 3 else 0
            parsed = {
                "train": train if train is not None else rem,
                "val": val if val is not None else rem,
                "test": test if test is not None else rem,
            }

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
    if not LuxonisDataset.exists(name, bucket_storage=bucket_storage):
        rprint(f"[red]Dataset [magenta]'{name}'[red] does not exist.")
        raise SystemExit(1)


def get_dataset_info(dataset: LuxonisDataset) -> tuple[set[str], list[str]]:
    all_classes = {
        c for classes in dataset.get_classes().values() for c in classes
    }
    return all_classes, dataset.get_task_names()


def print_info(dataset: LuxonisDataset) -> None:
    classes = dataset.get_classes()
    has_named_classes = any(k for k in classes if k)
    class_table = Table(
        title="Classes", box=rich.box.ROUNDED, row_styles=["yellow", "cyan"]
    )
    if has_named_classes:
        class_table.add_column(
            "Task Name", header_style="magenta i", max_width=30
        )
    class_table.add_column(
        "Class Names", header_style="magenta i", max_width=50
    )

    for task_name, c in classes.items():
        if has_named_classes:
            class_table.add_row(task_name, ", ".join(c))
        else:
            class_table.add_row(", ".join(c))

    tasks = dataset.get_tasks()
    has_named_tasks = any(k for k in tasks if k)

    task_table = Table(
        title="Tasks", box=rich.box.ROUNDED, row_styles=["yellow", "cyan"]
    )
    if has_named_tasks:
        task_table.add_column(
            "Task Name", header_style="magenta i", max_width=30
        )
    task_table.add_column("Task Types", header_style="magenta i", max_width=50)
    for task_name, task_types in tasks.items():
        task_types.sort()
        if has_named_tasks:
            task_table.add_row(task_name, ", ".join(task_types))
        else:
            task_table.add_row(", ".join(task_types))

    splits = dataset.get_splits()
    source_names = dataset.get_source_names()
    total_groups = len(dataset)
    if source_names:
        total_groups /= len(source_names)

    @group()
    def get_sizes_panel() -> Iterator[RenderableType]:
        if splits:
            for split, group in splits.items():
                split_size = len(group)
                percentage = (
                    (split_size / total_groups * 100)
                    if total_groups > 0
                    else 0
                )
                yield f"[magenta b]{split}: [not b cyan]{split_size:,} [dim]({percentage:.1f}%)[/dim]"
        else:
            yield "[red]No splits found"
        yield Rule()
        yield f"[magenta b]Total: [not b cyan]{int(total_groups)}"

    @group()
    def get_panels() -> Iterator[RenderableType]:
        yield f"[magenta b]Name: [not b cyan]{dataset.identifier}"
        yield f"[magenta b]Version: [not b cyan]{dataset.version}"
        yield f"[magenta b]Bucket Storage: [not b cyan]{dataset._bucket_storage.value}"
        yield f"[magenta b]Team ID: [not b cyan]{dataset._team_id}"
        yield ""
        yield Panel.fit(get_sizes_panel(), title="Split Sizes")
        yield class_table
        yield task_table

    rprint(Panel.fit(get_panels(), title="Dataset Info"))


def complete_dataset_name(incomplete: str) -> list[str]:
    datasets = LuxonisDataset.list_datasets()
    return [d for d in datasets if d.startswith(incomplete)]
