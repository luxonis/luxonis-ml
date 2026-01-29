import ast
from collections.abc import Iterator

import rich.box
import typer
from rich import print as rprint
from rich.console import RenderableType, group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.utils.enums import BucketStorage


def parse_split_ratio(
    value: str | None,
) -> dict[str, float | int] | None:
    """Parse split ratio argument.

    Expects a Python list (e.g., C{"[0.8, 0.1, 0.1]"}). If values sum to
    1.0, treated as ratios. Otherwise, treated as counts.
    """
    if value is None:
        return None

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        raise typer.BadParameter(f"Invalid list syntax: {e}") from e

    if not isinstance(parsed, list) or len(parsed) != 3:
        raise typer.BadParameter(
            "Split ratio must be a list of 3 values (train, val, test)."
        )

    if not all(isinstance(v, int | float) for v in parsed):
        raise typer.BadParameter("Split ratio values must be numbers.")

    all_ints = all(isinstance(v, int) for v in parsed)
    all_floats = all(isinstance(v, float) for v in parsed)

    if not (all_ints or all_floats):
        raise typer.BadParameter(
            "Split ratio values must be all integers (counts) "
            "or all floats (ratios), not a mix."
        )

    if all_floats and abs(sum(parsed) - 1.0) >= 1e-6:
        raise typer.BadParameter(
            f"Float ratios must sum to 1.0, but got {sum(parsed):.2f}."
        )

    keys = ["train", "val", "test"]
    return dict(
        zip(
            keys,
            parsed if all_floats else [int(v) for v in parsed],
            strict=True,
        )
    )


def check_exists(name: str, bucket_storage: BucketStorage) -> None:
    if not LuxonisDataset.exists(name, bucket_storage=bucket_storage):
        rprint(f"[red]Dataset [magenta]'{name}'[red] does not exist.")
        raise typer.Exit


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

    @group()
    def get_sizes_panel() -> Iterator[RenderableType]:
        if splits is not None:
            total_groups = len(dataset) / len(source_names)
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
        yield f"[magenta b]Bucket Storage: [not b cyan]{dataset.bucket_storage.value}"
        yield f"[magenta b]Team ID: [not b cyan]{dataset.team_id}"
        yield ""
        yield Panel.fit(get_sizes_panel(), title="Split Sizes")
        yield class_table
        yield task_table

    rprint(Panel.fit(get_panels(), title="Dataset Info"))


def complete_dataset_name(incomplete: str) -> list[str]:
    datasets = LuxonisDataset.list_datasets()
    return [d for d in datasets if d.startswith(incomplete)]
