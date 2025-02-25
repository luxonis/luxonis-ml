import random
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import rich.box
import typer
from rich import print
from rich.console import Console, group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from typing_extensions import Annotated

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.data.utils.visualizations import visualize
from luxonis_ml.enums import DatasetType

app = typer.Typer()


def complete_dataset_name(incomplete: str):
    datasets = LuxonisDataset.list_datasets()
    return [d for d in datasets if d.startswith(incomplete)]


DatasetNameArgument = Annotated[
    str,
    typer.Argument(
        ...,
        help="Name of the dataset.",
        autocompletion=complete_dataset_name,
        show_default=False,
    ),
]


def check_exists(name: str):
    if not LuxonisDataset.exists(name):
        print(f"[red]Dataset [magenta]'{name}'[red] does not exist.")
        raise typer.Exit()


def get_dataset_info(dataset: LuxonisDataset) -> Tuple[Set[str], List[str]]:
    all_classes = {
        c for classes in dataset.get_classes().values() for c in classes
    }
    return all_classes, dataset.get_task_names()


def print_info(name: str) -> None:
    dataset = LuxonisDataset(name)
    classes = dataset.get_classes()
    class_table = Table(
        title="Classes", box=rich.box.ROUNDED, row_styles=["yellow", "cyan"]
    )
    if len(classes) > 1 or next(iter(classes)):
        class_table.add_column(
            "Task Name", header_style="magenta i", max_width=30
        )
    class_table.add_column(
        "Class Names", header_style="magenta i", max_width=50
    )
    for task_name, c in classes.items():
        if not task_name:
            class_table.add_row(", ".join(c))
        else:
            class_table.add_row(task_name, ", ".join(c))

    tasks = dataset.get_tasks()
    task_table = Table(
        title="Tasks", box=rich.box.ROUNDED, row_styles=["yellow", "cyan"]
    )
    if tasks and (len(tasks) > 1 or next(iter(tasks))):
        task_table.add_column(
            "Task Name", header_style="magenta i", max_width=30
        )
    task_table.add_column("Task Types", header_style="magenta i", max_width=50)
    for task_name, task_types in tasks.items():
        task_types.sort()
        if not task_name:
            task_table.add_row(", ".join(task_types))
        else:
            task_table.add_row(task_name, ", ".join(task_types))

    splits = dataset.get_splits()

    @group()
    def get_sizes_panel():
        if splits is not None:
            for split, files in splits.items():
                yield f"[magenta b]{split}: [not b cyan]{len(files)}"
        else:
            yield "[red]No splits found"
        yield Rule()
        yield f"[magenta b]Total: [not b cyan]{len(dataset)}"

    @group()
    def get_panels():
        yield f"[magenta b]Name: [not b cyan]{name}"
        yield f"[magenta b]Version: [not b cyan]{dataset.version}"
        yield ""
        yield Panel.fit(get_sizes_panel(), title="Split Sizes")
        yield class_table
        yield task_table

    print(Panel.fit(get_panels(), title="Dataset Info"))


@app.command()
def info(name: DatasetNameArgument):
    """Prints information about a dataset."""
    check_exists(name)
    print_info(name)


@app.command()
def delete(name: DatasetNameArgument):
    """Deletes a dataset."""
    check_exists(name)

    dataset = LuxonisDataset(name)
    dataset.delete_dataset()
    print(f"Dataset '{name}' deleted.")


@app.command()
def ls(
    full: Annotated[
        bool, typer.Option("--full", "-f", help="Show full information.")
    ] = False,
):
    """Lists all datasets."""
    datasets = LuxonisDataset.list_datasets()
    table = Table(
        title="Datasets" + (" - Full Table" if full else ""),
        box=rich.box.ROUNDED,
        row_styles=["yellow", "cyan"],
    )
    table.add_column("Name", header_style="magenta i")
    table.add_column("Size", header_style="magenta i")
    if full:
        table.add_column("Classes", header_style="magenta i")
        table.add_column("Tasks", header_style="magenta i")
    for name in datasets:
        dataset = LuxonisDataset(name)
        rows = [name]
        try:
            size = len(dataset)
        except KeyError:
            size = -1
        rows.append(str(size))
        if full:
            classes, tasks = get_dataset_info(dataset)
            rows.extend(
                [
                    ", ".join(classes) if classes else "[red]<empty>[no red]",
                    ", ".join(tasks) if tasks else "[red]<empty>[no red]",
                ]
            )
        table.add_row(*rows)
    console = Console()
    console.print(table)


@app.command()
def inspect(
    name: DatasetNameArgument,
    view: Annotated[
        Optional[List[str]],
        typer.Option(
            ...,
            "--view",
            "-v",
            help="Which splits of the dataset to inspect.",
            case_sensitive=False,
            show_default=False,
        ),
    ] = None,
    aug_config: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--aug-config",
            "-a",
            help="Path to a config defining augmentations. "
            "This can be either a json or a yaml file.",
            metavar="PATH",
            show_default=False,
        ),
    ] = None,
    size_multiplier: Annotated[
        float,
        typer.Option(
            ...,
            "--size-multiplier",
            "-s",
            help=(
                "Multiplier for the image size. "
                "By default the images are shown in their original size."
            ),
            show_default=False,
        ),
    ] = 1.0,
    ignore_aspect_ratio: Annotated[
        bool,
        typer.Option(
            ...,
            "--ignore-aspect-ratio",
            "-i",
            help="Don't keep the aspect ratio when resizing images.",
        ),
    ] = False,
    deterministic: Annotated[
        bool,
        typer.Option(
            ...,
            "--deterministic",
            "-d",
            help="Deterministic mode. Useful for debugging.",
        ),
    ] = False,
    blend_all: Annotated[
        bool,
        typer.Option(
            ...,
            "--blend-all",
            "-b",
            help="Whether to draw labels belonging "
            "to different tasks on the same image. "
            "Doesn't apply to semantic segmentations.",
        ),
    ] = False,
):
    """Inspects images and annotations in a dataset."""

    if deterministic:
        np.random.seed(42)
        random.seed(42)

    view = view or ["train"]
    dataset = LuxonisDataset(name)
    loader = LuxonisLoader(dataset, view=view)

    if aug_config is not None:
        h, w, _ = loader[0][0].shape
        loader.augmentations = loader._init_augmentations(
            "albumentations", aug_config, h, w, not ignore_aspect_ratio
        )

    if len(dataset) == 0:
        raise ValueError(f"Dataset '{name}' is empty.")

    classes = dataset.get_classes()
    for image, labels in loader:
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        new_h, new_w = int(h * size_multiplier), int(w * size_multiplier)
        image = cv2.resize(image, (new_w, new_h))
        image = visualize(image, labels, classes, blend_all=blend_all)
        cv2.imshow("image", image)
        if cv2.waitKey() == ord("q"):
            break


@app.command()
def parse(
    dataset_dir: Annotated[
        str, typer.Argument(..., help="Path or URL to the dataset.")
    ],
    name: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--name",
            "-n",
            autocompletion=complete_dataset_name,
            help="Name of the dataset.",
            show_default=False,
        ),
    ] = None,
    dataset_type: Annotated[
        Optional[DatasetType],
        typer.Option(
            ...,
            "--type",
            "-t",
            help="Type of the dataset. If not provided, "
            "the parser will try to recognize it automatically.",
            show_default=False,
        ),
    ] = None,
    delete_existing: Annotated[
        bool,
        typer.Option(
            ...,
            "--delete",
            "-d",
            help="If an existing dataset with the same name should "
            "be deleted before parsing.",
        ),
    ] = False,
    save_dir: Annotated[
        Optional[Path],
        typer.Option(
            ...,
            "--save-dir",
            "-s",
            help="If a remote URL is provided in 'dataset_dir', "
            "the dataset will be downloaded to this directory. "
            "Otherwise, the dataset will be downloaded to the "
            "current working directory.",
            show_default=False,
        ),
    ] = None,
    task_name: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--task-name",
            "-tn",
            help="Name of the task that should be used with this dataset. "
            "If not provided, the name of the dataset format will be used.",
            show_default=False,
        ),
    ] = None,
):
    """Parses a directory with data and creates Luxonis dataset."""
    parser = LuxonisParser(
        dataset_dir,
        dataset_name=name,
        dataset_type=dataset_type,
        delete_existing=delete_existing,
        save_dir=save_dir,
        task_name=task_name,
    )
    dataset = parser.parse()

    print()
    print(Rule())
    print()
    print_info(dataset.identifier)


if __name__ == "__main__":
    app()
