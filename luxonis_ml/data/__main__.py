# test
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rich.box
import typer
import yaml
from rich import print
from rich.console import Console, group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from typing_extensions import Annotated

from luxonis_ml.data import (
    Augmentations,
    LabelType,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
)
from luxonis_ml.data.utils.visualizations import visualize
from luxonis_ml.enums import DatasetType

logger = logging.getLogger(__name__)

app = typer.Typer()


def complete_dataset_name(incomplete: str):
    datasets = LuxonisDataset.list_datasets()
    return [d for d in datasets if d.startswith(incomplete)]


DatasetNameArgument = Annotated[
    str,
    typer.Argument(
        ..., help="Name of the dataset.", autocompletion=complete_dataset_name
    ),
]


def check_exists(name: str):
    if not LuxonisDataset.exists(name):
        print(f"[red]Dataset [magenta]'{name}'[red] does not exist.")
        raise typer.Exit()


def get_dataset_info(name: str) -> Tuple[int, List[str], List[str]]:
    dataset = LuxonisDataset(name)
    size = len(dataset)
    classes, _ = dataset.get_classes()
    return size, classes, dataset.get_tasks()


def print_info(name: str) -> None:
    dataset = LuxonisDataset(name)
    _, classes = dataset.get_classes()
    table = Table(
        title="Classes", box=rich.box.ROUNDED, row_styles=["yellow", "cyan"], width=88
    )
    table.add_column("Task", header_style="magenta i")
    table.add_column("Class Names", header_style="magenta i")
    for task, c in classes.items():
        table.add_row(task, ", ".join(c))

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
        yield ""
        yield Panel.fit(get_sizes_panel(), title="Split Sizes")
        yield table

    print(
        Panel.fit(
            get_panels(),
            title="Dataset Info",
        )
    )


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
            _, classes, tasks = get_dataset_info(name)
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
):
    """Inspects images and annotations in a dataset."""

    view = view or ["train"]
    dataset = LuxonisDataset(name)
    h, w, _ = LuxonisLoader(dataset, view=view)[0][0].shape
    augmentations = None

    if aug_config is not None:
        with open(aug_config) as file:
            config = (
                yaml.safe_load(file)
                if Path(aug_config).suffix == ".yaml"
                else json.load(file)
            )
        augmentations = Augmentations([h, w], config)

    if len(dataset) == 0:
        raise ValueError(f"Dataset '{name}' is empty.")

    loader = LuxonisLoader(dataset, view=view, augmentations=augmentations)
    class_names = dataset.get_classes()[1]
    for image, labels in loader:
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        new_h, new_w = int(h * size_multiplier), int(w * size_multiplier)
        image = cv2.resize(image, (new_w, new_h))
        image = visualize(image, labels, class_names)
        cv2.imshow("image", image)
        if cv2.waitKey() == ord("q"):
            break


def _parse_tasks(values: Optional[List[str]]) -> List[Tuple[LabelType, str]]:
    if not values:
        return []
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid task format: {value}. Expected 'label_type=task_name'."
            )
        k, v = value.split("=")
        if k not in LabelType.__members__.values():
            raise ValueError(f"Invalid task type: {k}")
        result[LabelType(k.strip())] = v.strip()
    return list(result.items())


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
            help="Type of the dataset. If not provided, the parser will try to recognize it automatically.",
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
            help="If a remote URL is provided in 'dataset_dir', the dataset will "
            "be downloaded to this directory. Otherwise, the dataset will be "
            "downloaded to the current working directory.",
            show_default=False,
        ),
    ] = None,
    task_name: Annotated[
        Optional[List[str]],
        typer.Option(
            ...,
            "--task-name",
            "-tn",
            show_default=False,
            callback=_parse_tasks,
            help="Custom task names to override the default ones. "
            "Format: 'label_type=task_name'. E.g. 'boundingbox=detection-task'.",
        ),
    ] = None,
):
    """Parses a directory with data and creates Luxonis dataset."""
    task_name = task_name or []
    parser = LuxonisParser(
        dataset_dir,
        dataset_name=name,
        dataset_type=dataset_type,
        delete_existing=delete_existing,
        save_dir=save_dir,
        task_mapping=dict(task_name),  # type: ignore
    )
    dataset = parser.parse()

    print()
    print(Rule())
    print()
    print_info(dataset.identifier)


if __name__ == "__main__":
    app()
