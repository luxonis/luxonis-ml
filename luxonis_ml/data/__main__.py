import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import typer
from rich import box, print
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from typing_extensions import Annotated

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.enums import DatasetType, LabelType, SplitType

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
    try:
        loader = LuxonisLoader(dataset, view=SplitType.TRAIN.value)
        _, ann = next(iter(loader))
    except Exception:
        ann = []
    classes, _ = dataset.get_classes()
    tasks = [task.value for task in ann]
    return size, classes, tasks


def print_info(name: str) -> None:
    size, classes, tasks = get_dataset_info(name)
    print(
        Panel.fit(
            f"[magenta b]Name: [not b cyan]{name}\n"
            f"[magenta b]Size: [not b cyan]{size}\n"
            f"[magenta b]Classes: [not b cyan]{', '.join(classes)}\n"
            f"[magenta b]Tasks: [not b cyan]{', '.join(tasks)}",
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
        box=box.ROUNDED,
        row_styles=["yellow", "cyan"],
    )
    table.add_column("Name", header_style="magenta i")
    table.add_column("Size", header_style="magenta i")
    if full:
        table.add_column("Classes", header_style="magenta i")
        table.add_column("Tasks", header_style="magenta i")
    for name in datasets:
        dataset = LuxonisDataset(name)
        rows = [name, str(len(dataset))]
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
        SplitType,
        typer.Option(
            ...,
            "--view",
            "-v",
            help="Which split of the dataset to inspect.",
            case_sensitive=False,
        ),
    ] = "train",  # type: ignore
):
    """Inspects images and annotations in a dataset."""
    dataset = LuxonisDataset(name)
    loader = LuxonisLoader(dataset, view=view.value)
    for image, ann in loader:
        # cls = ann[LabelType.CLASSIFICATION]

        h, w, _ = image.shape
        if LabelType.BOUNDINGBOX in ann:
            box = ann[LabelType.BOUNDINGBOX]
            for b in box:
                cv2.rectangle(
                    image,
                    (int(b[1] * w), int(b[2] * h)),
                    (int(b[1] * w + b[3] * w), int(b[2] * h + b[4] * h)),
                    (255, 0, 0),
                    2,
                )
        if LabelType.SEGMENTATION in ann:
            seg = ann[LabelType.SEGMENTATION]
            mask_viz = np.zeros((h, w, 3)).astype(np.uint8)
            for i, mask in enumerate(seg):
                mask_viz[mask == 1] = 255 / len(seg) * (i + 1)
            image = cv2.addWeighted(image, 0.5, mask_viz, 0.5, 0)

        if LabelType.KEYPOINT in ann:
            kps = ann[LabelType.KEYPOINT]
            for kp in kps:
                kp = kp[1:].reshape(-1, 3)
                for k in kp:
                    cv2.circle(image, (int(k[0] * w), int(k[1] * h)), 2, (0, 255, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
):
    """Parses a directory with data and creates Luxonis dataset."""
    parser = LuxonisParser(
        dataset_dir,
        dataset_name=name,
        dataset_type=dataset_type,
        delete_existing=delete_existing,
        save_dir=save_dir,
    )
    dataset = parser.parse()

    print()
    print(Rule())
    print()
    print_info(dataset.identifier)


if __name__ == "__main__":
    app()
