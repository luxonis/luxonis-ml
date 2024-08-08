import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rich.box
import typer
import yaml
from rich import print
from rich.console import Console
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
from luxonis_ml.enums import DatasetType, SplitType

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


def create_text_image(
    text: str,
    width: int,
    height: int,
    font_size: float = 0.7,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
):
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size = cv2.getTextSize(text, font, font_size, 1)[0]

    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(
        img, text, (text_x, text_y), font, font_size, text_color, 1, cv2.LINE_AA
    )

    return img


def concat_images(
    image_dict: Dict[str, np.ndarray], padding: int = 10, label_height: int = 30
):
    image_dict = dict(
        sorted(image_dict.items(), key=lambda x: x[0] if x[0] != "image" else "a")
    )
    n_images = len(image_dict)
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)

    max_h = max(img.shape[0] for img in image_dict.values())
    max_w = max(img.shape[1] for img in image_dict.values())

    cell_height = max_h + 2 * padding + label_height
    cell_width = max_w + 2 * padding

    output = np.full(
        (cell_height * n_rows, cell_width * n_cols, 3), 255, dtype=np.uint8
    )

    for idx, (name, img) in enumerate(image_dict.items()):
        i = idx // n_cols
        j = idx % n_cols

        y_start = i * cell_height
        x_start = j * cell_width

        label = create_text_image(name, cell_width, label_height)
        output[y_start : y_start + label_height, x_start : x_start + cell_width] = label

        h, w = img.shape[:2]
        y_img = y_start + label_height + padding
        x_img = x_start + padding
        output[y_img : y_img + h, x_img : x_img + w] = img

    return output


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
        SplitType,
        typer.Option(
            ...,
            "--view",
            "-v",
            help="Which split of the dataset to inspect.",
            case_sensitive=False,
        ),
    ] = "train",  # type: ignore
    aug_config: Annotated[
        Optional[Path],
        typer.Option(
            ...,
            "--aug-config",
            "-a",
            help="Path to a config defining augmentations. "
            "This can be either a json or a yaml file.",
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

    def _task_to_rgb(string: str) -> tuple:
        h = int(hashlib.md5(string.encode()).hexdigest(), 16)
        r = (h & 0xFF0000) >> 16
        g = (h & 0x00FF00) >> 8
        b = h & 0x0000FF

        return (r, g, b)

    augmentations = None
    if aug_config is not None:
        with open(aug_config) as file:
            config = (
                yaml.safe_load(file)
                if aug_config.suffix == ".yaml"
                else json.load(file)
            )
        augmentations = Augmentations([720, 1280], config)

    dataset = LuxonisDataset(name)
    loader = LuxonisLoader(dataset, view=view.value, augmentations=augmentations)
    classes = dataset.get_classes()[1]
    for image, ann in loader:
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        new_h, new_w = int(h * size_multiplier), int(w * size_multiplier)
        image = cv2.resize(image, (new_w, new_h))
        images = {"image": image}

        for task, (arr, label_type) in ann.items():
            if label_type == LabelType.SEGMENTATION:
                mask_viz = np.zeros((h, w, 3)).astype(np.uint8)
                for i, mask in enumerate(arr):
                    mask_viz[mask == 1] = (
                        _task_to_rgb(f"{task}_{i}")
                        if (i != 0 or len(arr) == 1)
                        else (0, 0, 0)
                    )
                mask_viz = cv2.resize(mask_viz, (new_w, new_h))
                images[task] = mask_viz

        h, w, _ = image.shape

        for task, (arr, label_type) in ann.items():
            if label_type == LabelType.BOUNDINGBOX:
                for box in arr:
                    cv2.rectangle(
                        image,
                        (int(box[1] * w), int(box[2] * h)),
                        (int(box[1] * w + box[3] * w), int(box[2] * h + box[4] * h)),
                        _task_to_rgb(classes[task][int(box[0])]),
                        2,
                    )

            if label_type == LabelType.KEYPOINTS:
                for kp in arr:
                    kp = kp[1:].reshape(-1, 3)
                    for k in kp:
                        cv2.circle(
                            image,
                            (int(k[0] * w), int(k[1] * h)),
                            2,
                            _task_to_rgb(task),
                            2,
                        )

        image = concat_images(images)
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
