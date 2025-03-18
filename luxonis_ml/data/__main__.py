import math
import random
import shutil
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rich.box
import typer
from rich import print
from rich.console import Console, RenderableType, group
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.table import Table
from typing_extensions import Annotated

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.data.utils.enums import BucketStorage
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

bucket_option = typer.Option(
    "local",
    "--bucket-storage",
    "-b",
    help="Storage type for the dataset.",
)


def check_exists(name: str, bucket_storage: BucketStorage):
    if not LuxonisDataset.exists(name, bucket_storage=bucket_storage):
        print(f"[red]Dataset [magenta]'{name}'[red] does not exist.")
        raise typer.Exit


def get_dataset_info(dataset: LuxonisDataset) -> Tuple[Set[str], List[str]]:
    all_classes = {
        c for classes in dataset.get_classes().values() for c in classes
    }
    return all_classes, dataset.get_task_names()


def print_info(dataset: LuxonisDataset) -> None:
    classes = dataset.get_classes()
    class_table = Table(
        title="Classes", box=rich.box.ROUNDED, row_styles=["yellow", "cyan"]
    )
    if len(classes) > 1 or (classes and next(iter(classes))):
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
    def get_sizes_panel() -> Iterator[RenderableType]:
        if splits is not None:
            for split, files in splits.items():
                yield f"[magenta b]{split}: [not b cyan]{len(files)}"
        else:
            yield "[red]No splits found"
        yield Rule()
        yield f"[magenta b]Total: [not b cyan]{len(dataset)}"

    @group()
    def get_panels() -> Iterator[RenderableType]:
        yield f"[magenta b]Name: [not b cyan]{dataset.identifier}"
        yield f"[magenta b]Version: [not b cyan]{dataset.version}"
        yield ""
        yield Panel.fit(get_sizes_panel(), title="Split Sizes")
        yield class_table
        yield task_table

    print(Panel.fit(get_panels(), title="Dataset Info"))


@app.command()
def info(
    name: DatasetNameArgument,
    bucket_storage: BucketStorage = bucket_option,
):
    """Prints information about a dataset."""
    check_exists(name, bucket_storage)
    print_info(LuxonisDataset(name, bucket_storage=bucket_storage))


@app.command()
def delete(
    name: DatasetNameArgument,
    bucket_storage: BucketStorage = bucket_option,
):
    """Deletes a dataset."""
    check_exists(name, bucket_storage)

    if bucket_storage is not BucketStorage.LOCAL and not Confirm.ask(
        f"Are you sure you want to delete the dataset '{name}' "
        f"from remote storage? This will delete all remote files "
        "and cannot be undone. If you only want to delete your local "
        "copy, leave the '--bucket-storage' option as 'local' (default).",
    ):
        raise typer.Exit
    LuxonisDataset(name).delete_dataset(
        delete_remote=bucket_storage is not BucketStorage.LOCAL
    )
    print(f"Dataset '{name}' deleted.")


@app.command()
def ls(
    full: bool = typer.Option(
        False, "--full", "-f", help="Show full information."
    ),
    bucket_storage: BucketStorage = bucket_option,
):
    """Lists all datasets."""
    datasets = LuxonisDataset.list_datasets(bucket_storage=bucket_storage)
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
        dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
        rows = [name]
        try:
            size = len(dataset)
        except Exception:
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
    bucket_storage: BucketStorage = bucket_option,
):
    """Inspects images and annotations in a dataset."""

    if deterministic:
        np.random.seed(42)
        random.seed(42)

    view = view or ["train"]
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
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
def export(
    dataset_name: DatasetNameArgument,
    save_dir: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--save-dir",
            "-s",
            help="Directory where the dataset should be saved. "
            "If not provided, the dataset will be saved in the "
            "current working directory under the name of the dataset.",
            show_default=False,
        ),
    ] = None,
    dataset_type: Annotated[
        DatasetType,
        typer.Option(
            ...,
            "--type",
            "-t",
            help="Format of the exported dataset",
            show_default=False,
        ),
    ] = "coco",  # type: ignore
    delete_existing: Annotated[
        bool,
        typer.Option(
            ...,
            "--delete",
            "-d",
            help="Delete an existing `save_dir` before exporting.",
        ),
    ] = False,
    bucket_storage: BucketStorage = bucket_option,
):
    save_dir = save_dir or dataset_name
    if delete_existing and Path(save_dir).exists():
        shutil.rmtree(save_dir)
    dataset = LuxonisDataset(dataset_name, bucket_storage=bucket_storage)
    dataset.export(save_dir, dataset_type)


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
    print_info(dataset)


@app.command()
def health(
    name: DatasetNameArgument,
    bucket_storage: BucketStorage = bucket_option,
    save_path: Optional[str] = None,
):
    """Checks the health of a dataset."""

    check_exists(name, bucket_storage)
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    stats = dataset.get_statistics()
    console = Console()

    duplicate_uuids = stats["duplicates"]["duplicate_uuids"]
    for item in duplicate_uuids:
        console.print(
            f"[bold red]Warning:[/bold red] UUID [magenta]{item['uuid']}[/magenta] "
            f"appears in multiple files: [cyan]{item['files']}[/cyan]"
        )

    duplicate_annotations = stats.get("duplicates", {}).get(
        "duplicate_annotations", []
    )
    for item in duplicate_annotations:
        console.print(
            f"[bold red]Duplicate Annotation:[/bold red] File [cyan]'{item['file_name']}'[/cyan] "
            f"in task [magenta]'{item['task_name']}'[/magenta] has annotation "
            f"[yellow]'{item['annotation']}'[/yellow] (task type: [magenta]{item['task_type']}[/magenta]) repeated "
            f"{item['count']} times."
        )

    missing_annotations = stats["missing_annotations"]
    for file in missing_annotations:
        console.print(
            f"[bold yellow]Missing Annotation:[/bold yellow] File [cyan]'{file}'[/cyan]"
        )

    console.print(
        "\n[bold underline]Dataset Statistics Summary:[/bold underline]"
    )
    console.print(
        f"- Files with missing annotations: [cyan]{len(missing_annotations)}[/cyan]"
    )
    console.print(
        f"- Files with duplicate UUIDs: [cyan]{len(duplicate_uuids)}[/cyan]"
    )
    console.print(
        f"- Files with duplicate annotations: [cyan]{len(duplicate_annotations)}[/cyan]"
    )

    class_dist = stats["class_distribution"]
    bar_task_types = sorted(
        {
            item["task_type"]
            for item in class_dist
            if item["task_type"] is not None
        }
    )
    heat_task_types = [
        task_type
        for task_type, heat in stats["heatmaps"].items()
        if heat is not None
    ]

    all_task_types = sorted(set(bar_task_types) | set(heat_task_types))
    if not all_task_types:
        console.print("[info]No plots to display.[/info]")
        return

    nrows = len(all_task_types)
    square_size = 4
    fig, axs = plt.subplots(
        nrows, 2, figsize=(square_size * 2, square_size * nrows)
    )

    if nrows == 1:
        axs = [axs]

    for i, task in enumerate(all_task_types):
        ax_bar = axs[i][0]
        if task in bar_task_types:
            task_data = [
                item for item in class_dist if item["task_type"] == task
            ]
            if task_data:
                counts = [x["count"] for x in task_data]
                classes = [x["class_name"] for x in task_data]
                num_classes = len(classes)
                bar_width = (
                    1 / math.pow(num_classes, 0.2) if num_classes > 0 else 1
                )
                bars = ax_bar.bar(
                    classes, counts, width=bar_width, color="royalblue"
                )

                if counts:
                    max_count = max(counts)
                    ax_bar.set_ylim(top=max_count * 1.15)

                ax_bar.set_title(
                    f"{task} Class Distribution", fontsize=12, pad=15
                )
                ax_bar.set_ylabel("Count", fontsize=12)
                ax_bar.tick_params(axis="x", rotation=90, labelbottom=True)
                ax_bar.margins(x=0.01)

                for bar in bars:
                    height = bar.get_height()
                    ax_bar.annotate(
                        f"{height}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )
            else:
                ax_bar.text(
                    0.5, 0.5, "No Distribution Data", ha="center", va="center"
                )
                ax_bar.set_title(f"{task} Class Distribution", fontsize=12)
        else:
            ax_bar.axis("off")
            ax_bar.set_title(f"{task} Class Distribution", fontsize=12)

        ax_heat = axs[i][1]
        heatmap_data = stats["heatmaps"].get(task)
        if heatmap_data is not None:
            matrix = np.array(heatmap_data, dtype=np.float32)
            im = ax_heat.imshow(matrix, cmap="viridis", extent=[0, 1, 0, 1])
            fig.colorbar(im, ax=ax_heat, label="Annotation Density")
            ax_heat.set_title(f"{task} Heatmap", fontsize=12)
        else:
            ax_heat.axis("off")
            ax_heat.set_title(f"{task} Heatmap", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    if save_path:
        plt.savefig(f"{save_path}/dataset_health.png", dpi=150)
        console.print(f"[green]Plot saved to:[/green] {save_path}")
    else:
        plt.show(block=False)
        console.print(
            "[info]Displaying plot. Close the window or press any key to continue.[/info]"
        )
        if plt.waitforbuttonpress():
            plt.close()


if __name__ == "__main__":
    app()
