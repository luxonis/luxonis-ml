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

from luxonis_ml.data import (
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
    UpdateMode,
)
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.data.utils.plot_utils import (
    plot_class_distribution,
    plot_heatmap,
)
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

    @group()
    def get_sizes_panel() -> Iterator[RenderableType]:
        if splits is not None:
            total_files = len(dataset)
            for split, files in splits.items():
                split_size = len(files)
                percentage = (
                    (split_size / total_files * 100) if total_files > 0 else 0
                )
                yield f"[magenta b]{split}: [not b cyan]{split_size:,} [dim]({percentage:.1f}%)[/dim]"
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
    local: bool = typer.Option(
        False,
        "--local/--no-local",
        help="Delete the dataset from local storage.",
    ),
    remote: bool = typer.Option(
        False,
        "--remote/--no-remote",
        help="Delete the dataset from remote storage.",
    ),
):
    """Deletes a dataset from local storage or remote storage (or both),
    based on which options are passed."""
    check_exists(name, bucket_storage)

    if bucket_storage is BucketStorage.LOCAL and remote:
        print(
            "[yellow]Warning: You specified remote deletion, but the bucket is local. "
            "Remote deletion will not be performed.[/yellow]"
        )
        remote = False

    where = " and ".join(
        filter(None, ["local" if local else "", "remote" if remote else ""])
    )

    if not where:
        print(
            "[yellow]No deletion target specified (local or remote). Nothing to delete.[/yellow]"
        )
        raise typer.Exit

    if not Confirm.ask(
        f"Delete dataset '{name}' with specified bucket '{bucket_storage}' from {where} storage?"
    ):
        raise typer.Exit

    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    dataset.delete_dataset(
        delete_local=local,
        delete_remote=remote,
    )

    print(
        f"Dataset '{name}' deleted from: "
        f"{'local ' if local else ''}"
        f"{'remote ' if remote else ''}"
        f"storage."
    )


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
    force_update: Annotated[
        bool,
        typer.Option(
            ...,
            "--force-update",
            "-f",
            help="Force synchronization of the dataset with the "
            "remote storage. Only relevant for remote datasets.",
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
    per_instance: Annotated[
        bool,
        typer.Option(
            ...,
            "--per-instance",
            "-pi",
            help="Show each label instance in a separate window.",
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
    loader = LuxonisLoader(
        dataset,
        view=view,
        update_mode="all" if force_update else "missing",
    )

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
        instance_keys = [
            "/boundingbox",
            "/keypoints",
            "/instance_segmentation",
        ]
        matched_instance_keys = [
            k for k in labels if any(k.endswith(ik) for ik in instance_keys)
        ]
        if per_instance and matched_instance_keys:
            extra_keys = [k for k in labels if k not in matched_instance_keys]
            if extra_keys:
                print(
                    f"[yellow]Warning: Ignoring non-instance keys in labels: {extra_keys}[/yellow]"
                )
            n_instances = len(labels[matched_instance_keys[0]])
            for i in range(n_instances):
                instance_labels = {
                    k: np.expand_dims(v[i], axis=0)
                    for k, v in labels.items()
                    if k in matched_instance_keys and len(v) > i
                }
                instance_image = visualize(
                    image.copy(), instance_labels, classes, blend_all=blend_all
                )
                cv2.imshow("image", instance_image)
                if cv2.waitKey() == ord("q"):
                    break
        else:
            if per_instance:
                print(
                    "[yellow]Warning: Per-instance mode is not supported for this dataset. "
                    "Showing all labels in one window.[/yellow]"
                )
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
    ] = "native",  # type: ignore
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
    delete_local: Annotated[
        bool,
        typer.Option(
            ...,
            "--delete",
            "-d",
            help="If an existing local dataset with the same name should "
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
        delete_local=delete_local,
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
    view: Optional[str] = typer.Option(
        None,
        "--view",
        "-v",
        help="Which splits of the dataset to inspect. If not provided, all dataset will be used.",
        show_default=False,
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample-size",
        "-n",
        help="Number of annotation rows to sample from the dataset. Note that each task type annotation is in a separate row.",
        show_default=False,
    ),
    save_dir: Optional[str] = typer.Option(
        None,
        "--save-dir",
        "-s",
        help="Directory where the plots should be saved. "
        "If not provided, the plots will be displayed.",
        show_default=False,
    ),
    bucket_storage: BucketStorage = bucket_option,
):
    """Plots class distributions and heatmaps for every task type and
    corresponding task name in the dataset.

    Also checks for files with missing annotations, files that share the
    same UUIDs, and files with duplicate annotations.
    """
    check_exists(name, bucket_storage)
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    stats = dataset.get_statistics(sample_size=sample_size, view=view)
    console = Console()

    missing_annotations = stats["missing_annotations"]
    duplicate_uuids = stats["duplicates"]["duplicate_uuids"]
    duplicate_annotations = stats.get("duplicates", {}).get(
        "duplicate_annotations", []
    )

    if duplicate_uuids:
        duuid_table = Table(
            title="Duplicate UUIDs", box=rich.box.ROUNDED, row_styles=["red"]
        )
        duuid_table.add_column("UUID", style="magenta")
        duuid_table.add_column("Files", style="cyan")

        for item in duplicate_uuids:
            duuid_table.add_row(item["uuid"], ", ".join(item["files"]))

        console.print(duuid_table)

    if duplicate_annotations:
        dann_table = Table(
            title="Duplicate Annotations",
            box=rich.box.ROUNDED,
            row_styles=["red"],
        )
        dann_table.add_column("File Name", style="cyan")
        dann_table.add_column("Task Name", style="magenta")
        dann_table.add_column("Task Type", style="magenta")
        dann_table.add_column("Annotation", style="yellow")
        dann_table.add_column("Count", style="green")

        for item in duplicate_annotations:
            dann_table.add_row(
                item["file_name"],
                item["task_name"],
                item["task_type"],
                str(item["annotation"]),
                str(item["count"]),
            )

        console.print(dann_table)

    if missing_annotations:
        missing_table = Table(
            title="Files With Missing Annotations",
            box=rich.box.ROUNDED,
            row_styles=["yellow"],
        )
        missing_table.add_column("File Name", style="cyan")

        for file in missing_annotations:
            missing_table.add_row(file)

        console.print(missing_table)

    summary_table = Table(
        title="Dataset Health Summary",
        box=rich.box.ROUNDED,
        show_header=False,
        row_styles=["cyan", "yellow", "green"],
    )
    summary_table.add_column("Metric")
    summary_table.add_column("Count")
    summary_table.add_row(
        "Files with missing annotations", str(len(missing_annotations))
    )
    summary_table.add_row(
        "Files with duplicate UUIDs", str(len(duplicate_uuids))
    )
    summary_table.add_row(
        "Files with duplicate annotations", str(len(duplicate_annotations))
    )

    console.print(summary_table)

    all_task_names = sorted(
        set(stats["class_distributions"].keys())
        | set(stats["heatmaps"].keys())
    )
    if not all_task_names:
        console.print("[info]No plots to display.[/info]")
        return

    for task_name in all_task_names:
        class_dist_by_type = stats["class_distributions"].get(task_name, {})
        heatmaps_by_type = stats["heatmaps"].get(task_name, {})
        all_task_types = sorted(
            set(class_dist_by_type.keys()) | set(heatmaps_by_type.keys())
        )

        if not all_task_types:
            console.print(f"[info]No plots for task name: {task_name}[/info]")
            continue

        nrows = len(all_task_types)
        square_size = 4
        fig, axs = plt.subplots(
            nrows, 2, figsize=(square_size * 2, square_size * nrows)
        )
        if task_name != "":
            fig.suptitle(f"Task Name: {task_name}", fontsize=14)

        if nrows == 1:
            axs = [axs]

        for i, task_type in enumerate(all_task_types):
            plot_class_distribution(
                axs[i][0], task_type, class_dist_by_type.get(task_type, [])
            )
            plot_heatmap(
                axs[i][1],
                fig,  # type: ignore
                task_type,
                heatmaps_by_type.get(task_type),  # type: ignore
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.5)

        if save_dir:
            fig.savefig(f"{save_dir}/dataset_health_{task_name}.png", dpi=150)  # type: ignore
            plt.close(fig)
        else:
            plt.show(block=False)
            if plt.waitforbuttonpress():
                plt.close(fig)


@app.command()
def push(
    name: DatasetNameArgument,
    force_update: Annotated[
        bool,
        typer.Option(
            ...,
            "--force-update",
            "-f",
            help="Force pushing all media files, even if they already exist in the cloud. Annotations and metadata will always be pushed.",
        ),
    ] = False,
    target_bucket_storage: BucketStorage = bucket_option,
):
    """Push a local dataset to cloud storage."""
    check_exists(name, BucketStorage.LOCAL)
    dataset = LuxonisDataset(name, bucket_storage=BucketStorage.LOCAL)

    if target_bucket_storage == BucketStorage.LOCAL:
        print(
            "[red]Cannot push to LOCAL storage. Please specify a cloud target."
        )
        raise typer.Exit(1)

    if LuxonisDataset.exists(
        name, bucket_storage=target_bucket_storage
    ) and not Confirm.ask(
        f"Dataset '{name}' already exists in {target_bucket_storage} bucket. If you are unsure about the dataset, please delete it from the cloud storage and try again. Do you want to overwrite it?"
    ):
        raise typer.Exit

    print(
        f"Pushing dataset '{name}' to {target_bucket_storage.value} storage..."
    )

    update_mode = UpdateMode.ALL if force_update else UpdateMode.MISSING
    dataset.push_to_cloud(
        bucket_storage=target_bucket_storage, update_mode=update_mode
    )

    print(
        f"[green]Dataset '{name}' successfully pushed to {target_bucket_storage.value}."
    )


@app.command()
def pull(
    name: DatasetNameArgument,
    force_update: Annotated[
        bool,
        typer.Option(
            ...,
            "--force-update",
            "-f",
            help="Force pulling all media files, even if they already exist locally.",
        ),
    ] = False,
    bucket_storage: BucketStorage = bucket_option,
):
    """Pull a remote dataset to local storage."""
    if bucket_storage == BucketStorage.LOCAL:
        print(
            "[red]Cannot pull from LOCAL storage. Please specify a cloud source."
        )
        raise typer.Exit(1)

    if not LuxonisDataset.exists(name, bucket_storage=bucket_storage):
        print(
            f"[red]Dataset '{name}' does not exist in {bucket_storage.value} storage."
        )
        raise typer.Exit

    print(f"Pulling dataset '{name}' from {bucket_storage.value} storage...")

    update_mode = UpdateMode.ALL if force_update else UpdateMode.MISSING
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    dataset.pull_from_cloud(update_mode=update_mode)

    print(
        f"[green]Dataset '{name}' successfully pulled from {bucket_storage.value}."
    )


@app.command()
def clone(
    name: DatasetNameArgument,
    new_name: Annotated[
        str,
        typer.Argument(
            ..., help="Name of the new dataset.", show_default=False
        ),
    ],
    push_to_cloud: Annotated[
        bool,
        typer.Option(
            "--push/--no-push",
            help="Whether to upload the newly cloned remote dataset back to cloud storage.",
            show_default=True,
        ),
    ] = True,
    bucket_storage: BucketStorage = bucket_option,
):
    """Clone an existing dataset with a new name.

    Optionally push it to cloud storage if it is a remote dataset.
    """

    check_exists(name, bucket_storage)

    if LuxonisDataset.exists(
        new_name, bucket_storage=BucketStorage.LOCAL
    ) and not Confirm.ask(
        f"Dataset '{new_name}' already exists locally. Overwrite it?"
    ):
        raise typer.Exit

    print(f"Cloning dataset '{name}' to '{new_name}'...")
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    dataset.clone(new_dataset_name=new_name, push_to_cloud=push_to_cloud)
    print(f"[green]Dataset '{name}' successfully cloned to '{new_name}'.")


@app.command()
def merge(
    source_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Name of the source dataset.",
            autocompletion=complete_dataset_name,
        ),
    ],
    target_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Name of the target dataset.",
            autocompletion=complete_dataset_name,
        ),
    ],
    new_name: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--new-name",
            "-n",
            help="Name for the new merged dataset. If not provided, will merge into the target dataset.",
            show_default=False,
        ),
    ] = None,
    bucket_storage: BucketStorage = bucket_option,
):
    """Merge two datasets stored in the same type of bucket."""
    check_exists(source_name, bucket_storage)
    check_exists(target_name, bucket_storage)

    inplace = new_name is None
    if inplace and not Confirm.ask(
        f"This will merge dataset '{source_name}' into '{target_name}'. Continue?"
    ):
        raise typer.Exit

    if (
        not inplace
        and LuxonisDataset.exists(new_name, bucket_storage=bucket_storage)
        and not Confirm.ask(
            f"Dataset '{new_name}' already exists in {bucket_storage.value} bucket. Overwrite it?"
        )
    ):
        raise typer.Exit

    source_dataset = LuxonisDataset(source_name, bucket_storage=bucket_storage)
    target_dataset = LuxonisDataset(target_name, bucket_storage=bucket_storage)

    operation = "into" if inplace else "creating new dataset"
    print(
        f"Merging dataset '{source_name}' with '{target_name}', {operation}..."
    )

    _ = target_dataset.merge_with(
        source_dataset, inplace=inplace, new_dataset_name=new_name
    )

    if inplace:
        print(
            f"[green]Dataset '{source_name}' successfully merged into '{target_name}'."
        )
    else:
        print(
            f"[green]Datasets merged successfully into new dataset '{new_name}'."
        )


if __name__ == "__main__":
    app()
