"""Commands of ``luxonis_ml data``."""

import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rich.box
from cyclopts import App, Parameter, validators
from loguru import logger
from rich import print
from rich.console import Console
from rich.prompt import Confirm
from rich.rule import Rule
from rich.table import Table

from luxonis_ml.data import (
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
    UpdateMode,
)
from luxonis_ml.data.utils.cli_utils import (
    check_exists,
    get_dataset_info,
    get_tracked_augmentations,
    parse_split_ratio,
    print_info,
)
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.data.utils.plot_utils import (
    plot_class_distribution,
    plot_heatmap,
)
from luxonis_ml.data.utils.visualizations import (
    add_augmentation_footer,
    visualize,
)
from luxonis_ml.enums import DatasetType
from luxonis_ml.typing import (
    Labels,
    LoaderOutput,
    Params,
    TrackedAugmentations,
)

app = App(help="Dataset utilities.")


BucketStorageT: TypeAlias = Annotated[BucketStorage, Parameter(alias="-b")]

# pydoctor parses each string in an annotation as a forward reference,
# and ".json" is not one. The extensions stay out of the annotation.
AUG_CONFIG_VALIDATOR = validators.Path(
    exists=True, ext={".json", ".yaml", ".yml"}
)


@app.command
def info(
    name: str,
    *,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """Print information about a dataset.

    Args:
        name: Name of the dataset.
        bucket_storage: Storage type of the dataset.

    """
    check_exists(name, bucket_storage)
    print_info(LuxonisDataset(name, bucket_storage=bucket_storage))


@app.command(alias=["rm", "remove"])
def delete(
    *names: str,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
    local: Annotated[
        bool,
        Parameter(alias="-l", negative=()),
    ] = False,
    remote: Annotated[
        bool,
        Parameter(alias="-r", negative=()),
    ] = False,
    yes: Annotated[bool, Parameter(alias="-y", negative=())] = False,
):
    """Delete a dataset from local storage, remote storage, or both.

    Args:
        names: Name(s) of the dataset to delete.
        bucket_storage: Storage type of the dataset.
        local: If True, delete the dataset from local storage.
        remote: If True, delete the dataset from remote storage.
        yes: If True, skip confirmation prompt and delete immediately.

    """
    if not names:
        print("[red]At least one dataset name must be provided.[/red]")
        raise SystemExit(1)

    if not local and not remote:
        print(
            "[red]No deletion target specified (local or remote). "
            "Nothing to delete.[/red]"
        )
        raise SystemExit(2)

    for name in names:
        check_exists(name, bucket_storage)

        if bucket_storage is BucketStorage.LOCAL and remote:
            print(
                "[yellow]Warning: You specified remote deletion, "
                "but the bucket is local. "
                "Remote deletion will not be performed.[/yellow]"
            )
            remote = False

        storage = _storage_label(local=local, remote=remote)
        if not yes and not Confirm.ask(
            f"Delete dataset '{name}' with specified bucket "
            f"'{bucket_storage}' from {storage} storage?"
        ):
            continue

        LuxonisDataset(
            name,
            bucket_storage=bucket_storage,
            delete_local=local,
            delete_remote=remote,
        ).delete_dataset(
            delete_local=local,
            delete_remote=remote,
        )
        print(f"Dataset '{name}' deleted from {storage} storage.")


def _storage_label(*, local: bool, remote: bool) -> str:
    """Name the storage that a deletion targets."""
    if local and remote:
        return "local and remote"
    return "local" if local else "remote"


@app.command
def ls(
    *,
    full: Annotated[
        bool,
        Parameter(alias="-f"),
    ] = False,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """List datasets.

    Args:
        full: If True, show full information about each dataset,
            including classes and tasks.
        bucket_storage: Storage type of the dataset.

    """
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


@app.command
def inspect(
    name: str,
    *,
    view: Annotated[list[str] | None, Parameter(alias="-v")] = None,
    aug_config: Annotated[
        Path | None,
        Parameter(alias="-a", validator=AUG_CONFIG_VALIDATOR),
    ] = None,
    size_multiplier: Annotated[
        float,
        Parameter(alias="-s"),
    ] = 1.0,
    ignore_aspect_ratio: Annotated[
        bool,
        Parameter(alias="-i", negative=()),
    ] = False,
    deterministic: Annotated[
        bool,
        Parameter(alias="-d", negative=()),
    ] = False,
    force_update: Annotated[
        bool,
        Parameter(alias="-f", negative=()),
    ] = False,
    blend_all: Annotated[
        bool,
        Parameter(alias="-bl", negative=()),
    ] = False,
    per_instance: Annotated[
        bool,
        Parameter(alias="-pi", negative=()),
    ] = False,
    list_augmentations: Annotated[
        bool,
        Parameter(negative=()),
    ] = False,
    print_sample_metadata: Annotated[
        bool,
        Parameter(negative=()),
    ] = True,
    skeletons: Annotated[
        bool,
        Parameter(negative=()),
    ] = False,
    keypoint_labels: Annotated[
        Literal["none", "numbers", "names", "full"],
        Parameter(),
    ] = "numbers",
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """Inspect images and annotations in a dataset.

    Args:
        name: Name of the dataset to inspect.
        view: Which splits of the dataset to inspect.
            If not provided, the "train" split will be inspected by default.
        aug_config: Path to a JSON or YAML config defining
            augmentations to apply when inspecting the dataset.
            If not provided, no augmentations will be applied.
        size_multiplier: Multiplier for the displayed image size.
        ignore_aspect_ratio: Do not keep the aspect ratio when
            resizing images.
        deterministic: Use deterministic augmentation mode.
        force_update: Force synchronization with remote storage first.
        blend_all: Draw labels belonging to different tasks on the
            same image.
        per_instance: Show each label instance in a separate window.
        list_augmentations: Show the augmentations applied to each
            displayed image. Requires '--aug-config' to be set.
        print_sample_metadata: Print sample metadata for each displayed sample.
        skeletons: Draw keypoint skeleton edges.
        keypoint_labels: The label to draw next to each keypoint.
            ``none`` draws no label, ``numbers`` draws the index,
            ``names`` draws the name, and ``full`` draws the index and
            the name.
        bucket_storage: Storage type of the dataset.

    """
    check_exists(name, bucket_storage)
    if list_augmentations and aug_config is None:
        logger.warning(
            "The '--list-augmentations' option requires "
            "'--aug-config' to be set. No augmentations will be listed."
        )
        list_augmentations = False

    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    if len(dataset) == 0:
        raise ValueError(f"Dataset '{name}' is empty.")

    loader = LuxonisLoader(
        dataset,
        view=view or ["train"],
        update_mode="all" if force_update else "missing",
    )
    if aug_config is not None:
        _apply_aug_config(
            loader,
            aug_config,
            keep_aspect_ratio=not ignore_aspect_ratio,
            deterministic=deterministic,
        )

    draw_names = keypoint_labels in ("names", "full")
    draw = partial(
        visualize,
        classes=dataset.get_classes(),
        blend_all=blend_all,
        categorical_encodings=dataset.get_categorical_encodings(),
        keypoint_metadata=dataset.get_keypoint_metadata()
        if skeletons or draw_names
        else None,
        draw_skeletons=skeletons,
        keypoint_label_mode=keypoint_labels,
    )
    viewer = _SampleViewer(
        draw,
        size_multiplier=size_multiplier,
        per_instance=per_instance,
        list_augmentations=list_augmentations,
        print_sample_metadata=print_sample_metadata,
    )
    for data in loader:
        viewer.show(data)
        if cv2.waitKey() == ord("q"):
            break


def _apply_aug_config(
    loader: LuxonisLoader,
    aug_config: Path,
    *,
    keep_aspect_ratio: bool,
    deterministic: bool,
) -> None:
    """Give a loader the augmentations of a config, sized like its first image."""
    sample_img = loader[0][0]
    img = (
        next(iter(sample_img.values()))
        if isinstance(sample_img, dict)
        else sample_img
    )
    h, w = img.shape[:2]

    loader._augmentations = loader._init_augmentations(
        augmentation_engine="albumentations",
        augmentation_config=aug_config,
        height=h,
        width=w,
        keep_aspect_ratio=keep_aspect_ratio,
        seed=42 if deterministic else None,
    )


_INSTANCE_TASK_TYPES = ("/boundingbox", "/keypoints", "/instance_segmentation")


class _SampleViewer:
    """Show the images of loader samples in OpenCV windows, one per source."""

    def __init__(
        self,
        draw: Callable[[np.ndarray, str, Labels], np.ndarray],
        *,
        size_multiplier: float,
        per_instance: bool,
        list_augmentations: bool,
        print_sample_metadata: bool,
    ):
        self._draw = draw
        self._size_multiplier = size_multiplier
        self._per_instance = per_instance
        self._list_augmentations = list_augmentations
        self._print_sample_metadata = print_sample_metadata
        self._windows: set[str] = set()

    def show(self, data: LoaderOutput) -> None:
        """Show one sample and close the windows of sources it lacks."""
        tracked_augmentations = get_tracked_augmentations(data.metadata)
        if self._print_sample_metadata:
            self._print_metadata(data.metadata, tracked_augmentations)

        current_windows = set(data.images)
        for stale_window in self._windows - current_windows:
            cv2.destroyWindow(stale_window)
        self._windows = current_windows

        footer = None
        if self._list_augmentations:
            footer = list(tracked_augmentations or {})
        for source_name, image in data.images.items():
            self._show_source(
                source_name, self._display_image(image), data.labels, footer
            )

    def _print_metadata(
        self,
        metadata: Params,
        tracked_augmentations: TrackedAugmentations | None,
    ) -> None:
        """Print the sample metadata."""
        if not self._list_augmentations and tracked_augmentations is not None:
            # The runtime parameters of every transformation would bury
            # the record metadata this flag exists to show.
            metadata = {
                k: v for k, v in metadata.items() if k != "augmentations"
            }
        print("Sample metadata:", metadata)

    def _display_image(self, image: np.ndarray) -> np.ndarray:
        """Convert an image to BGR and scale it by the size multiplier."""
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        new_h, new_w = (
            int(h * self._size_multiplier),
            int(w * self._size_multiplier),
        )
        return cv2.resize(image, (new_w, new_h))

    def _show_source(
        self,
        source_name: str,
        image: np.ndarray,
        labels: Labels,
        footer: list[str] | None,
    ) -> None:
        """Show the labels of one source, per instance when asked and possible."""
        cv2.namedWindow(source_name, cv2.WINDOW_NORMAL)
        instance_keys = [k for k in labels if k.endswith(_INSTANCE_TASK_TYPES)]
        if self._per_instance and instance_keys:
            self._show_instances(
                source_name, image, labels, instance_keys, footer
            )
            return

        if self._per_instance:
            print(
                "[yellow]Warning: Per-instance mode is not supported for this dataset. "
                f"Showing all labels in one window for '{source_name}'.[/yellow]"
            )
        self._show_window(
            source_name, self._draw(image, source_name, labels), footer
        )

    def _show_instances(
        self,
        source_name: str,
        image: np.ndarray,
        labels: Labels,
        instance_keys: list[str],
        footer: list[str] | None,
    ) -> None:
        """Show the instances of one source one by one; ``q`` skips the rest."""
        extra_keys = [k for k in labels if k not in instance_keys]
        if extra_keys:
            print(
                f"[yellow]Warning: Ignoring non-instance keys in labels: {extra_keys}[/yellow]"
            )
        n_instances = len(labels[instance_keys[0]])
        for i in range(n_instances):
            instance_labels = {
                k: np.expand_dims(v[i], axis=0)
                for k, v in labels.items()
                if k in instance_keys and len(v) > i
            }
            self._show_window(
                source_name,
                self._draw(image.copy(), source_name, instance_labels),
                footer,
            )
            if cv2.waitKey() == ord("q"):
                break

    @staticmethod
    def _show_window(
        window: str, image: np.ndarray, footer: list[str] | None
    ) -> None:
        """Show an image in its window, with the augmentation footer if any."""
        if footer is not None:
            image = add_augmentation_footer(image, footer)
        cv2.resizeWindow(window, image.shape[1], image.shape[0])
        cv2.imshow(window, image)


@app.command
def export(
    name: str,
    *,
    save_dir: Annotated[
        str | None,
        Parameter(alias="-s"),
    ] = None,
    dataset_type: Annotated[
        DatasetType,
        Parameter(
            name="--type",
            alias="-t",
        ),
    ] = DatasetType.NATIVE,
    delete_existing: Annotated[
        bool,
        Parameter(
            name="--delete",
            alias="-d",
            negative=(),
        ),
    ] = False,
    max_partition_size_gb: Annotated[
        float | None,
        Parameter(alias="-m"),
    ] = None,
    zip: bool = True,
    ldf_version: str | None = None,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """Export a Luxonis dataset to disk.

    Args:
        name: Name of the dataset to export.
        save_dir: Directory where the exported dataset will be
            saved. If not provided, a directory with the same name as the
            dataset will be created in the current working directory.
        dataset_type: Format of the exported dataset.
        delete_existing: If True, delete any existing directory at
            the save location before exporting.
        max_partition_size_gb: Maximum size of each
            partition in GB. If not provided, no partitioning will be done.
        zip: If ``True``, the exported dataset will be zipped into one
            archive, or into one archive for each partition. If
            ``False``, the dataset will be exported as a directory with
            the specified structure.
        ldf_version: LDF version to write, so an older luxonis-ml can
            read the export: ``2.1`` for luxonis-ml 0.9 to 0.10 and
            ``2.0`` for luxonis-ml 0.6 to 0.8. Only valid with
            ``--type native``. Downgrading is lossy and warns about what
            it drops.
        bucket_storage: Storage type of the dataset.

    """
    save_dir = save_dir or name
    if delete_existing and Path(save_dir).exists():
        shutil.rmtree(save_dir)
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    dataset.export(
        save_dir,
        dataset_type=dataset_type,
        max_partition_size_gb=max_partition_size_gb,
        zip_output=zip,
        ldf_version=ldf_version,
    )


@app.command
def parse(
    dataset: Annotated[
        str,
        Parameter(alias="--dataset-dir"),
    ],
    *,
    name: Annotated[
        str | None,
        Parameter(alias="-n"),
    ] = None,
    dataset_type: Annotated[
        DatasetType | None,
        Parameter(
            name="--type",
            alias="-t",
        ),
    ] = None,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
    delete_local: Annotated[
        bool,
        Parameter(
            name="--delete",
            alias="-d",
            negative=(),
        ),
    ] = False,
    save_dir: Annotated[
        Path | None,
        Parameter(alias="-s"),
    ] = None,
    task_name: Annotated[
        str | None,
        Parameter(alias="-tn"),
    ] = None,
    log_all_warnings: bool = False,
    split_ratio: Annotated[
        str | None,
        Parameter(alias="-sr"),
    ] = None,
    train: float | None = None,
    val: float | None = None,
    test: float | None = None,
):
    """Parse a directory with data and create a Luxonis dataset.

    Args:
        dataset: Path or URL to the dataset.
        name: Name of the dataset.
            If not provided, the directory name will be used.
        dataset_type: Type of the dataset.
            If not provided, the parser will attempt to detect it.
        bucket_storage: Storage type of the dataset.
        delete_local: If True, delete any existing local
            dataset with the same name before parsing.
        save_dir: If dataset_dir is a remote URL,
            this is the local directory where the dataset will
            be downloaded before parsing. If not provided,
            the dataset will be downloaded to the current working directory.
        task_name: Task name to use for all records
            parsed from this dataset.
        log_all_warnings: Log all skipped annotation warnings
            instead of capping the output at 10.
        split_ratio: A string representation of a Python list
            specifying the split ratios for train, val, and test sets.
            Deprecated in favor of ``--train``, ``--val``, and ``--test``.
        train: Ratio or count of records to assign
            to the training set. Can be used together with
            ``--val`` and ``--test``. If only some of these options
            are provided, the remaining split(s) receive an equal share
            of the leftover records (only supported for ratios, not counts).
        val: Ratio or count of records to assign
            to the validation set.
        test: Ratio or count of records to assign
            to the test set.

    """
    parser = LuxonisParser(
        dataset,
        dataset_name=name,
        dataset_type=dataset_type,
        delete_local=delete_local,
        save_dir=save_dir,
        task_name=task_name,
        full_warnings=log_all_warnings,
        bucket_storage=bucket_storage,
    )

    print()
    print(Rule())
    print()
    print_info(
        parser.parse(
            split_ratios=parse_split_ratio(split_ratio, train, val, test)
        )
    )


@app.command
def health(
    name: str,
    *,
    view: Annotated[
        str | None,
        Parameter(alias="-v"),
    ] = None,
    sample_size: Annotated[
        int | None,
        Parameter(alias="-n"),
    ] = None,
    save_dir: Annotated[
        str | None,
        Parameter(alias="-s"),
    ] = None,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """Plot class distributions and heatmaps for every task type and
    corresponding task name in the dataset.

    Also checks for files with missing annotations, files that share the
    same UUIDs, and files with duplicate annotations.

    Args:
        name: Name of the dataset to inspect.
        view: Which split of the dataset to inspect.
            If not provided, all splits will be inspected.
        sample_size: Number of annotation rows to sample
            from the dataset for calculating statistics and plots.
            If not provided, all annotations will be used.
        save_dir: Directory where the generated plots
            will be saved. If not provided, the plots will be displayed
            interactively instead of being saved.
        bucket_storage: Storage type of the dataset.

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
        _print_table(
            console,
            "Duplicate UUIDs",
            {"UUID": "magenta", "Files": "cyan"},
            [
                (item["uuid"], ", ".join(item["files"]))
                for item in duplicate_uuids
            ],
            row_style="red",
        )

    if duplicate_annotations:
        _print_table(
            console,
            "Duplicate Annotations",
            {
                "File Name": "cyan",
                "Task Name": "magenta",
                "Task Type": "magenta",
                "Annotation": "yellow",
                "Count": "green",
            },
            [
                (
                    item["file_name"],
                    item["task_name"],
                    item["task_type"],
                    str(item["annotation"]),
                    str(item["count"]),
                )
                for item in duplicate_annotations
            ],
            row_style="red",
        )

    if missing_annotations:
        _print_table(
            console,
            "Files With Missing Annotations",
            {"File Name": "cyan"},
            [(file,) for file in missing_annotations],
            row_style="yellow",
        )

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

    if missing_annotations or duplicate_uuids or duplicate_annotations:
        console.print(
            "[bold red]Dataset is unhealthy![/bold red] "
            "Run [green]luxonis_ml data sanitize[/green] "
            "to automatically remove duplicates and missing entries."
        )

    all_task_names = sorted(
        set(stats["class_distributions"].keys())
        | set(stats["heatmaps"].keys())
    )
    if not all_task_names:
        console.print("[info]No plots to display.[/info]")
        return

    for task_name in all_task_names:
        _plot_task_health(
            console,
            task_name,
            stats["class_distributions"].get(task_name, {}),
            stats["heatmaps"].get(task_name, {}),
            save_dir,
        )


def _print_table(
    console: Console,
    title: str,
    columns: dict[str, str],
    rows: list[tuple[str, ...]],
    *,
    row_style: str,
) -> None:
    """Print a rounded table with one style per column."""
    table = Table(title=title, box=rich.box.ROUNDED, row_styles=[row_style])
    for name, style in columns.items():
        table.add_column(name, style=style)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def _plot_task_health(
    console: Console,
    task_name: str,
    class_dist_by_type: dict[str, list],
    heatmaps_by_type: dict[str, np.ndarray],
    save_dir: str | None,
) -> None:
    """Plot the class distribution and heatmap of each task type.

    The figure is saved under ``save_dir`` when given, else shown.
    """
    all_task_types = sorted(
        set(class_dist_by_type.keys()) | set(heatmaps_by_type.keys())
    )
    if not all_task_types:
        console.print(f"[info]No plots for task name: {task_name}[/info]")
        return

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


@app.command
def push(
    name: str,
    *,
    bucket_storage: BucketStorage,
    force: Annotated[
        bool,
        Parameter(alias="-f", negative=()),
    ] = False,
):
    """Push a local dataset to cloud storage.

    Args:
        name: Name of the dataset to push.
        bucket_storage: Cloud storage type to push to.
            Cannot be LOCAL.
        force: If True, push all media files even
            if they already exist in the target cloud storage.

    """
    check_exists(name, BucketStorage.LOCAL)
    dataset = LuxonisDataset(name, bucket_storage=BucketStorage.LOCAL)

    if bucket_storage == BucketStorage.LOCAL:
        print(
            "[red]Cannot push to LOCAL storage. Please specify a cloud target."
        )
        raise SystemExit(1)

    if LuxonisDataset.exists(
        name, bucket_storage=bucket_storage
    ) and not Confirm.ask(
        f"Dataset '{name}' already exists in {bucket_storage} bucket. "
        "If you are unsure about the dataset, please delete it from "
        "the cloud storage and try again. Do you want to overwrite it?"
    ):
        raise SystemExit

    print(f"Pushing dataset '{name}' to {bucket_storage.value} storage...")

    update_mode = UpdateMode.ALL if force else UpdateMode.MISSING
    dataset.push_to_cloud(
        bucket_storage=bucket_storage, update_mode=update_mode
    )

    print(
        f"[green]Dataset '{name}' successfully pushed to {bucket_storage.value}."
    )


@app.command
def pull(
    name: str,
    *,
    force: Annotated[
        bool,
        Parameter(alias="-f", negative=()),
    ] = False,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """Pull a remote dataset to local storage.

    Args:
        name: Name of the dataset to pull.
        force: If True, pull all media files even
            if they already exist locally.
        bucket_storage: Cloud storage type to pull from.
            Cannot be LOCAL.

    """
    if bucket_storage == BucketStorage.LOCAL:
        print(
            "[red]Cannot pull from LOCAL storage. Please specify a cloud source."
        )
        raise SystemExit(1)

    if not LuxonisDataset.exists(name, bucket_storage=bucket_storage):
        print(
            f"[red]Dataset '{name}' does not exist in {bucket_storage.value} storage."
        )
        raise SystemExit(1)

    print(f"Pulling dataset '{name}' from {bucket_storage.value} storage...")

    update_mode = UpdateMode.ALL if force else UpdateMode.MISSING
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    dataset.pull_from_cloud(update_mode=update_mode)

    print(
        f"[green]Dataset '{name}' successfully pulled from {bucket_storage.value}."
    )


@app.command
def clone(
    name: str,
    new_name: str,
    *,
    push: Annotated[
        bool,
        Parameter(alias="-p", negative=()),
    ] = True,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
    split: Annotated[
        list[str] | None,
        Parameter(alias="-s"),
    ] = None,
    team_id: Annotated[
        str | None,
        Parameter(alias="-t"),
    ] = None,
):
    """Clone an existing dataset with a new name.

    Optionally push it to cloud storage if it is a remote dataset.

    Args:
        name: Name of the source dataset to clone.
        new_name: Name of the new cloned dataset.
        push: If True, upload the newly cloned dataset to cloud storage.
        bucket_storage: Storage type of the source dataset.
        split: List of split names to clone.
            If not provided, all splits will be cloned.
            Example: ``--split train --split val`` to clone only the "train" and "val" splits.
        team_id: Team ID to use for the new dataset.

    """
    check_exists(name, bucket_storage)

    if LuxonisDataset.exists(
        new_name, bucket_storage=BucketStorage.LOCAL
    ) and not Confirm.ask(
        f"Dataset '{new_name}' already exists locally. Overwrite it?"
    ):
        raise SystemExit

    print(f"Cloning dataset '{name}' to '{new_name}'...")
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    dataset.clone(
        new_dataset_name=new_name,
        push_to_cloud=push,
        splits_to_clone=split,
        team_id=team_id,
    )
    print(f"[green]Dataset '{name}' successfully cloned to '{new_name}'.")


@app.command
def merge(
    source_name: str,
    target_name: str,
    *,
    new_name: Annotated[
        str | None,
        Parameter(alias="-n"),
    ] = None,
    splits_to_merge: Annotated[
        str | None,
        Parameter(
            name="--split",
            alias="-s",
        ),
    ] = None,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
    team_id: Annotated[
        str | None,
        Parameter(alias="-t"),
    ] = None,
):
    """Merge two datasets stored in the same type of bucket.

    Args:
        source_name: Name of the source dataset to merge from.
        target_name: Name of the target dataset to merge into.
        new_name: If provided, the name of the new merged dataset.
            If not provided, the source dataset will be
            merged into the target dataset in place.
        splits_to_merge: Comma-separated list of split
            names to merge. If not provided, all splits will be merged.
        bucket_storage: Storage type for both datasets.
        team_id: Team ID to use for the new dataset.
            If not provided, the team ID of the target dataset will be used.

    """
    check_exists(source_name, bucket_storage)
    check_exists(target_name, bucket_storage)

    inplace = new_name is None
    if inplace and not Confirm.ask(
        f"This will merge dataset '{source_name}' "
        f"into '{target_name}'. Continue?"
    ):
        raise SystemExit

    if (
        not inplace
        and LuxonisDataset.exists(new_name, bucket_storage=bucket_storage)
        and not Confirm.ask(
            f"Dataset '{new_name}' already exists in "
            f"{bucket_storage.value} bucket. Overwrite it?"
        )
    ):
        raise SystemExit

    if splits_to_merge:
        split_list = [
            s.strip() for s in splits_to_merge.split(",") if s.strip()
        ]
    else:
        split_list = None

    source_dataset = LuxonisDataset(source_name, bucket_storage=bucket_storage)
    target_dataset = LuxonisDataset(target_name, bucket_storage=bucket_storage)

    operation = "in place" if inplace else ""
    print(f"Merging dataset '{source_name}' with '{target_name}' {operation}")

    _ = target_dataset.merge_with(
        source_dataset,
        inplace=inplace,
        new_dataset_name=new_name,
        splits_to_merge=split_list,
        team_id=team_id,
    )

    if inplace:
        print(
            f"[green]Dataset '{source_name}' "
            f"successfully merged into '{target_name}'."
        )
    else:
        print(
            f"[green]Datasets merged successfully "
            f"into new dataset '{new_name}'."
        )


@app.command
def sanitize(
    name: str,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """Remove duplicate annotations and duplicate files from the
    dataset.

    Args:
        name: Name of the dataset to sanitize.
        bucket_storage: Storage type of the dataset.

    """
    check_exists(name, bucket_storage)
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    dataset.remove_duplicates()
    print(f"[green]Duplicates removed from dataset '{name}'.")


if __name__ == "__main__":
    app()
