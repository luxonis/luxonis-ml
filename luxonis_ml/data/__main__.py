import math
import shutil
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
from luxonis_ml.data.utils.augmentations_collector import (
    AugmentationsCollector,
)
from luxonis_ml.data.utils.cli_utils import (
    check_exists,
    get_dataset_info,
    parse_split_ratio,
    print_info,
)
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.data.utils.plot_utils import (
    plot_class_distribution,
    plot_heatmap,
)
from luxonis_ml.data.utils.task_utils import get_task_name, get_task_type
from luxonis_ml.enums import DatasetType

app = App(help="Dataset utilities.")


BucketStorageT: TypeAlias = Annotated[BucketStorage, Parameter(alias="-b")]


def _screen_size() -> tuple[int, int] | None:
    """Best-effort screen resolution in pixels, or ``None`` if unavailable.

    Uses Tk (standard library) to query the display; returns ``None`` on any
    failure (headless session, Tk missing), letting callers fall back to
    unscaled, un-centered display.
    """
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
    except Exception:
        return None
    else:
        return size


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
        Parameter(alias="-l", negative=""),
    ] = False,
    remote: Annotated[
        bool,
        Parameter(alias="-r", negative=""),
    ] = False,
    yes: Annotated[bool, Parameter(alias="-y", negative="")] = False,
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

        storage = (
            "local and remote"
            if local and remote
            else "local"
            if local
            else "remote"
        )
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
        Parameter(
            alias="-a",
            validator=validators.Path(
                exists=True, ext={".json", ".yaml", ".yml"}
            ),
        ),
    ] = None,
    size_multiplier: Annotated[
        float | None,
        Parameter(alias="-s"),
    ] = None,
    ignore_aspect_ratio: Annotated[
        bool,
        Parameter(alias="-i", negative=""),
    ] = False,
    deterministic: Annotated[
        bool,
        Parameter(alias="-d", negative=""),
    ] = False,
    force_update: Annotated[
        bool,
        Parameter(alias="-f", negative=""),
    ] = False,
    blend_all: Annotated[
        bool,
        Parameter(alias="-bl", negative=""),
    ] = False,
    per_instance: Annotated[
        bool,
        Parameter(alias="-pi", negative=""),
    ] = False,
    list_augmentations: Annotated[
        bool,
        Parameter(negative=""),
    ] = False,
    skeletons: Annotated[
        bool,
        Parameter(negative=""),
    ] = False,
    keypoint_labels: Annotated[
        Literal["none", "numbers", "names", "full"],
        Parameter(),
    ] = "none",
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
        size_multiplier: Multiplier for the displayed image size. If
            not given, the image is scaled automatically to fit the
            screen (accounting for the metadata panel) and centered.
        ignore_aspect_ratio: Do not keep the aspect ratio when
            resizing images.
        deterministic: Use deterministic augmentation mode.
        force_update: Force synchronization with remote storage first.
        blend_all: Draw labels belonging to different tasks on the
            same image.
        per_instance: Show each label instance in a separate window.
        list_augmentations: Show the augmentations applied to each
            displayed image. Requires '--aug-config' to be set.
        skeletons: Draw keypoint skeleton edges.
        keypoint_labels: Specify how to draw keypoint labels.
        bucket_storage: Storage type of the dataset.

    """
    check_exists(name, bucket_storage)

    view = view or ["train"]
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)

    if len(dataset) == 0:
        raise ValueError(f"Dataset '{name}' is empty.")

    loader = LuxonisLoader(
        dataset,
        view=view,
        update_mode="all" if force_update else "missing",
    )

    if aug_config is not None:
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
            keep_aspect_ratio=not ignore_aspect_ratio,
            seed=42 if deterministic else None,
        )

    if list_augmentations:
        if aug_config is None:
            logger.warning(
                "--list-augmentations was set but --aug-config was not "
                "provided. No augmentations will be shown."
            )
            get_applied_augmentations = list
        elif loader._augmentations is not None:
            collector = AugmentationsCollector(
                loader._augmentations,  # type: ignore
                aug_config,
            )
            get_applied_augmentations = collector.get_applied_augmentations
        else:
            get_applied_augmentations = list
    else:
        get_applied_augmentations = list

    classes = dataset.get_classes()
    categorical_encodings = dataset.get_categorical_encodings()
    keypoint_skeletons = dataset.get_skeletons()

    try:
        from luxonis_ml.data.loaders.label_converter import (
            loader_output_to_records,
        )
        from luxonis_ml.vizlab import (
            Image,
            Palette,
            Skeleton,
            VizConfig,
            grid,
            visualize_record,
        )
        from luxonis_ml.vizlab.ldf import detection_to_annotations
    except ImportError as e:
        raise SystemExit(
            "Visualization requires the 'viz' extra. "
            "Install it with `pip install luxonis-ml[viz]`."
        ) from e

    # Pre-seed a palette with every class name (in id order per task) so class
    # colors stay stable across samples.
    class_names: list[str] = []
    for task_classes in classes.values():
        for class_name in sorted(task_classes, key=task_classes.__getitem__):
            if class_name not in class_names:
                class_names.append(class_name)
    config = VizConfig(
        palette=Palette(class_names),
        skeletons={
            task: Skeleton.from_ldf(labels, edges)
            for task, (labels, edges) in keypoint_skeletons.items()
        },
        keypoint_label_mode=keypoint_labels,
        draw_skeletons=skeletons,
    )

    def build_panel(sample_labels: dict, sample_metadata: dict) -> dict:
        panel: dict = dict(sample_metadata) if sample_metadata else {}
        arrays = {
            get_task_name(k): list(v.shape)
            for k, v in sample_labels.items()
            if get_task_type(k) == "array"
        }
        if arrays:
            panel["arrays"] = arrays
        if list_augmentations:
            applied = get_applied_augmentations()
            if applied:
                panel["augmentations"] = list(applied)
        return panel

    screen = _screen_size()
    # The metadata panel is drawn at a fixed pixel width, independent of the
    # image scale, so reserve horizontal room for it when fitting to the screen.
    panel_reserve = 400.0

    def target_size(
        width: int,
        height: int,
        *,
        reserve: float = 0.0,
        cols: int = 1,
        rows: int = 1,
    ) -> tuple[int, int] | None:
        """Display size ``(w, h)`` for a source image, or ``None`` to keep native.

        vizlab paints mask fills at the source resolution and scales once to this
        size, then draws strokes and labels crisply at it, so annotations stay
        sharp without resampling every mask. ``reserve`` leaves horizontal room
        for the fixed-width metadata panel; ``cols``/``rows`` divide the budget
        when several tiles share the screen (grid view).
        """
        if size_multiplier is not None:
            scale = size_multiplier
        elif screen is not None:
            # Fit within 90% of the screen (leaving room for the panel/tiles).
            avail_w = max(1.0, 0.9 * screen[0] - reserve) / cols
            avail_h = (0.9 * screen[1]) / rows
            scale = min(avail_w / width, avail_h / height)
        else:
            return None
        if scale == 1.0:
            return None
        return (max(1, round(width * scale)), max(1, round(height * scale)))

    def show(source_name: str, viz: Image) -> None:
        out = viz.to_numpy(mode="bgr")
        out_h, out_w = out.shape[:2]
        win_w, win_h = out_w, out_h
        if screen is not None and size_multiplier is None:
            # Safety net: never present a window larger than the screen (e.g. a
            # multi-tile grid). The image was already drawn at fit size, so this
            # only engages for composites that still overflow.
            fit = min(0.9 * screen[0] / out_w, 0.9 * screen[1] / out_h, 1.0)
            win_w, win_h = round(out_w * fit), round(out_h * fit)
        cv2.resizeWindow(source_name, win_w, win_h)
        if screen is not None:
            # Center the window on the screen.
            cv2.moveWindow(
                source_name,
                max(0, (screen[0] - win_w) // 2),
                max(0, (screen[1] - win_h) // 2),
            )
        cv2.imshow(source_name, out)

    prev_windows = set()

    for data in loader:
        images_dict = data.images

        current_windows = set(images_dict.keys())
        for stale_window in prev_windows - current_windows:
            cv2.destroyWindow(stale_window)

        records = loader_output_to_records(
            data.labels,
            classes=classes,
            categorical_encodings=categorical_encodings,
        )
        panel = build_panel(data.labels, data.metadata)
        instances = [
            (record.task_name, detection)
            for record in records.values()
            for detection in record._annotations()
            if detection.boundingbox is not None
            or detection.keypoints is not None
            or detection.instance_segmentation is not None
        ]

        quit_requested = False
        for source_name, image in images_dict.items():
            image = image.astype(np.uint8)
            height, width = image.shape[:2]
            cv2.namedWindow(source_name, cv2.WINDOW_NORMAL)

            if per_instance and instances:
                # Per-instance windows carry no panel, so reserve no room for it.
                size = target_size(width, height)
                for task_name, detection in instances:
                    viz = Image(image, config=config).render_at(size)
                    for annotation in detection_to_annotations(
                        detection, config, task_name=task_name
                    ):
                        viz.add(annotation)
                    show(source_name, viz)
                    if cv2.waitKey() == ord("q"):
                        quit_requested = True
                        break
                if quit_requested:
                    break
                continue

            if per_instance:
                print(
                    "[yellow]Warning: Per-instance mode is not supported for "
                    f"this dataset. Showing all labels for '{source_name}'.[/yellow]"
                )

            reserve = panel_reserve if panel else 0.0
            if blend_all or len(records) <= 1:
                size = target_size(width, height, reserve=reserve)
                viz = Image(image, config=config).render_at(size)
                for record in records.values():
                    for detection in record._annotations():
                        for annotation in detection_to_annotations(
                            detection, config, task_name=record.task_name
                        ):
                            viz.add(annotation)
            else:
                # Fit the whole grid of tiles within the screen.
                cols = max(1, math.ceil(math.sqrt(len(records))))
                rows = math.ceil(len(records) / cols)
                size = target_size(
                    width, height, reserve=reserve, cols=cols, rows=rows
                )
                tiles = [
                    visualize_record(record, image, config=config, size=size)
                    for record in records.values()
                ]
                viz = grid(tiles, titles=list(records))
            if panel:
                viz = viz.with_panel(panel, title="metadata")
            show(source_name, viz)

        prev_windows = current_windows

        if quit_requested or cv2.waitKey() == ord("q"):
            break


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
            negative="",
        ),
    ] = False,
    max_partition_size_gb: Annotated[
        float | None,
        Parameter(alias="-m"),
    ] = None,
    zip: bool = True,
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
        zip: If ``True``, the exported dataset will be zipped into a
            single archive. If ``False``, the dataset will be exported as a
            directory with the specified structure.
        bucket_storage: Storage type of the dataset.

    """
    save_dir = save_dir or name
    if delete_existing and Path(save_dir).exists():
        shutil.rmtree(save_dir)
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    dataset.export(save_dir, dataset_type, max_partition_size_gb, zip)


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
            negative="",
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
            instead of capping the output at 50.
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


@app.command
def push(
    name: str,
    *,
    bucket_storage: BucketStorage,
    force: Annotated[
        bool,
        Parameter(alias="-f", negative=""),
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
        Parameter(alias="-f", negative=""),
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
        Parameter(alias="-p", negative=""),
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
