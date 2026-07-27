import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import cv2
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
from luxonis_ml.data.utils.task_utils import get_task_name, get_task_type
from luxonis_ml.enums import DatasetType

if TYPE_CHECKING:
    from luxonis_ml.vizlab import Image

app = App(help="Dataset utilities.")


BucketStorageT: TypeAlias = Annotated[BucketStorage, Parameter(alias="-b")]


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
        float | Literal["auto"],
        Parameter(alias="-s"),
    ] = "auto",
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
    legend: Annotated[
        bool,
        Parameter(alias="-lg", negative=""),
    ] = False,
    show_background: Annotated[
        bool,
        Parameter(alias="-bg", negative=""),
    ] = False,
    theme: Annotated[
        Literal["dark", "light"],
        Parameter(alias="-t"),
    ] = "dark",
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """Inspect images and annotations in a dataset.

    Hovering the mouse over a detection that carries annotation metadata shows
    that metadata in a tooltip, so dense scenes stay uncluttered. Press any key
    to advance to the next sample, or 'q' to quit.

    Args:
        name: Name of the dataset to inspect.
        view: Which splits of the dataset to inspect.
            If not provided, the "train" split will be inspected by default.
        aug_config: Path to a JSON or YAML config defining
            augmentations to apply when inspecting the dataset.
            If not provided, no augmentations will be applied.
        size_multiplier: Multiplier for the displayed image size. If
            not given, the image is scaled automatically to fit the
            screen (accounting for all components).
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
        legend: Draw a class-color legend on each image.
        show_background: Render the semantic-segmentation background class
            (hidden by default) and include it in the palette and legend.
        theme: Visual theme of the visualization: ``dark`` or ``light``.
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
            _BACKGROUND,
            loader_output_to_records,
        )
        from luxonis_ml.vizlab import (
            DARK_THEME,
            LIGHT_THEME,
            HitMap,
            Image,
            Legend,
            Palette,
            VizConfig,
            fit_grid,
            grid_hits,
            set_default_theme,
            visualize_record,
        )
        from luxonis_ml.vizlab.convert import (
            blend_records_to_annotations,
            detection_to_annotations,
            metadata_annotations,
        )
        from luxonis_ml.vizlab.viewer import Viewer
    except ImportError as e:
        raise SystemExit(
            "Visualization requires the 'viz' extra. "
            "Install it with `pip install luxonis-ml[viz]`."
        ) from e

    # A class name can appear under several tasks (e.g. "car" in car/boundingbox,
    # car/keypoints and car/classification of a multitask dataset). Dedupe while
    # preserving first-seen order so the palette and legend carry one row per
    # class, not one per (task, class) pair. Names are stripped to match the
    # loader (which renders detections/masks under ``name.strip()``); without this
    # a metadata name like " background" or " car" would key a different palette
    # slot than the rendered mask, so legend colors would not match the drawing.
    stripped_names = list(
        dict.fromkeys(
            class_name.strip()
            for classes in dataset.get_class_names().values()
            for class_name in classes
        )
    )
    # Real classes are always seeded first, in the same order regardless of the
    # ``--show-background`` flag, so their colors never change when it is toggled.
    # Background (never drawn for detection/classification) is only appended — at
    # the end, taking its own trailing color — when ``--show-background`` renders
    # its segmentation mask; otherwise it is dropped entirely.
    class_names: list[str] = [n for n in stripped_names if n != _BACKGROUND]
    if show_background and _BACKGROUND in stripped_names:
        class_names.append(_BACKGROUND)

    viz_theme = LIGHT_THEME if theme == "light" else DARK_THEME
    # Make single images, panels, and grid backgrounds all follow the theme.
    set_default_theme(viz_theme)

    config = VizConfig(
        palette=Palette(class_names),
        skeletons=keypoint_skeletons,
        keypoint_label_mode=keypoint_labels,
        draw_skeletons=skeletons,
        theme=viz_theme,
        hover_metadata=True,
    )
    class_legend = (
        Legend(
            entries=class_names,
            palette=config.palette,
            title="classes",
        )
        if legend and class_names
        else None
    )

    def build_panel(sample_labels: dict, sample_metadata: dict) -> dict:
        panel = dict(sample_metadata) if sample_metadata else {}
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

    # vizlab now owns layout, screen-fit sizing, hover hit-testing, and the
    # interactive window loop; this command only prepares data and hands frames
    # (and their hit maps) to the viewer.
    viewer = Viewer()
    screen = viewer.screen
    # The metadata panel is a fixed pixel width, independent of the image scale,
    # so reserve horizontal room for it when fitting a composite to the screen.
    panel_reserve = 400.0

    def display_size(
        width: int, height: int, reserve: float
    ) -> tuple[int, int] | None:
        """Display size for one source image, or ``None`` to keep native.

        An explicit ``--size-multiplier`` scales the source directly; ``auto``
        fits it within 90% of the screen (leaving room for the panel), never
        upscaling. ``None`` means render at the source size.
        """
        if size_multiplier != "auto":
            scale = size_multiplier
        elif screen is not None:
            avail_w = max(1.0, 0.9 * screen[0] - reserve)
            avail_h = max(1.0, 0.9 * screen[1])
            scale = min(avail_w / width, avail_h / height, 1.0)
        else:
            return None
        if scale == 1.0:
            return None
        return (max(1, round(width * scale)), max(1, round(height * scale)))

    def compose_tiles(
        tiles: list[Image], cols: int, titles: list[str], reserve: float
    ) -> tuple[Image, HitMap]:
        """Grid record tiles, sizing them for the screen (or the multiplier)."""
        if size_multiplier != "auto":
            scaled = [
                tile.copy().render_at(
                    (
                        max(1, round(tile.width * size_multiplier)),
                        max(1, round(tile.height * size_multiplier)),
                    )
                )
                for tile in tiles
            ]
            return grid_hits(
                scaled, ncols=cols, titles=titles, bg=viz_theme.background
            )
        if screen is not None:
            target = (round(0.9 * screen[0]), round(0.9 * screen[1]))
            return fit_grid(
                tiles,
                target=target,
                ncols=cols,
                reserve=reserve,
                titles=titles,
                bg=viz_theme.background,
            )
        return grid_hits(
            tiles, ncols=cols, titles=titles, bg=viz_theme.background
        )

    def build_frame(
        image: np.ndarray, records: dict, panel: dict, reserve: float
    ) -> tuple[Image, HitMap]:
        """Build the ``(display, hit map)`` for one non-per-instance source.

        The legend is an overlay and the panel is attached on the right, so a hit
        map captured before framing stays valid on the framed image.
        """
        height, width = image.shape[:2]
        if blend_all or len(records) <= 1:
            viz = Image(image, config=config).render_at(
                display_size(width, height, reserve)
            )
            # Blending several tasks onto one image: a classification task's
            # corner chip is redundant next to boxes/keypoints/masks, so it is
            # dropped unless a class tag is all there is to show.
            for annotation in blend_records_to_annotations(
                records.values(), config
            ):
                viz.add(annotation)
            # Box-less metadata has nothing to hover, so show it as a card; a
            # lone object is carded too, so a single detection needs no hover.
            for overlay in metadata_annotations(
                [d for r in records.values() for d in r._annotations()],
                text_key=config.text_metadata_key,
                lone_object_card=True,
            ):
                viz.add(overlay)
            if class_legend is not None:
                viz.add(class_legend)
            _, hitmap = viz.render_hits()
            display = (
                viz.with_panel(panel, title="Sample metadata")
                if panel
                else viz
            )
            return display, hitmap
        # A grid of per-record tiles; compose_tiles sizes them so the whole
        # composite fits the screen and returns the composed hit map.
        cols = max(1, math.ceil(math.sqrt(len(records))))
        tiles = [
            visualize_record(record, image, config=config)
            for record in records.values()
        ]
        grid_img, hitmap = compose_tiles(tiles, cols, list(records), reserve)
        if class_legend is not None:
            grid_img.add(class_legend)
        display = (
            grid_img.with_panel(panel, title="Sample metadata")
            if panel
            else grid_img
        )
        return display, hitmap

    def show_instances(
        source_name: str,
        image: np.ndarray,
        instances: list,
        panel: dict,
        reserve: float,
    ) -> bool:
        """Show each instance in its own window; return ``True`` on quit."""
        height, width = image.shape[:2]
        size = display_size(width, height, reserve)
        for task_name, detection in instances:
            viz = Image(image, config=config).render_at(size)
            for annotation in detection_to_annotations(
                detection, config, task_name=task_name
            ):
                viz.add(annotation)
            # A single instance per window: card its metadata (no hover).
            for overlay in metadata_annotations(
                [detection],
                text_key=config.text_metadata_key,
                lone_object_card=True,
            ):
                viz.add(overlay)
            if class_legend is not None:
                viz.add(class_legend)
            display = (
                viz.with_panel(panel, title="Sample metadata")
                if panel
                else viz
            )
            if viewer.show_blocking(source_name, display) == "q":
                return True
        return False

    for data in loader:
        images_dict = data.images
        records = loader_output_to_records(
            data.labels,
            classes=classes,
            categorical_encodings=categorical_encodings,
            render_background=show_background,
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
        reserve = panel_reserve if panel else 0.0

        quit_requested = False
        needs_wait = False
        for source_name, image in images_dict.items():
            image = image.astype(np.uint8)
            if per_instance and instances:
                if show_instances(
                    source_name, image, instances, panel, reserve
                ):
                    quit_requested = True
                    break
                continue
            if per_instance:
                print(
                    "[yellow]Warning: Per-instance mode is not supported for "
                    f"this dataset. Showing all labels for '{source_name}'.[/yellow]"
                )
            display, hitmap = build_frame(image, records, panel, reserve)
            viewer.show(source_name, display, hitmap)
            needs_wait = True

        # Windows for sources no longer present (a differing next sample) close.
        viewer.destroy_stale(set(images_dict.keys()))
        if quit_requested:
            break
        # Per-instance mode already blocked on each instance; otherwise block for
        # a keypress while hover tooltips redraw.
        if needs_wait and viewer.wait() == "q":
            break

    viewer.close()


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
    theme: Annotated[
        Literal["dark", "light"],
        Parameter(alias="-t"),
    ] = "dark",
    gradient: Annotated[
        str,
        Parameter(alias="-g"),
    ] = "viridis",
    distribution: Annotated[
        Literal["bars", "chips", "stacked", "pie", "donut"],
        Parameter(alias="-m"),
    ] = "stacked",
    scale: Annotated[
        float,
        Parameter(alias="-c"),
    ] = 1.0,
    per_class: Annotated[
        bool,
        Parameter(alias="-p"),
    ] = False,
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
            will be saved. If not provided, the plots are shown in
            interactive OpenCV windows instead.
        theme: Visual theme of the plots: ``dark`` or ``light``.
        gradient: Name of the heatmap colormap (e.g. ``viridis``,
            ``turbo``, ``magma``, ``inferno``, ``jet``).
        distribution: How class counts are drawn: ``bars``, ``chips``,
            the ``stacked`` proportion strip, or a ``pie``/``donut`` chart.
        scale: Font and mark scale for the plots (``1.0`` is nominal;
            increase for larger text and marks).
        per_class: Render one heatmap per class (each in its class color)
            instead of a single combined heatmap. Best for datasets with a
            handful of classes.
        bucket_storage: Storage type of the dataset.

    """
    check_exists(name, bucket_storage)
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    stats = dataset.get_statistics(
        sample_size=sample_size, view=view, per_class_heatmaps=per_class
    )
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

    try:
        from luxonis_ml.data.utils import health_plots
        from luxonis_ml.vizlab import DARK_THEME, GRADIENTS, LIGHT_THEME
        from luxonis_ml.vizlab.viewer import Cv2Backend
    except ImportError as exc:
        console.print(
            f"[red]Health charts require the 'viz' extra: {exc}[/red]"
        )
        return

    if gradient not in GRADIENTS:
        options = ", ".join(sorted(GRADIENTS))
        console.print(
            f"[red]Unknown gradient '{gradient}'. Options: {options}[/red]"
        )
        return

    plot_theme = LIGHT_THEME if theme == "light" else DARK_THEME
    # The viewer's cv2 backend also provides the best-effort screen size.
    screen = Cv2Backend().screen_size() if not save_dir else None

    for task_name in all_task_names:
        class_dist_by_type = stats["class_distributions"].get(task_name, {})
        heatmaps_by_type = stats["heatmaps"].get(task_name, {})
        class_heatmaps_by_type = stats.get("class_heatmaps", {}).get(
            task_name, {}
        )
        if not (set(class_dist_by_type) | set(heatmaps_by_type)):
            console.print(f"[info]No plots for task name: {task_name}[/info]")
            continue

        def render_grid(
            s: float,
            _task: str = task_name,
            _dist: dict = class_dist_by_type,
            _heat: dict = heatmaps_by_type,
            _cls: dict = class_heatmaps_by_type,
        ) -> "Image":
            return health_plots.build_health_grid(
                _task,
                _dist,
                _heat,
                class_heatmaps_by_type=_cls or None,
                theme=plot_theme,
                gradient=gradient,
                mode=distribution,
                scale=s,
            )

        image = render_grid(scale)
        if screen is not None:
            # Draw the charts/text at the size they will actually be shown: if
            # the grid overflows the screen, re-render it smaller rather than
            # downscaling the finished raster (resampling drawn vector content
            # always softens/aliases it).
            fit = min(
                0.9 * screen[0] / image.width,
                0.9 * screen[1] / image.height,
                1.0,
            )
            if fit < 0.98:
                image = render_grid(scale * fit)
        if save_dir:
            image.save(f"{save_dir}/dataset_health_{task_name}.png")
            continue

        window = (
            f"dataset health: {task_name}" if task_name else "dataset health"
        )
        out = image.to_numpy("bgr")
        # If the grid is larger than the screen, shrink it ourselves with a
        # high-quality area filter and show 1:1. Letting OpenCV's WINDOW_NORMAL
        # scale the full-size raster instead uses a crude filter that re-aliases
        # the smooth chart edges.
        if screen is not None:
            out_h, out_w = out.shape[:2]
            fit = min(0.9 * screen[0] / out_w, 0.9 * screen[1] / out_h, 1.0)
            if fit < 1.0:
                out = cv2.resize(
                    out,
                    (max(1, round(out_w * fit)), max(1, round(out_h * fit))),
                    interpolation=cv2.INTER_AREA,
                )
        out_h, out_w = out.shape[:2]
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, out_w, out_h)
        if screen is not None:
            cv2.moveWindow(
                window,
                max(0, (screen[0] - out_w) // 2),
                max(0, (screen[1] - out_h) // 2),
            )
        cv2.imshow(window, out)
        console.print(
            "[info]Press any key for the next task, or 'q' to quit.[/info]"
        )
        if cv2.waitKey(0) == ord("q"):
            break

    if not save_dir:
        cv2.destroyAllWindows()


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
