import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, cast

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
from luxonis_ml.data.utils.task_utils import get_task_type
from luxonis_ml.enums import DatasetType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from luxonis_ml.data.utils.data_utils import ClassDistributionRow
    from luxonis_ml.ldf import DatasetRecord, Detection
    from luxonis_ml.typing import Labels, LoaderOutput, Params, ParamValue
    from luxonis_ml.vizlab import (
        ComparisonReport,
        ComparisonResult,
        Renderable,
    )
    from luxonis_ml.vizlab.panel import PanelData

app = App(help="Dataset utilities.")


BucketStorageT: TypeAlias = Annotated[BucketStorage, Parameter(alias="-b")]
_SampleIdentity: TypeAlias = tuple[tuple[str, str], ...]
_ClassMappings: TypeAlias = dict[str, dict[str, int]]


def _deduped_class_names(
    dataset: LuxonisDataset, *, show_background: bool
) -> list[str]:
    """Dataset class names, deduped across tasks, for a stable palette and legend.

    A class name can appear under several tasks (e.g. "car" in car/boundingbox,
    car/keypoints and car/classification of a multitask dataset). Names are deduped
    while preserving first-seen order so the palette and legend carry one row per
    class, not one per (task, class) pair, and stripped to match the loader (which
    renders under ``name.strip()``) so a metadata name like " car" keys the same
    palette slot as the rendered box. Real classes are seeded first, in the same
    order regardless of ``show_background``, so their colors never shift when it is
    toggled; background (never drawn for detection/classification) is appended only
    when ``show_background`` renders its segmentation mask.
    """
    from luxonis_ml.data.loaders.label_converter import _BACKGROUND

    stripped = list(
        dict.fromkeys(
            name.strip()
            for names in dataset.get_class_names().values()
            for name in names
        )
    )
    classes = [n for n in stripped if n != _BACKGROUND]
    if show_background and _BACKGROUND in stripped:
        classes.append(_BACKGROUND)
    return classes


def _present_classes(records: "Iterable[DatasetRecord]") -> list[str]:
    """Class names present in ``records``, first-seen order, stripped.

    Drives which classes the inspector's ``c`` key cycles focus through, so it
    only offers what is actually on screen. Matches the loader's rendered names
    (``class_name.strip()``) and skips blank names.
    """
    seen: dict[str, None] = {}
    for record in records:
        for detection in record._annotations():
            name = (detection.class_name or "").strip()
            if name:
                seen.setdefault(name, None)
    return list(seen)


def _filter_records_by_task(
    records: "Mapping[str, DatasetRecord]",
    task_names: frozenset[str] | None,
) -> "dict[str, DatasetRecord]":
    """Keep only records whose complete task name was requested."""
    if task_names is None:
        return dict(records)
    return {
        name: record for name, record in records.items() if name in task_names
    }


def _array_shapes(
    labels: "Labels",
    task_names: frozenset[str] | None = None,
) -> dict[str, list[int]]:
    """Return array shapes keyed by complete, optionally filtered task path."""
    suffix = "/array"
    return {
        key[: -len(suffix)]: list(value.shape)
        for key, value in labels.items()
        if get_task_type(key) == "array"
        and key.endswith(suffix)
        and (task_names is None or key[: -len(suffix)] in task_names)
    }


def _panel_value(value: "ParamValue") -> "PanelData":
    """Normalize recursive loader metadata to the panel's data model."""
    if isinstance(value, Mapping):
        return {str(key): _panel_value(item) for key, item in value.items()}
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return [_panel_value(item) for item in value]
    return value


def _present_sample_metadata(
    sample_metadata: "Params",
) -> "dict[str, PanelData]":
    """Reshape batched-augmentation sample metadata for a readable side panel.

    A batch augmentation (e.g. MixUp, Mosaic) fuses several source samples into
    one output. The loader keeps the first sample's metadata at the top level and
    also stashes *every* input's metadata under ``batch_augmentation_metadata`` as
    a list of ``{"input_index", "sample_metadata"}`` entries (see
    `luxonis_ml.data.LuxonisLoader`) — handy for machine use, but it prints as a
    redundant, confusing blob (sample one appears twice, wrapped in bookkeeping
    keys). When that key is present this instead presents one clearly-labelled
    ``"sample N"`` group per contributing input, dropping the duplicated top-level
    copy; a single input collapses back to its plain metadata, and anything
    without the key is returned unchanged. Each sample's single-source
    ``filenames`` mapping is also flattened (see `_flatten_filenames`).

    Args:
        sample_metadata: The raw per-sample metadata from the loader.

    Returns:
        Metadata shaped for display: per-input ``"sample N"`` groups when the
        sample is a batch-augmentation fusion, else the input unchanged.

    """
    batch = sample_metadata.get("batch_augmentation_metadata")
    if not isinstance(batch, list) or not batch:
        return _flatten_filenames(sample_metadata)

    def sample_md(entry: "ParamValue", fallback: int) -> tuple[int, "Params"]:
        if isinstance(entry, Mapping):
            raw_index = entry.get("input_index", fallback)
            index = (
                int(raw_index)
                if isinstance(raw_index, (str, int, float))
                else fallback
            )
            raw_metadata = entry.get("sample_metadata")
            if isinstance(raw_metadata, Mapping):
                return index, {
                    str(key): value for key, value in raw_metadata.items()
                }
        return fallback, {}

    if len(batch) == 1:
        return _flatten_filenames(sample_md(batch[0], 0)[1])
    presented: dict[str, PanelData] = {}
    for position, entry in enumerate(batch):
        index, md = sample_md(entry, position)
        presented[f"sample {index + 1}"] = _flatten_filenames(md) or (
            "(no metadata)"
        )
    return presented


def _flatten_filenames(
    sample_metadata: "Params",
) -> "dict[str, PanelData]":
    """Collapse a single-source ``filenames`` mapping to a ``filename`` field.

    Records support multiple image sources, so the loader always reports
    ``filenames`` as a ``{source_name: basename}`` mapping (see
    `luxonis_ml.data.LuxonisLoader`). In the common single-image case that
    renders as a needless one-entry nested block, so this replaces it in place
    with a ``"filename"`` field wrapped in ``Block`` — the panel
    then shows it as a labelled line with the name below (full width, ellipsized
    if very long) rather than cramped after an inline prefix. Multi-source
    records keep the full mapping; metadata without ``filenames`` is unchanged.

    Args:
        sample_metadata: One sample's metadata.

    Returns:
        The metadata with a lone ``filenames`` entry flattened, or the input
        unchanged (same object) when there is nothing to flatten.

    """
    from luxonis_ml.vizlab import Block

    files = sample_metadata.get("filenames")
    if not (isinstance(files, Mapping) and len(files) == 1):
        # Params and PanelData describe the same recursive scalar/container
        # values here; this branch deliberately preserves the caller's dict.
        return cast("dict[str, PanelData]", sample_metadata)
    normalized = {
        key: _panel_value(value) for key, value in sample_metadata.items()
    }
    only = next(iter(files.values()))
    return {
        ("filename" if key == "filenames" else key): (
            Block(str(only)) if key == "filenames" else value
        )
        for key, value in normalized.items()
    }


def _print_comparison_summary(
    report: "ComparisonReport",
    gt_name: str,
    pred_name: str,
    per_class: bool,
) -> None:
    """Print a `ComparisonReport` as aggregate, per-class, and worst-image tables."""
    console = Console()
    summary = report.summary()
    aggregate = Table(
        title=f"{pred_name} vs {gt_name} — {summary['images']} images",
        box=rich.box.ROUNDED,
    )
    aggregate.add_column("metric", style="magenta")
    aggregate.add_column("value", justify="right")
    for metric in ("precision", "recall", "F1", "mean IoU", "TP", "FP", "FN"):
        aggregate.add_row(metric, str(summary[metric]))
    console.print(aggregate)

    if per_class and report.per_class():
        classes = Table(title="per class", box=rich.box.ROUNDED)
        classes.add_column("class", style="cyan")
        for header in ("P", "R", "TP", "FP", "FN"):
            classes.add_column(header, justify="right")
        for name, values in report.per_class().items():
            classes.add_row(
                name,
                f"{values['precision']:.3f}",
                f"{values['recall']:.3f}",
                str(values["tp"]),
                str(values["fp"]),
                str(values["fn"]),
            )
        console.print(classes)

    worst = report.worst(5)
    if worst:
        table = Table(
            title="worst images (by error count)", box=rich.box.ROUNDED
        )
        table.add_column("errors", justify="right", style="red")
        table.add_column("image")
        for count, image_name in worst:
            table.add_row(str(count), image_name)
        console.print(table)


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
    task_name: Annotated[list[str] | None, Parameter()] = None,
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
    ] = True,
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
    fast: Annotated[
        bool,
        Parameter(alias="-fa", negative=""),
    ] = False,
    show_all: Annotated[
        bool,
        Parameter(alias="-sa", negative=""),
    ] = False,
    theme: Annotated[
        Literal["dark", "light"],
        Parameter(alias="-t"),
    ] = "dark",
    plain: Annotated[
        bool,
        Parameter(alias="-p", negative=""),
    ] = False,
    save: Annotated[
        Path | None,
        Parameter(alias="-o"),
    ] = None,
    save_format: Annotated[
        Literal["png", "svg"],
        Parameter(alias="-fmt"),
    ] = "png",
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """Inspect images and annotations in a dataset.

    Hovering the mouse over a detection that carries annotation metadata shows
    that metadata in a tooltip, so dense scenes stay uncluttered. With
    ``--per-instance``, every spatial detection gets a tooltip even when it has
    no custom metadata.

    Interactive controls (also listed, and clickable, in the side panel):

    - ``m`` / ``k`` / ``b`` / ``l`` — toggle masks / keypoints / boxes / labels.
    - ``d`` — toggle decluttering (on by default): in busy scenes, tiny
      detections ringed by many others are hidden as noise; press to show all.
    - ``c`` — cycle a class focus through the classes present in the sample
      (isolating one class at a time; again to show all). After toggling classes
      by clicking the legend, ``c`` first resets them and restarts the cycle.
    - click a control row in the panel to trigger it, or a legend swatch to
      toggle that class on/off (disabled classes stay in the legend, dimmed).
    - any other key advances to the next sample; ``q`` quits.

    Args:
        name: Name of the dataset to inspect.
        view: Which splits of the dataset to inspect.
            If not provided, the "train" split will be inspected by default.
        task_name: Render only records matching these complete task names.
            May be provided more than once to select multiple tasks. By default,
            all tasks are rendered.
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
        per_instance: Show all spatial instances together, assigning a distinct
            color to each instance instead of each class. The class legend is
            hidden and hovering an instance shows its ID, class, task,
            annotation types, and metadata.
        list_augmentations: Show the augmentations applied to each
            displayed image. Requires '--aug-config' to be set.
        skeletons: Draw keypoint skeleton edges.
        keypoint_labels: Specify how to draw keypoint labels.
        legend: Draw a class-color legend on each image. Ignored in
            per-instance mode because colors identify instances, not classes.
        show_background: Render the semantic-segmentation background class
            (hidden by default) and include it in the palette and legend.
        fast: Lighter, much faster rendering for large or dense datasets: draw
            masks as fills only (no contour outlines), drop the soft drop shadows,
            and turn off shape anti-aliasing. Shape edges look slightly harder and
            text stays anti-aliased; every mask still shows (the ``m`` key still
            toggles them).
        show_all: Start with decluttering off, so tiny detections in crowded
            scenes are drawn instead of hidden (equivalent to pressing ``d`` up
            front). Toggle it back on any time with the ``d`` key.
        theme: Visual theme of the visualization: ``dark`` or ``light``.
        plain: Render just the framed image, without the side panel (controls,
            class legend, and sample metadata).
        save: Directory to write renders to instead of opening a window. One file
            is written per source image (annotations blended onto it); the viewer
            is never opened, so this works headless. The directory is created if
            needed.
        save_format: File format when ``--save`` is set: ``png`` (raster) or
            ``svg`` (annotations and metadata panel as crisp vectors over the
            embedded photo, scalable to any zoom). Both keep the panel unless
            ``--plain``.
        bucket_storage: Storage type of the dataset.

    """
    check_exists(name, bucket_storage)

    view = view or ["train"]
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)

    if len(dataset) == 0:
        raise ValueError(f"Dataset '{name}' is empty.")

    requested_tasks = tuple(dict.fromkeys(task_name or []))
    task_filter = frozenset(requested_tasks) if requested_tasks else None
    if task_filter is not None:
        available_tasks = dataset.get_task_names()
        unknown_tasks = [
            task for task in requested_tasks if task not in available_tasks
        ]
        if unknown_tasks:
            unknown = ", ".join(repr(task) for task in unknown_tasks)
            available = ", ".join(repr(task) for task in available_tasks)
            raise ValueError(
                f"Unknown task name(s): {unknown}. "
                f"Available task names: {available or '(none)'}."
            )

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
            DARK_THEME,
            LIGHT_THEME,
            Controls,
            Frame,
            Image,
            MaskOutline,
            Palette,
            Renderable,
            RenderOptions,
            Swatches,
            fit_grid,
            grid_hits,
            set_default_options,
            visualize_record,
        )
        from luxonis_ml.vizlab.adapters.instances import (
            instances_to_annotations,
            spatial_instances,
        )
        from luxonis_ml.vizlab.convert import (
            blend_records_to_annotations,
            metadata_annotations,
        )
        from luxonis_ml.vizlab.viewer import LayerState, Viewer
    except ImportError as e:
        raise SystemExit(
            "Visualization requires the 'viz' extra. "
            "Install it with `pip install luxonis-ml[viz]`."
        ) from e

    class_names = _deduped_class_names(
        dataset, show_background=show_background
    )

    viz_theme = LIGHT_THEME if theme == "light" else DARK_THEME
    if fast:
        # Lighter rendering for big/dense datasets via the theme's default style:
        # fill-only masks (contour tracing is the biggest per-mask cost) and no
        # soft drop shadows. Shape anti-aliasing is dropped below, on the options.
        viz_theme = viz_theme.with_style(
            viz_theme.style.merge(mask_outline=MaskOutline.NONE, shadow=False)
        )
    # The palette is pinned to the dataset's class order for stable colors; it
    # lives on the theme, and the whole bundle is the render options.
    palette = Palette(class_names)
    options = RenderOptions(
        theme=viz_theme.with_palette(palette),
        skeletons=keypoint_skeletons,
        keypoint_label_mode=keypoint_labels,
        draw_skeletons=skeletons,
        hover_metadata=True,
        # --fast also drops shape anti-aliasing (a render-wide setting).
        antialias=not fast,
    )
    # Make single images, panels, and grid backgrounds all follow the options
    # (so bare Images in this command pick up the theme/palette too).
    set_default_options(options)
    # Per-instance mode deliberately has its own sequence: the first instance
    # should receive the first generated color regardless of how many class
    # names were registered in the dataset palette above.
    instance_palette = Palette()

    # The legend reserves width for the dataset's longest class name (capped, so
    # one very long outlier does not blow the panel out), keeping the panel a
    # stable width sample to sample.
    longest_class = max(class_names, key=len, default="")[:24]

    def sidebar(
        panel: "Mapping[str, PanelData]",
        layers: "LayerState",
        *,
        controls: bool = True,
    ) -> "dict[str, PanelData]":
        """Prepend the CONTROLS and CLASSES sections to a sample's metadata panel.

        The interactive controls and the class-color legend live in the side
        panel (not floated over the image): controls come from ``layers`` (so they
        reflect the current toggles and refresh on every re-render), and the class
        swatches are keyed to the classes present in the sample (``layers.classes``)
        with stable full-dataset palette colors. ``controls=False`` omits the
        controls where they do not apply — currently the non-interactive saved
        renders.
        """
        out: dict[str, PanelData] = {}
        if controls:
            out["controls"] = Controls(
                tuple(
                    (c.key, c.name, c.value, c.active)
                    for c in layers.controls()
                )
            )
        names = layers.classes
        if legend and not per_instance and names:
            out["classes"] = Swatches(
                tuple((palette.color_for(name), name) for name in names),
                disabled=frozenset(layers.hidden),
                # Hold the legend (and panel) width to the dataset's longest class
                # name so it stays put as the per-sample class set changes.
                reserve=longest_class,
            )
        out.update(panel)
        return out

    def blend_annotations(
        image: np.ndarray,
        records: "Mapping[str, DatasetRecord]",
        layers: "LayerState",
        *,
        instances: "Sequence[tuple[str, Detection]] | None" = None,
    ) -> Image:
        """Draw every record's annotations onto one image, layer toggles applied.

        Shared by the interactive single/blended view and the headless save path.
        Detections from all tasks are blended together (a redundant classification
        chip is dropped next to boxes/keypoints/masks), the current ``layers``
        filter what is shown, and box-less metadata is added as hover-free cards.
        The image is returned unsized; the caller sets any display size.
        """
        viz = Image(image, options=options)
        detections = (
            instances_to_annotations(
                instances,
                options=options,
                palette=instance_palette,
            )
            if instances is not None
            else blend_records_to_annotations(records.values(), options)
        )
        for annotation in layers.apply_layers(detections, palette):
            viz.add(annotation)
        if instances is None:
            for overlay in metadata_annotations(
                [d for r in records.values() for d in r._annotations()],
                lone_object_card=True,
            ):
                viz.add(overlay)
        return viz

    def build_panel(
        sample_labels: "Labels", sample_metadata: "Params"
    ) -> "dict[str, PanelData]":
        panel = (
            dict(_present_sample_metadata(sample_metadata))
            if sample_metadata
            else {}
        )
        arrays = _array_shapes(sample_labels, task_filter)
        if arrays:
            panel["arrays"] = arrays
        if list_augmentations:
            applied = get_applied_augmentations()
            if applied:
                panel["augmentations"] = list(applied)
        return panel

    def records_from_labels(labels: "Labels") -> "dict[str, DatasetRecord]":
        """Convert loader labels and apply the command's task-name filter."""
        records = loader_output_to_records(
            labels,
            classes=classes,
            categorical_encodings=categorical_encodings,
            render_background=show_background,
        )
        return _filter_records_by_task(records, task_filter)

    def save_size(width: int, height: int) -> tuple[int, int] | None:
        """File render size: the ``--size-multiplier``, or native when ``auto``.

        There is no screen to fit to when saving, so ``auto`` keeps the source
        resolution rather than scaling.
        """
        if isinstance(size_multiplier, str):  # "auto"
            return None
        return (
            max(1, round(width * size_multiplier)),
            max(1, round(height * size_multiplier)),
        )

    def save_renders(directory: Path) -> None:
        """Write each source image to a file instead of opening a window.

        Fully headless (no viewer, no screen): every source image is built with
        `blend_annotations`, framed with the metadata panel (unless ``--plain``),
        and written by `Renderable.save`, whose extension picks the format — a
        ``png`` raster or a crisp vector ``svg`` (annotations and panel as vectors
        over the embedded photo). Decluttering follows ``--show-all``.
        """
        directory.mkdir(parents=True, exist_ok=True)
        layers = LayerState(declutter=not show_all)
        count = 0
        for data in loader:
            records = records_from_labels(data.labels)
            layers.update_classes(_present_classes(records.values()))
            panel = build_panel(data.labels, data.metadata)
            instances = (
                spatial_instances(records.values()) if per_instance else None
            )
            if per_instance and not instances:
                print(
                    "[yellow]Warning: Per-instance mode is not supported for "
                    "this sample. Showing all labels.[/yellow]"
                )
                instances = None
            for source_name, image in data.images.items():
                viz: Renderable = blend_annotations(
                    image.astype(np.uint8),
                    records,
                    layers,
                    instances=instances,
                ).render_at(save_size(image.shape[1], image.shape[0]))
                if not plain:
                    viz = viz.with_panel(
                        sidebar(panel, layers, controls=False)
                    )
                stem = f"{count:04d}_{Path(source_name).stem or 'image'}"
                viz.save(directory / f"{stem}.{save_format}")
                count += 1
        print(f"[green]Saved {count} render(s) to '{directory}'.[/green]")

    if save is not None:
        save_renders(save)
        return

    # vizlab now owns layout, screen-fit sizing, hover hit-testing, and the
    # interactive window loop; this command only prepares data and hands frames
    # (and their hit maps) to the viewer.
    # The controls live in the side panel now (see `sidebar`), so the viewer does
    # not also float its HUD over the image.
    viewer = Viewer(hud=False)
    # Decluttering hides tiny detections in crowded scenes by default; --show-all
    # starts with it off (the `d` key still toggles it live either way).
    viewer.layers.declutter = not show_all
    screen = viewer.screen
    # The `c` key cycles a class focus; the set is refreshed per sample to the
    # classes actually present (see the loop below).
    # The metadata panel is a fixed pixel width, independent of the image scale,
    # so reserve horizontal room for it when fitting a composite to the screen.
    panel_reserve = 400.0

    def display_size(
        width: int, height: int, reserve: float
    ) -> tuple[int, int] | None:
        """Display size for one source image, or ``None`` to keep native.

        An explicit ``--size-multiplier`` scales the source directly; ``auto``
        fits it to 90% of the screen (leaving room for the panel), scaling a
        small image *up* as well as a large one down so it is never shown tiny.
        ``None`` means render at the source size.
        """
        if size_multiplier != "auto":
            scale = size_multiplier
        elif screen is not None:
            avail_w = max(1.0, 0.9 * screen[0] - reserve)
            avail_h = max(1.0, 0.9 * screen[1])
            scale = min(avail_w / width, avail_h / height)
        else:
            return None
        if scale == 1.0:
            return None
        return (max(1, round(width * scale)), max(1, round(height * scale)))

    def compose_tiles(
        tiles: list[Renderable], cols: int, titles: list[str], reserve: float
    ) -> Frame:
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
                allow_upscale=True,
            )
        return grid_hits(
            tiles, ncols=cols, titles=titles, bg=viz_theme.background
        )

    def framed(frame: Frame, panel: "Mapping[str, PanelData]") -> Frame:
        """Attach the class legend (overlay) and metadata panel (right side).

        The panel (controls + classes + metadata) reframes the image as a rounded
        surface at a margin offset, so `Frame.with_panel` shifts the hover map to
        match. Unless ``--plain`` is active, it is always attached — the controls
        make it non-empty.
        """
        if plain:
            return frame
        return frame.with_panel(sidebar(panel, viewer.layers))

    def build_frame(
        image: np.ndarray,
        records: "Mapping[str, DatasetRecord]",
        panel: "Mapping[str, PanelData]",
        reserve: float,
        instances: "Sequence[tuple[str, Detection]] | None",
    ) -> Frame:
        """Build the display `Frame` for one source."""
        height, width = image.shape[:2]
        # The viewer's interactive layer toggles (masks/keypoints/labels, a class
        # focus) filter what is drawn without disturbing the metadata cards,
        # legend, or panel — `blend_annotations` applies them to the detections.
        if instances is not None or blend_all or len(records) <= 1:
            viz = blend_annotations(
                image,
                records,
                viewer.layers,
                instances=instances,
            )
            viz.render_at(display_size(width, height, reserve))
            return framed(viz.frame(), panel)
        # A grid of per-record tiles; compose_tiles sizes them so the whole
        # composite fits the screen and returns the composed hit map.
        cols = max(1, math.ceil(math.sqrt(len(records))))
        tiles = [
            visualize_record(record, image, options=options)
            for record in records.values()
        ]
        for tile in tiles:
            # Grid tiles carry no per-record panel here, so they are plain images
            # whose annotations the layer toggles filter.
            if isinstance(tile, Image):
                tile.annotations[:] = viewer.layers.apply_layers(
                    tile.annotations, palette
                )
        return framed(
            compose_tiles(tiles, cols, list(records), reserve), panel
        )

    for data in loader:
        images_dict = data.images
        records = records_from_labels(data.labels)
        # Cycle the class focus (`c`) over just the classes in this sample.
        viewer.layers.update_classes(_present_classes(records.values()))
        panel = build_panel(data.labels, data.metadata)
        instances = (
            spatial_instances(records.values()) if per_instance else None
        )
        if per_instance and not instances:
            print(
                "[yellow]Warning: Per-instance mode is not supported for "
                "this sample. Showing all labels.[/yellow]"
            )
            instances = None
        has_sidebar = not plain
        reserve = panel_reserve if has_sidebar else 0.0

        needs_wait = False
        for source_name, image in images_dict.items():
            image = image.astype(np.uint8)
            viewer.show(
                source_name,
                build_frame(image, records, panel, reserve, instances),
                # Re-render this window's frame whenever a layer toggle changes;
                # bind the loop-varying data so each window rebuilds its own.
                render=lambda _, image=image, records=records, panel=panel, reserve=reserve, instances=instances: (
                    build_frame(image, records, panel, reserve, instances)
                ),
            )
            needs_wait = True

        # Windows for sources no longer present (a differing next sample) close.
        viewer.destroy_stale(set(images_dict.keys()))
        if needs_wait and viewer.wait() == "q":
            break

    viewer.close()


@app.command
def compare(
    name: str,
    predictions: str,
    *,
    view: Annotated[list[str] | None, Parameter(alias="-v")] = None,
    layout: Annotated[
        Literal["overlay", "dual", "triple"],
        Parameter(alias="-l"),
    ] = "overlay",
    iou_threshold: Annotated[float, Parameter(alias="--iou")] = 0.5,
    score_threshold: Annotated[float, Parameter(alias="--score")] = 0.25,
    class_agnostic: Annotated[bool, Parameter(negative="")] = False,
    per_class: Annotated[bool, Parameter(alias="-pc", negative="")] = False,
    errors_only: Annotated[bool, Parameter(alias="-e", negative="")] = False,
    summary: Annotated[bool, Parameter(negative="")] = False,
    size_multiplier: Annotated[
        float | Literal["auto"], Parameter(alias="-s")
    ] = "auto",
    skeletons: Annotated[bool, Parameter(negative="--no-skeletons")] = True,
    keypoint_labels: Annotated[
        Literal["none", "numbers", "names", "full"], Parameter()
    ] = "none",
    legend: Annotated[bool, Parameter(alias="-lg", negative="")] = False,
    show_background: Annotated[
        bool, Parameter(alias="-bg", negative="")
    ] = False,
    theme: Annotated[Literal["dark", "light"], Parameter(alias="-t")] = "dark",
    force_update: Annotated[bool, Parameter(alias="-f", negative="")] = False,
    bucket_storage: BucketStorageT = BucketStorage.LOCAL,
):
    """Compare a prediction dataset against a ground-truth dataset.

    Treats ``predictions`` as a model's outputs and ``name`` as the ground truth,
    matches them sample by sample (COCO-style: greedy by confidence, class-aware,
    at ``--iou-threshold``), and draws each frame colored by outcome — green hit,
    red false alarm, dashed-amber miss, orange class error — with a metrics side
    panel. Matched poses are graded per keypoint. Samples are paired by their
    source filenames, and missing/extra samples are reported. Press any key to
    advance, 'q' to quit.

    Args:
        name: Name of the ground-truth dataset.
        predictions: Name of the dataset to treat as predictions.
        view: Which splits to compare (default: the "train" split).
        layout: ``overlay`` (verdict colors on one frame, hoverable), ``dual``
            (ground truth beside prediction, colored by identity), or ``triple``
            (ground truth | prediction | verdict diff).
        iou_threshold: Overlap threshold for a localized match.
        score_threshold: Confidence cutoff for predictions. LDF prediction
            datasets preserve confidence as per-instance ``score`` or
            ``confidence`` metadata.
        class_agnostic: Match regardless of class label (no class-error verdict).
        per_class: Add a per-class precision/recall breakdown to the panel.
        errors_only: Draw only mistakes (false alarms, misses, class errors),
            hiding correct detections; the metrics panel still reflects all.
        summary: Skip the interactive viewer; run the whole view, print
            dataset-wide precision/recall/F1 (with ``--per-class``, a per-class
            table) and the worst images, and write a confusion-matrix figure.
        size_multiplier: Display scale; ``auto`` fits the screen.
        skeletons: Draw keypoint skeleton limbs (gradient-colored between graded
            joints).
        keypoint_labels: How to label keypoints.
        legend: Draw a class-color legend on each frame.
        show_background: Render the semantic-segmentation background class.
        theme: Visual theme: ``dark`` or ``light``.
        force_update: Force synchronization with remote storage first.
        bucket_storage: Storage type of the datasets.

    """
    check_exists(name, bucket_storage)
    check_exists(predictions, bucket_storage)

    view = view or ["train"]
    gt_dataset = LuxonisDataset(name, bucket_storage=bucket_storage)
    pred_dataset = LuxonisDataset(predictions, bucket_storage=bucket_storage)
    if len(gt_dataset) == 0:
        raise ValueError(f"Dataset '{name}' is empty.")
    if len(pred_dataset) == 0:
        raise ValueError(f"Prediction dataset '{predictions}' is empty.")

    def _loader(dataset: LuxonisDataset) -> LuxonisLoader:
        return LuxonisLoader(
            dataset,
            view=view,
            update_mode="all" if force_update else "missing",
        )

    gt_loader, pred_loader = _loader(gt_dataset), _loader(pred_dataset)
    gt_classes = gt_dataset.get_classes()
    gt_categorical = gt_dataset.get_categorical_encodings()
    pred_classes = pred_dataset.get_classes()
    pred_categorical = pred_dataset.get_categorical_encodings()

    try:
        from luxonis_ml.data.loaders.label_converter import (
            loader_output_to_records,
        )
        from luxonis_ml.vizlab import (
            DARK_THEME,
            LIGHT_THEME,
            ComparisonReport,
            Frame,
            Image,
            Legend,
            Palette,
            RenderOptions,
            Verdict,
            confusion_matrix_figure,
            match_detections,
            set_default_options,
        )
        from luxonis_ml.vizlab import (
            compare as viz_compare,
        )
        from luxonis_ml.vizlab.viewer import Viewer
    except ImportError as e:
        raise SystemExit(
            "Visualization requires the 'viz' extra. "
            "Install it with `pip install luxonis-ml[viz]`."
        ) from e

    class_names = _deduped_class_names(
        gt_dataset, show_background=show_background
    )
    viz_theme = LIGHT_THEME if theme == "light" else DARK_THEME
    palette = Palette(class_names)
    options = RenderOptions(
        theme=viz_theme.with_palette(palette),
        skeletons=gt_dataset.get_skeletons(),
        keypoint_label_mode=keypoint_labels,
        draw_skeletons=skeletons,
    )
    set_default_options(options)
    class_legend = (
        Legend(entries=class_names, palette=palette, title="classes")
        if legend and class_names
        else None
    )

    layout_show: Literal["overlay", "side_by_side", "triptych"]
    if layout == "overlay":
        layout_show = "overlay"
    elif layout == "dual":
        layout_show = "side_by_side"
    else:
        layout_show = "triptych"

    verdicts = (
        {Verdict.FP, Verdict.FN, Verdict.CLASS_ERROR} if errors_only else None
    )

    def records_for(
        data: "LoaderOutput",
        classes: _ClassMappings,
        categorical: _ClassMappings,
    ) -> "dict[str, DatasetRecord]":
        return loader_output_to_records(
            data.labels,
            classes=classes,
            categorical_encodings=categorical,
            render_background=show_background,
        )

    def sample_identity(data: "LoaderOutput") -> _SampleIdentity:
        """Stable sample identity from the loader's source-name/filename map."""
        metadata = getattr(data, "metadata", None)
        filenames = (
            metadata.get("filenames") if isinstance(metadata, dict) else None
        )
        if not isinstance(filenames, dict) or not filenames:
            raise ValueError(
                "Dataset comparison requires loader filename metadata to match "
                "samples by identity."
            )
        return tuple(
            sorted(
                (str(source), str(filename))
                for source, filename in filenames.items()
            )
        )

    def identity_index(
        loader: LuxonisLoader, dataset_name: str
    ) -> dict[_SampleIdentity, int]:
        """Map unique sample identities to loader indices."""
        indexed: dict[_SampleIdentity, int] = {}
        for index, data in enumerate(loader):
            identity = sample_identity(data)
            if identity in indexed:
                shown = ", ".join(
                    f"{source}={filename}" for source, filename in identity
                )
                raise ValueError(
                    f"Dataset '{dataset_name}' contains duplicate sample "
                    f"identity: {shown}."
                )
            indexed[identity] = index
        return indexed

    def identity_label(identity: _SampleIdentity) -> str:
        return ", ".join(
            f"{source}={filename}" for source, filename in identity
        )

    def report_unpaired(
        identities: set[_SampleIdentity], *, description: str
    ) -> None:
        if not identities:
            return
        ordered = sorted(identity_label(identity) for identity in identities)
        preview = ", ".join(ordered[:10])
        remainder = len(ordered) - 10
        if remainder > 0:
            preview += f", and {remainder} more"
        print(f"[yellow]{description} ({len(ordered)}): {preview}.[/yellow]")

    def match_sample(
        gt_records: "Mapping[str, DatasetRecord]",
        pred_records: "Mapping[str, DatasetRecord]",
    ) -> "ComparisonResult":
        gt_dets = [d for r in gt_records.values() for d in r._annotations()]
        pred_dets = [
            d for r in pred_records.values() for d in r._annotations()
        ]
        return match_detections(
            gt_dets,
            pred_dets,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            class_aware=not class_agnostic,
        )

    gt_indices = identity_index(gt_loader, name)
    pred_indices = identity_index(pred_loader, predictions)
    gt_identities = set(gt_indices)
    pred_identities = set(pred_indices)
    report_unpaired(
        gt_identities - pred_identities,
        description="Missing prediction samples",
    )
    report_unpaired(
        pred_identities - gt_identities,
        description="Extra prediction samples",
    )
    shared = sorted(
        gt_identities & pred_identities, key=lambda item: gt_indices[item]
    )
    if not shared:
        raise ValueError(
            "The ground-truth and prediction datasets have no samples in "
            "common by source filename."
        )

    # ``--summary``: iterate the whole view headlessly, accumulate a report,
    # print it, and write a confusion-matrix figure — no interactive window.
    if summary:
        report = ComparisonReport()
        for identity in shared:
            gt_data = gt_loader[gt_indices[identity]]
            pred_data = pred_loader[pred_indices[identity]]
            report.add(
                match_sample(
                    records_for(gt_data, gt_classes, gt_categorical),
                    records_for(pred_data, pred_classes, pred_categorical),
                ),
                name=identity_label(identity),
            )
        _print_comparison_summary(report, name, predictions, per_class)
        out_path = Path(f"{name}_vs_{predictions}_confusion.png")
        confusion_matrix_figure(report, options=options).save(out_path)
        print(f"[green]Wrote confusion matrix to {out_path}[/green]")
        return

    viewer = Viewer()
    screen = viewer.screen
    # The metrics panel is a fixed pixel width; reserve room for it when fitting.
    panel_reserve = 400.0

    def display_size(width: int, height: int) -> tuple[int, int] | None:
        """Fit ``width`` x ``height`` to the screen (or apply the multiplier)."""
        if size_multiplier != "auto":
            scale = size_multiplier
        elif screen is not None:
            avail_w = max(1.0, 0.9 * screen[0] - panel_reserve)
            avail_h = max(1.0, 0.9 * screen[1])
            scale = min(avail_w / width, avail_h / height, 1.0)
        else:
            return None
        if scale == 1.0:
            return None
        return (max(1, round(width * scale)), max(1, round(height * scale)))

    def build_frame(
        image: np.ndarray,
        gt_records: "Mapping[str, DatasetRecord]",
        pred_records: "Mapping[str, DatasetRecord]",
    ) -> Frame:
        """Match GT vs predictions for one image and build its display frame."""
        result = match_sample(gt_records, pred_records)
        viz = viz_compare(
            image,
            result=result,
            options=options,
            show=layout_show,
            panel=False,
            verdicts=verdicts,
        )
        # panel=False, so viz is the plain comparison scene (an image for the
        # overlay layout, a grid composite for side-by-side / triptych).
        viz.render_at(display_size(viz.width, viz.height))
        frame = viz.frame()
        display = frame.image
        if class_legend is not None:
            # Overlay the class legend; bake a grid composite to an image first
            # so it carries a mutable annotation list to `add` onto.
            if not isinstance(display, Image):
                display = Image(frame.render()).with_hitmap(frame.hitmap)
            display.add(class_legend)
        metrics: dict[str, PanelData] = dict(result.summary())
        if per_class and len(result.per_class) > 1:
            metrics["by class"] = result.per_class_panel()
        return display.with_panel(metrics, title="Comparison").frame()

    for identity in shared:
        gt_data = gt_loader[gt_indices[identity]]
        pred_data = pred_loader[pred_indices[identity]]
        gt_records = records_for(gt_data, gt_classes, gt_categorical)
        pred_records = records_for(pred_data, pred_classes, pred_categorical)
        for source_name, image in gt_data.images.items():
            viewer.show(
                source_name,
                build_frame(image.astype(np.uint8), gt_records, pred_records),
            )
        viewer.destroy_stale(set(gt_data.images.keys()))
        if viewer.wait() == "q":
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
    ] = "bars",
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

    # Only spatial annotations have heatmaps. Metadata can have class counts,
    # but it has no meaningful health plot and should not create a window.
    all_task_names = sorted(stats["heatmaps"])
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
        if not heatmaps_by_type:
            console.print(f"[info]No plots for task name: {task_name}[/info]")
            continue

        def render_grid(
            s: float,
            _dist: "Mapping[str, list[ClassDistributionRow]]" = (
                class_dist_by_type
            ),
            _heat: "Mapping[str, Sequence[Sequence[float]] | None]" = (
                heatmaps_by_type
            ),
            _cls: (
                "Mapping[str, Mapping[str, Sequence[Sequence[float]]]]"
            ) = class_heatmaps_by_type,
        ) -> "Renderable":
            return health_plots.build_health_grid(
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
