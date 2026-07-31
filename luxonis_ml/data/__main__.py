import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, cast

import cv2
import numpy as np
import rich.box
from cyclopts import App, Group, Parameter, validators
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
from luxonis_ml.data.utils.data_utils import HEATMAP_TASK_TYPES
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.data.utils.inspection import (
    InspectionAnnotationType,
    NameFilterMode,
    SampleFilterConfig,
)
from luxonis_ml.data.utils.task_utils import get_task_type
from luxonis_ml.enums import DatasetType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from luxonis_ml.data.utils.data_utils import ClassDistributionRow
    from luxonis_ml.ldf import DatasetRecord
    from luxonis_ml.typing import Labels, LoaderOutput, Params, ParamValue
    from luxonis_ml.vizlab import (
        Color,
        ComparisonReport,
        ComparisonResult,
        Renderable,
    )
    from luxonis_ml.vizlab.adapters.instances import ColorBy
    from luxonis_ml.vizlab.layout.panel import PanelData
    from luxonis_ml.vizlab.viewer import LayerState, PreparedFrame

    _InspectionSample: TypeAlias = tuple[
        LoaderOutput,
        dict[str, DatasetRecord],
        dict[str, PanelData],
    ]

app = App(help="Dataset utilities.")


BucketStorageT: TypeAlias = Annotated[BucketStorage, Parameter(alias="-b")]
_SampleIdentity: TypeAlias = tuple[tuple[str, str], ...]
#: Mirrors `luxonis_ml.vizlab.options.ArrayKind`, spelled out here because this
#: module must import without the ``viz`` extra installed.
_ArrayKindName: TypeAlias = Literal[
    "scalar",
    "signed",
    "flow",
    "normals",
    "image",
    "scores",
    "confidence",
    "class_confidence",
]
_ClassMappings: TypeAlias = dict[str, dict[str, int]]
_NO_SAMPLE_FILTERS = SampleFilterConfig()
_DATASET_OPTIONS = Group("Dataset options", sort_key=10)
_SAMPLE_FILTERS = Group("Sample filters", sort_key=20)
_AUGMENTATION_OPTIONS = Group("Augmentation options", sort_key=30)
_MATCHING_OPTIONS = Group("Matching options", sort_key=30)
_VISUALIZATION_OPTIONS = Group("Visualization options", sort_key=40)
_VIEWER_OPTIONS = Group("Viewer options", sort_key=50)
_REPORT_OPTIONS = Group("Reporting options", sort_key=50)
_OUTPUT_OPTIONS = Group("Output options", sort_key=60)


@dataclass(frozen=True, slots=True)
class _PreparedInspectionSample:
    """A sample whose single composed frame finished background rendering.

    Every image source of a sample is tiled into that one frame rather than
    getting a window of its own, so a stereo pair reads as a pair instead of as
    two windows the viewer would stack on the same screen point.
    """

    images: "dict[str, np.ndarray]"
    arrays: "dict[str, np.ndarray]"
    records: "dict[str, DatasetRecord]"
    panel: "dict[str, PanelData]"
    layers: "LayerState"
    color_by: "ColorBy"
    frame: "PreparedFrame"


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
    mode: NameFilterMode = "include",
) -> "dict[str, DatasetRecord]":
    """Apply an inclusive or exclusive complete-task-name filter."""
    if task_names is None:
        return dict(records)
    return {
        name: record
        for name, record in records.items()
        if (name in task_names) == (mode == "include")
    }


def _unique_titles(titles: Sequence[str]) -> list[str]:
    """Disambiguate repeated tile titles so none is lost as a mapping key."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for title in titles:
        count = seen.get(title, 0)
        seen[title] = count + 1
        unique.append(title if not count else f"{title} ({count + 1})")
    return unique


def _array_labels(
    labels: "Labels",
    task_names: frozenset[str] | None = None,
    mode: NameFilterMode = "include",
) -> dict[str, np.ndarray]:
    """Return array labels keyed by complete, optionally filtered task path.

    The one place that decides which labels are arrays, so the annotation-type
    filter and the renderer can never disagree about what ``--task-name``
    scopes.
    """
    suffix = "/array"
    return {
        key[: -len(suffix)]: value
        for key, value in labels.items()
        if get_task_type(key) == "array"
        and key.endswith(suffix)
        and (
            task_names is None
            or (key[: -len(suffix)] in task_names) == (mode == "include")
        )
    }


def _loader_annotation_types(
    labels: "Labels",
    task_names: frozenset[str] | None = None,
    mode: NameFilterMode = "include",
) -> frozenset[InspectionAnnotationType]:
    """Annotation families present only in raw loader labels."""
    return (
        frozenset({"array"})
        if _array_labels(labels, task_names, mode)
        else frozenset()
    )


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


def _plain_by_default(plain: "bool | None", save: "Path | None") -> bool:
    """Resolve ``--plain`` / ``--no-plain``, defaulting it on for a clip.

    A clip is encoded onto one fixed canvas, but the side panel's width follows
    each sample's metadata and the rounded surface adds a margin around the
    photo — so leaving them on makes every frame that disagrees with the first
    one letterbox. The bare image is almost always what a clip is for, and
    ``--no-plain`` says otherwise.

    Args:
        plain: The flag as given, or ``None`` when it was not passed at all.
        save: The ``--save`` destination, if any.

    Returns:
        Whether to render without the panel and its surround.

    """
    from luxonis_ml.vizlab import is_video_path

    if plain is not None:
        return plain
    return save is not None and is_video_path(save)


def _write_renders(
    renders: "Iterable[tuple[str, Renderable]]",
    destination: Path,
    *,
    image_format: str,
    fps: float,
    background: "Color",
    empty_note: str,
) -> None:
    """Write headless renders to a directory of stills, or to a single clip.

    ``destination`` chooses between the two: an extension `VideoWriter` knows
    (see `luxonis_ml.vizlab.VIDEO_FORMATS`) encodes every render into one
    playable file, and anything else is treated as a directory that gets one
    ``image_format`` file per render. Shared by ``inspect`` and ``compare`` so
    the two save the same way.

    Args:
        renders: ``(source_name, render)`` pairs, in the order to write them.
        destination: The output clip or directory.
        image_format: Extension for the per-render files, ``png`` or ``svg``.
            Unused by the clip form, which takes its encoder from the extension.
        fps: Frame rate for the clip form.
        background: Color shown behind renders that do not fill the clip.
        empty_note: What to print when ``renders`` turns out to be empty.

    """
    from luxonis_ml.vizlab import VideoWriter, is_video_path

    if is_video_path(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        clip = VideoWriter(destination, fps=fps, background=background)
        try:
            for _, render in renders:
                clip.add(render)
        finally:
            # Reporting an empty run is the command's job, not the writer's: a
            # filter that matched nothing deserves the same note the directory
            # form prints rather than a traceback about the file it skipped.
            clip.close(quiet=True)
        written = len(clip)
        size = clip.size
        summary = (
            f"a {written}-frame clip ({clip.codec}, {size[0]}x{size[1]})"
            if size is not None
            else ""
        )
    else:
        destination.mkdir(parents=True, exist_ok=True)
        written = 0
        for source_name, render in renders:
            stem = f"{written:04d}_{Path(source_name).stem or 'image'}"
            render.save(destination / f"{stem}.{image_format}")
            written += 1
        summary = f"{written} render(s)"
    if not written:
        print(f"[yellow]{empty_note}[/yellow]")
        return
    print(f"[green]Saved {summary} to '{destination}'.[/green]")


@app.command
def inspect(
    name: str,
    *,
    view: Annotated[
        list[str] | None,
        Parameter(alias="-v", group=_DATASET_OPTIONS),
    ] = None,
    filters: Annotated[
        SampleFilterConfig,
        Parameter(name="*", group=_SAMPLE_FILTERS),
    ] = _NO_SAMPLE_FILTERS,
    aug_config: Annotated[
        Path | None,
        Parameter(
            alias="-a",
            group=_AUGMENTATION_OPTIONS,
            validator=validators.Path(
                exists=True, ext={".json", ".yaml", ".yml"}
            ),
        ),
    ] = None,
    size_multiplier: Annotated[
        float | Literal["auto"],
        Parameter(alias="-s", group=_VISUALIZATION_OPTIONS),
    ] = "auto",
    ignore_aspect_ratio: Annotated[
        bool,
        Parameter(
            alias="-i",
            negative="",
            group=_AUGMENTATION_OPTIONS,
        ),
    ] = False,
    deterministic: Annotated[
        bool,
        Parameter(alias="-d", negative="", group=_AUGMENTATION_OPTIONS),
    ] = False,
    force_update: Annotated[
        bool,
        Parameter(alias="-f", negative="", group=_DATASET_OPTIONS),
    ] = False,
    blend_all: Annotated[
        bool,
        Parameter(alias="-bl", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    per_instance: Annotated[
        bool,
        Parameter(alias="-pi", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    color_by: Annotated[
        Literal["class", "instance", "task"] | None,
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = None,
    prefetch: Annotated[int, Parameter(group=_VIEWER_OPTIONS)] = 2,
    list_augmentations: Annotated[
        bool,
        Parameter(negative="", group=_AUGMENTATION_OPTIONS),
    ] = True,
    skeletons: Annotated[
        bool,
        Parameter(negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    keypoint_labels: Annotated[
        Literal["none", "numbers", "names", "full"],
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = "none",
    legend: Annotated[
        bool,
        Parameter(alias="-lg", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    show_background: Annotated[
        bool,
        Parameter(alias="-bg", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    array_viz: Annotated[
        bool,
        Parameter(alias="-av", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    array_mode: Annotated[
        Literal["tile", "overlay"],
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = "tile",
    array_kind: Annotated[
        "list[tuple[str, _ArrayKindName]] | None",
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = None,
    array_gradient: Annotated[
        str | None,
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = None,
    array_vmin: Annotated[
        float | None,
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = None,
    array_vmax: Annotated[
        float | None,
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = None,
    array_ignore: Annotated[
        float | None,
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = None,
    array_source: Annotated[
        str | None,
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = None,
    array_colorbar: Annotated[
        bool,
        Parameter(
            negative="--no-array-colorbar", group=_VISUALIZATION_OPTIONS
        ),
    ] = True,
    fast: Annotated[
        bool,
        Parameter(alias="-fa", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    show_all: Annotated[
        bool,
        Parameter(alias="-sa", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    theme: Annotated[
        Literal["dark", "light"],
        Parameter(alias="-t", group=_VISUALIZATION_OPTIONS),
    ] = "dark",
    plain: Annotated[
        bool | None,
        Parameter(alias="-p", group=_VISUALIZATION_OPTIONS),
    ] = None,
    save: Annotated[
        Path | None,
        Parameter(alias="-o", group=_OUTPUT_OPTIONS),
    ] = None,
    save_format: Annotated[
        Literal["png", "svg"],
        Parameter(alias="-fmt", group=_OUTPUT_OPTIONS),
    ] = "png",
    fps: Annotated[
        float,
        Parameter(group=_OUTPUT_OPTIONS),
    ] = 5.0,
    bucket_storage: Annotated[
        BucketStorageT,
        Parameter(group=_DATASET_OPTIONS),
    ] = BucketStorage.LOCAL,
):
    """Inspect images and annotations in a dataset.

    Hovering the mouse over a detection that carries annotation metadata shows
    that metadata in a tooltip, so dense scenes stay uncluttered. With
    ``--color-by instance`` (or its ``--per-instance`` compatibility alias),
    every spatial detection gets a tooltip even when it has no custom metadata.

    Interactive controls (also listed, and clickable, in the side panel):

    - ``m`` / ``k`` / ``b`` / ``l`` — toggle masks / keypoints / boxes / labels.
    - ``d`` — toggle decluttering (on by default): in busy scenes, tiny
      detections ringed by many others are hidden as noise; press to show all.
    - ``a`` — toggle array fields, offered only for datasets that
      have them (see ``--array-viz``).
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
        filters: Shared task, class, annotation, metadata, confidence,
            instance-count, unlabeled, and text-search sample filters.
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
            color to each instance instead of each class. Compatibility alias
            for ``--color-by instance``.
        color_by: Assign colors by class (default), spatial instance, or task.
            Instance coloring hides the legend and adds identity tooltips. Task
            coloring blends tasks and shows an informational task legend.
        prefetch: Number of fully rendered samples to keep in a bounded
            background queue while the interactive viewer is open. Set to zero
            to disable render-ahead.
        list_augmentations: Show the augmentations applied to each
            displayed image. Requires '--aug-config' to be set.
        skeletons: Draw keypoint skeleton edges.
        keypoint_labels: Specify how to draw keypoint labels.
        legend: Draw a class or task color key on each image. Ignored in
            instance-color mode because colors identify individual instances.
        show_background: Render the semantic-segmentation background class
            (hidden by default) and include it in the palette and legend.
        array_viz: Draw array labels whose shape is understandable as a dense
            field — a disparity or depth map, an uncertainty field. Off by
            default because most arrays (embeddings, say) are not pictures;
            those are left undrawn however this is set.
        array_kind: How to read a given array, as ``<task> <kind>`` pairs;
            repeat the flag once per array. Overrides both the reserved task
            names (``disparity``, ``flow``, ``segmentation``, …) and the shape
            guess, and is scoped per task because one sample may hold several
            arrays of different kinds. Kinds are ``scalar``, ``signed`` (a
            diverging map centered on zero, for error fields), ``flow``,
            ``normals``, ``image``, ``scores`` (argmax of a per-class stack
            to a segmentation mask), and ``confidence`` (the same stack read
            as how sure the winning class was, rather than which it was), and
            ``class_confidence`` (both at once — class colors, faded where
            the model hesitated).
        array_mode: How ``--array-viz`` draws a field: ``tile`` gives it its own
            panel beside the images, ``overlay`` blends it onto the image. An
            overlay is only drawn when the field's proportions match the photo's,
            and only onto one source (see ``--array-source``).
        array_gradient: Colormap for array fields, e.g. ``viridis`` or
            ``turbo``. Also applies to any other heatmap in the scene.
        array_vmin: Pin the low end of the value range. Without it each field is
            scaled to its own minimum, which makes two datasets look alike even
            when their values differ — pin both ends to compare them.
        array_vmax: Pin the high end of the value range.
        array_ignore: A "no data" value in the field, drawn transparent and left
            out of the automatic range. Use ``--array-ignore 0`` for a stereo
            disparity map that marks unmatched pixels with zero.
        array_source: Which image source ``--array-mode overlay`` draws onto.
            Defaults to the first, since a field describes the reference view.
        array_colorbar: Draw a colour key beside each field, so its values can
            be read off the colours.
        fast: Lighter, much faster rendering for large or dense datasets: draw
            masks as fills only (no contour outlines), drop the soft drop shadows,
            and turn off shape anti-aliasing. Shape edges look slightly harder and
            text stays anti-aliased; every mask still shows (the ``m`` key still
            toggles them).
        show_all: Start with decluttering off, so tiny detections in crowded
            scenes are drawn instead of hidden (equivalent to pressing ``d`` up
            front). Toggle it back on any time with the ``d`` key.
        theme: Visual theme of the visualization: ``dark`` or ``light``.
        plain: Render just the image, without the side panel (controls, class
            legend, and sample metadata) or the rounded surface it is mounted
            on. Defaults to off, except when ``--save`` names a clip: a clip has
            one fixed canvas but the panel's width follows each sample's
            metadata, so keeping it would letterbox every frame that disagrees.
            Pass ``--no-plain`` to keep the panel in a clip anyway.
        save: Where to write renders instead of opening a window; headless
            either way. A path ending in a clip extension (``.mp4``, ``.webm``,
            ``.avi``, ``.mkv``, ``.mov``, ``.gif``, ``.webp``, ``.apng``,
            ``.avif``) encodes every sample into that single file; any other
            path is a directory, created if needed, that gets one file per
            source image.
        save_format: File format for the directory form of ``--save``: ``png``
            (raster) or ``svg`` (annotations and metadata panel as crisp vectors
            over the embedded photo, scalable to any zoom). Ignored when
            ``--save`` names a clip, whose extension already picks the encoder.
            Both forms follow ``--plain`` for the side panel.
        fps: Frames per second when ``--save`` names a clip. Every sample
            contributes one frame, so this sets how long each is on screen.
        bucket_storage: Storage type of the dataset.

    """
    check_exists(name, bucket_storage)

    view = view or ["train"]
    dataset = LuxonisDataset(name, bucket_storage=bucket_storage)

    if len(dataset) == 0:
        raise ValueError(f"Dataset '{name}' is empty.")

    if per_instance and color_by not in (None, "instance"):
        raise ValueError(
            "--per-instance cannot be combined with a different --color-by "
            f"value ({color_by!r})."
        )
    effective_color_by = "instance" if per_instance else color_by or "class"
    if prefetch < 0:
        raise ValueError("--prefetch must be non-negative.")
    # Refining something that is switched off would silently do nothing, so say
    # so rather than rendering a frame that ignores what was asked for.
    refinements = {
        "--array-mode": array_mode != "tile",
        "--array-kind": bool(array_kind),
        "--array-gradient": array_gradient is not None,
        "--array-vmin": array_vmin is not None,
        "--array-vmax": array_vmax is not None,
        "--array-ignore": array_ignore is not None,
        "--array-source": array_source is not None,
        "--no-array-colorbar": not array_colorbar,
    }
    if not array_viz and (
        given := [f for f, used in refinements.items() if used]
    ):
        raise ValueError(
            f"{', '.join(given)} requires --array-viz, which is off by default."
        )

    available_tasks = (
        dataset.get_task_names() if filters.task_name is not None else ()
    )
    available_classes = (
        (
            candidate
            for names in dataset.get_class_names().values()
            for candidate in names
        )
        if filters.class_name is not None
        else ()
    )
    filters.validate(
        available_tasks=available_tasks,
        available_classes=available_classes,
    )
    task_filter = filters.task_filter
    query = filters.query()

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
    # Class names per task, so an array whose channels ride the LDF class
    # axis comes back with those channels named rather than numbered.
    array_class_names = dataset.get_class_names()
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
            combine,
            order_by_position,
            resolve_gradient,
            set_default_options,
            visualize_record,
        )
        from luxonis_ml.vizlab.adapters.arrays import array_annotations
        from luxonis_ml.vizlab.adapters.instances import (
            records_to_colored_annotations,
            spatial_instances,
        )
        from luxonis_ml.vizlab.adapters.ldf import (
            metadata_annotations,
        )
        from luxonis_ml.vizlab.viewer import (
            LayerState,
            PrefetchIterator,
            Viewer,
        )
    except ImportError as e:
        raise SystemExit(
            "Visualization requires the 'viz' extra. "
            "Install it with `pip install luxonis-ml[viz]`."
        ) from e

    plain = _plain_by_default(plain, save)

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
        array_view=array_mode if array_viz else "off",
        array_vmin=array_vmin,
        array_vmax=array_vmax,
        array_ignore_value=array_ignore,
        array_overlay_source=array_source,
        array_colorbar=array_colorbar,
        array_kinds=tuple(array_kind or ()),
    )
    if array_gradient:
        # Resolved eagerly so a typo fails now, with the list of valid names,
        # rather than at the first render.
        options = options.replace(gradient=resolve_gradient(array_gradient))
    # Make single images, panels, and grid backgrounds all follow the options
    # (so bare Images in this command pick up the theme/palette too).
    set_default_options(options)
    # Identity coloring deliberately has its own sequence: the first instance or
    # task should receive the first generated color regardless of how many class
    # names were registered in the dataset palette above.
    identity_palette = Palette()

    # The legend reserves width for the dataset's longest class name (capped, so
    # one very long outlier does not blow the panel out), keeping the panel a
    # stable width sample to sample.
    longest_class = max(class_names, key=len, default="")[:24]
    legend_tasks = (
        dataset.get_task_names()
        if legend and effective_color_by == "task"
        else []
    )
    longest_task = max(legend_tasks, key=len, default="")[:24]

    def sidebar(
        panel: "Mapping[str, PanelData]",
        layers: "LayerState",
        *,
        task_names: Sequence[str] = (),
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
        if legend and effective_color_by == "class" and names:
            out["classes"] = Swatches(
                tuple((palette.color_for(name), name) for name in names),
                disabled=frozenset(layers.hidden),
                # Hold the legend (and panel) width to the dataset's longest class
                # name so it stays put as the per-sample class set changes.
                reserve=longest_class,
            )
        elif legend and effective_color_by == "task" and task_names:
            out["tasks"] = Swatches(
                tuple(
                    (identity_palette.color_for(name), name)
                    for name in task_names
                ),
                reserve=longest_task,
                interactive=False,
            )
        out.update(panel)
        return out

    def overlay_arrays(
        viz: Image,
        arrays: "Mapping[str, np.ndarray]",
        layers: "LayerState",
    ) -> None:
        """Blend any array field onto ``viz``, when ``--array-mode overlay``.

        The caller decides *which* source gets them; see `sample_tiles`.
        """
        if options.array_view != "overlay" or not arrays or not layers.arrays:
            return
        for drawing in array_annotations(
            arrays,
            options=options,
            image_shape=(viz.height, viz.width),
            class_names=array_class_names,
        ):
            # Added before the detections so the field paints beneath them.
            for annotation in drawing.annotations():
                viz.add(annotation)

    def array_tiles(
        arrays: "Mapping[str, np.ndarray]", layers: "LayerState"
    ) -> "tuple[list[Renderable], list[str]]":
        """Render each array field as its own tile, when ``--array-mode tile``."""
        if options.array_view != "tile" or not arrays or not layers.arrays:
            return [], []
        tiles: list[Renderable] = []
        titles: list[str] = []
        background = viz_theme.background
        for drawing in array_annotations(
            arrays, options=options, class_names=array_class_names
        ):
            field = drawing.field.field()
            if field is None:
                continue
            height, width = field.shape[-2:]
            # On the theme background rather than black, so a nodata pixel reads
            # as absent instead of as a low value.
            canvas = np.full(
                (height, width, 3),
                (background.r, background.g, background.b),
                np.uint8,
            )
            tile = Image(canvas, options=options)
            for annotation in drawing.annotations():
                tile.add(annotation)
            tiles.append(tile)
            titles.append(f"{drawing.task_name} {drawing.kind}")
        return tiles, titles

    def blend_annotations(
        image: np.ndarray,
        records: "Mapping[str, DatasetRecord]",
        layers: "LayerState",
        *,
        color_by: "ColorBy",
        arrays: "Mapping[str, np.ndarray] | None" = None,
    ) -> Image:
        """Draw every record's annotations onto one image, layer toggles applied.

        Shared by the interactive single/blended view and the headless save path.
        Detections from all tasks are blended together (a redundant classification
        chip is dropped next to boxes/keypoints/masks), the current ``layers``
        filter what is shown, and box-less metadata is added as hover-free cards.
        The image is returned unsized; the caller sets any display size.
        """
        viz = Image(image, options=options)
        overlay_arrays(viz, arrays or {}, layers)
        detections = records_to_colored_annotations(
            list(records.values()),
            color_by=color_by,
            options=options,
            identity_palette=identity_palette,
        )
        for annotation in layers.apply_layers(detections, palette):
            viz.add(annotation)
        if color_by != "instance":
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
        return _filter_records_by_task(
            records,
            task_filter,
            filters.task_name_mode,
        )

    def prepared_samples() -> "Iterable[_InspectionSample]":
        """Convert, filter, and snapshot panel data in loader order."""
        for data in loader:
            records = records_from_labels(data.labels)
            if not query.matches(
                records,
                data.metadata,
                extra_annotation_types=_loader_annotation_types(
                    data.labels,
                    task_filter,
                    filters.task_name_mode,
                ),
            ):
                continue
            # Build this before advancing the loader again: augmentation
            # provenance belongs to the current output and may be replaced by
            # the next augmentation call on the prefetch thread.
            panel = build_panel(data.labels, data.metadata)
            yield data, records, panel

    def sample_color_mode(
        records: "Mapping[str, DatasetRecord]",
    ) -> "ColorBy":
        """Resolve per-sample fallback for unsupported instance coloring."""
        if effective_color_by != "instance" or spatial_instances(
            records.values()
        ):
            return effective_color_by
        print(
            "[yellow]Warning: Instance coloring is not supported for this "
            "sample. Falling back to class colors.[/yellow]"
        )
        return "class"

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

    def save_grid(
        tiles: "list[Renderable]", titles: "list[str]"
    ) -> "Renderable":
        """Grid a sample's tiles for a file, at whatever size ``--size`` implies.

        The headless twin of `compose_tiles`: with no screen to fit to there is
        nothing to shrink toward, so tiles keep their (optionally scaled) size.
        """
        sized = [
            tile.render_at(save_size(tile.width, tile.height))
            for tile in tiles
        ]
        named: Mapping[str, Renderable] | Renderable = (
            dict(zip(_unique_titles(titles), sized, strict=True))
            if titles
            else sized[0]
        )
        return combine(named, bg=viz_theme.background)

    def source_tiles(
        source_name: str,
        image: np.ndarray,
        arrays: "Mapping[str, np.ndarray]",
        records: "Mapping[str, DatasetRecord]",
        layers: "LayerState",
        color_by: "ColorBy",
    ) -> "tuple[list[Renderable], list[str]]":
        """Build the tiles one image source contributes, with their titles.

        Usually one — the source with every task's detections blended onto it.
        A multi-task dataset in the default class-color mode instead gets one
        tile per record, so the tasks stay legible side by side.
        """
        # The viewer's interactive layer toggles (masks/keypoints/labels, a class
        # focus) filter what is drawn without disturbing the metadata cards,
        # legend, or panel — `blend_annotations` applies them to the detections.
        if color_by != "class" or blend_all or len(records) <= 1:
            return [
                blend_annotations(
                    image, records, layers, color_by=color_by, arrays=arrays
                )
            ], [source_name]
        tiles: list[Renderable] = []
        for record in records.values():
            tile = visualize_record(record, image, options=options)
            # Grid tiles carry no per-record panel here, so they are plain images
            # whose annotations the layer toggles filter.
            if isinstance(tile, Image):
                tile.annotations[:] = layers.apply_layers(
                    tile.annotations, palette
                )
            tiles.append(tile)
        return tiles, [f"{source_name} · {task}" for task in records]

    def sample_tiles(
        images: "Mapping[str, np.ndarray]",
        arrays: "Mapping[str, np.ndarray]",
        records: "Mapping[str, DatasetRecord]",
        layers: "LayerState",
        color_by: "ColorBy",
    ) -> "tuple[list[Renderable], list[str]]":
        """Build every tile of one sample: its sources, then any array fields."""
        tiles: list[Renderable] = []
        titles: list[str] = []
        # Source names carry their own placement: a rig storing `right` before
        # `left` should still be drawn left-to-right.
        images = order_by_position(images)
        # A field describes the reference view, so an overlay lands on exactly
        # one source; the others are built without it.
        overlay_source = options.array_overlay_source or next(iter(images), "")
        for source_name, source_image in images.items():
            source, names = source_tiles(
                source_name,
                source_image.astype(np.uint8),
                arrays if source_name == overlay_source else {},
                records,
                layers,
                color_by,
            )
            tiles.extend(source)
            titles.extend(names)
        field_tiles, field_titles = array_tiles(arrays, layers)
        tiles.extend(field_tiles)
        titles.extend(field_titles)
        # A lone tile needs no title band; several are only distinguishable with
        # one. Titles are dropped rather than blanked so the band collapses.
        return tiles, ([] if len(tiles) == 1 else titles)

    def headless_renders() -> "Iterable[tuple[str, Renderable]]":
        """Build one render per sample, paired with a name for its output file.

        Fully headless (no viewer, no screen): a sample's image sources — and
        any array field — are tiled into one render, framed with the metadata
        panel unless ``--plain``. Decluttering follows ``--show-all``. Shared by
        both save forms so a clip and a directory show exactly the same pixels.
        """
        layers = LayerState(declutter=not show_all)
        for data, records, panel in prepared_samples():
            layers.update_classes(_present_classes(records.values()))
            arrays = _array_labels(
                data.labels, task_filter, filters.task_name_mode
            )
            layers.has_arrays = bool(arrays)
            sample_color_by = sample_color_mode(records)
            tiles, titles = sample_tiles(
                data.images, arrays, records, layers, sample_color_by
            )
            if not tiles:
                continue
            if len(tiles) == 1:
                viz: Renderable = tiles[0]
                viz.render_at(save_size(viz.width, viz.height))
            else:
                viz = save_grid(tiles, titles)
            if not plain:
                viz = viz.with_panel(
                    sidebar(
                        panel,
                        layers,
                        task_names=tuple(records),
                        controls=False,
                    ),
                )
            yield next(iter(data.images), "image"), viz

    if save is not None:
        _write_renders(
            headless_renders(),
            save,
            image_format=save_format,
            fps=fps,
            background=viz_theme.background,
            empty_note="No samples matched the inspection filters.",
        )
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
        tiles: list[Renderable], titles: list[str], reserve: float
    ) -> Frame:
        """Hand a sample's tiles to vizlab, which picks the layout and sizing.

        The arrangement is not decided here: `combine` chooses the column count
        that shows the tiles largest inside the screen budget, orders positional
        source names, and drops the grid chrome when there is only one tile.
        """
        if size_multiplier != "auto":
            tiles = [
                tile.copy().render_at(
                    (
                        max(1, round(tile.width * size_multiplier)),
                        max(1, round(tile.height * size_multiplier)),
                    )
                )
                for tile in tiles
            ]
        named: Mapping[str, Renderable] | Renderable = (
            dict(zip(_unique_titles(titles), tiles, strict=True))
            if titles
            else tiles[0]
        )
        if size_multiplier == "auto" and screen is not None:
            return combine(
                named,
                target=(round(0.9 * screen[0]), round(0.9 * screen[1])),
                reserve=reserve,
                bg=viz_theme.background,
                allow_upscale=True,
            ).frame()
        return combine(named, bg=viz_theme.background).frame()

    def framed(
        frame: Frame,
        panel: "Mapping[str, PanelData]",
        task_names: Sequence[str],
        layers: "LayerState",
    ) -> Frame:
        """Attach the class legend (overlay) and metadata panel (right side).

        The panel (controls + classes + metadata) reframes the image as a rounded
        surface at a margin offset, so `Frame.with_panel` shifts the hover map to
        match. Unless ``--plain`` is active, it is always attached — the controls
        make it non-empty.
        """
        if plain:
            return frame
        return frame.with_panel(sidebar(panel, layers, task_names=task_names))

    def build_frame(
        images: "Mapping[str, np.ndarray]",
        arrays: "Mapping[str, np.ndarray]",
        records: "Mapping[str, DatasetRecord]",
        panel: "Mapping[str, PanelData]",
        reserve: float,
        layers: "LayerState",
        color_by: "ColorBy",
    ) -> Frame:
        """Compose one sample's tiles into a single `Frame`."""
        tiles, titles = sample_tiles(images, arrays, records, layers, color_by)
        return framed(
            compose_tiles(tiles, titles, reserve),
            panel,
            tuple(records),
            layers,
        )

    def render_samples(
        samples: "Iterable[_InspectionSample]",
    ) -> "Iterable[_PreparedInspectionSample]":
        """Build display-ready frames in source order."""
        for data, records, panel in samples:
            layers = viewer.layers.copy()
            layers.update_classes(_present_classes(records.values()))
            arrays = _array_labels(
                data.labels, task_filter, filters.task_name_mode
            )
            layers.has_arrays = bool(arrays)
            color_by = sample_color_mode(records)
            reserve = panel_reserve if not plain else 0.0
            yield _PreparedInspectionSample(
                images=data.images,
                arrays=arrays,
                records=records,
                panel=panel,
                layers=layers,
                color_by=color_by,
                frame=viewer.prepare(
                    build_frame(
                        data.images,
                        arrays,
                        records,
                        panel,
                        reserve,
                        layers,
                        color_by,
                    )
                ),
            )

    def inspect_samples(
        samples: "Iterable[_PreparedInspectionSample]",
    ) -> int:
        """Present rendered-ahead samples until exhaustion or user exit."""
        shown = 0
        for sample in samples:
            viewer.layers.update_classes(sample.layers.classes)
            # Content state, not a toggle: without syncing it the live and
            # snapshot states never compare equal for a sample that has arrays,
            # and every frame would be re-rendered instead of using the prefetch.
            viewer.layers.has_arrays = sample.layers.has_arrays
            layers_match = viewer.layers == sample.layers
            reserve = panel_reserve if not plain else 0.0

            def rerender(
                _layers: "LayerState",
                *,
                sources: "Mapping[str, np.ndarray]" = sample.images,
                sample_arrays: "Mapping[str, np.ndarray]" = sample.arrays,
                sample_records: "Mapping[str, DatasetRecord]" = sample.records,
                sample_panel: "Mapping[str, PanelData]" = sample.panel,
                panel_width: float = reserve,
                identity: "ColorBy" = sample.color_by,
            ) -> Frame:
                return build_frame(
                    sources,
                    sample_arrays,
                    sample_records,
                    sample_panel,
                    panel_width,
                    _layers,
                    identity,
                )

            prepared = (
                sample.frame
                if layers_match
                else viewer.prepare(rerender(viewer.layers))
            )
            viewer.show_prepared(
                name,
                prepared,
                # Default-bound values keep the window attached to its own
                # sample when layer toggles trigger a re-render.
                render=rerender,
            )
            if sample.images:
                shown += 1
                if viewer.wait() == "q":
                    break
        return shown

    try:
        if prefetch:
            with PrefetchIterator(
                render_samples(prepared_samples()),
                capacity=prefetch,
            ) as samples:
                shown = inspect_samples(samples)
        else:
            shown = inspect_samples(render_samples(prepared_samples()))
        if shown == 0 and query.active:
            print(
                "[yellow]No samples matched the inspection filters.[/yellow]"
            )
    finally:
        viewer.close()


@app.command
def compare(
    name: str,
    predictions: str,
    *,
    view: Annotated[
        list[str] | None,
        Parameter(alias="-v", group=_DATASET_OPTIONS),
    ] = None,
    filters: Annotated[
        SampleFilterConfig,
        Parameter(name="*", group=_SAMPLE_FILTERS),
    ] = _NO_SAMPLE_FILTERS,
    layout: Annotated[
        Literal["overlay", "dual", "triple"],
        Parameter(alias="-l", group=_VISUALIZATION_OPTIONS),
    ] = "overlay",
    iou_threshold: Annotated[
        float,
        Parameter(alias="--iou", group=_MATCHING_OPTIONS),
    ] = 0.5,
    score_threshold: Annotated[
        float,
        Parameter(alias="--score", group=_MATCHING_OPTIONS),
    ] = 0.25,
    class_agnostic: Annotated[
        bool,
        Parameter(negative="", group=_MATCHING_OPTIONS),
    ] = False,
    per_class: Annotated[
        bool,
        Parameter(alias="-pc", negative="", group=_REPORT_OPTIONS),
    ] = False,
    errors_only: Annotated[
        bool,
        Parameter(alias="-e", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    summary: Annotated[
        bool,
        Parameter(negative="", group=_REPORT_OPTIONS),
    ] = False,
    size_multiplier: Annotated[
        float | Literal["auto"],
        Parameter(alias="-s", group=_VISUALIZATION_OPTIONS),
    ] = "auto",
    skeletons: Annotated[
        bool,
        Parameter(
            negative="--no-skeletons",
            group=_VISUALIZATION_OPTIONS,
        ),
    ] = True,
    keypoint_labels: Annotated[
        Literal["none", "numbers", "names", "full"],
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = "none",
    legend: Annotated[
        bool,
        Parameter(alias="-lg", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    show_background: Annotated[
        bool,
        Parameter(alias="-bg", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    theme: Annotated[
        Literal["dark", "light"],
        Parameter(alias="-t", group=_VISUALIZATION_OPTIONS),
    ] = "dark",
    plain: Annotated[
        bool | None,
        Parameter(alias="-p", group=_VISUALIZATION_OPTIONS),
    ] = None,
    save: Annotated[
        Path | None,
        Parameter(alias="-o", group=_OUTPUT_OPTIONS),
    ] = None,
    save_format: Annotated[
        Literal["png", "svg"],
        Parameter(alias="-fmt", group=_OUTPUT_OPTIONS),
    ] = "png",
    fps: Annotated[
        float,
        Parameter(group=_OUTPUT_OPTIONS),
    ] = 5.0,
    force_update: Annotated[
        bool,
        Parameter(alias="-f", negative="", group=_DATASET_OPTIONS),
    ] = False,
    bucket_storage: Annotated[
        BucketStorageT,
        Parameter(group=_DATASET_OPTIONS),
    ] = BucketStorage.LOCAL,
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
        filters: Shared task, class, annotation, metadata, confidence,
            instance-count, unlabeled, and text-search sample filters. A paired
            sample is selected when either its ground-truth or prediction side
            matches. Task names scope annotations on both sides.
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
        plain: Render just the verdict frame, without the metrics side panel or
            the rounded surface it is mounted on. Defaults to off, except when
            ``--save`` names a clip, whose single fixed canvas the panel would
            otherwise letterbox every disagreeing frame against. The panel is
            where precision, recall, and the TP/FP/FN counts live, so reach for
            ``--no-plain`` when the clip is meant to carry them.
        save: Where to write the comparison frames instead of opening a window;
            headless either way. A path ending in a clip extension (``.mp4``,
            ``.webm``, ``.avi``, ``.mkv``, ``.mov``, ``.gif``, ``.webp``,
            ``.apng``, ``.avif``) encodes the whole comparison into that single
            file — a scrubbable record of how a model did across the view. Any
            other path is a directory, created if needed, that gets one file per
            compared image.
        save_format: File format for the directory form of ``--save``: ``png``
            or ``svg``. Ignored when ``--save`` names a clip.
        fps: Frames per second when ``--save`` names a clip.
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

    available_tasks = (
        (
            *gt_dataset.get_task_names(),
            *pred_dataset.get_task_names(),
        )
        if filters.task_name is not None
        else ()
    )
    available_classes = (
        (
            candidate
            for dataset in (gt_dataset, pred_dataset)
            for names in dataset.get_class_names().values()
            for candidate in names
        )
        if filters.class_name is not None
        else ()
    )
    filters.validate(
        available_tasks=available_tasks,
        available_classes=available_classes,
    )
    task_filter = filters.task_filter
    query = filters.query()

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
            combine,
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

    plain = _plain_by_default(plain, save)

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
        records = loader_output_to_records(
            data.labels,
            classes=classes,
            categorical_encodings=categorical,
            render_background=show_background,
        )
        return _filter_records_by_task(
            records,
            task_filter,
            filters.task_name_mode,
        )

    def sample_matches(
        data: "LoaderOutput",
        classes: _ClassMappings,
        categorical: _ClassMappings,
    ) -> bool:
        """Whether one side of a comparison pair passes sample filters."""
        if not query.active:
            return True
        return query.matches(
            records_for(data, classes, categorical),
            data.metadata,
            extra_annotation_types=_loader_annotation_types(
                data.labels,
                task_filter,
                filters.task_name_mode,
            ),
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
        loader: LuxonisLoader,
        dataset_name: str,
        classes: _ClassMappings,
        categorical: _ClassMappings,
    ) -> tuple[
        dict[_SampleIdentity, int],
        dict[_SampleIdentity, bool],
    ]:
        """Map unique identities to loader indices and filter decisions."""
        indexed: dict[_SampleIdentity, int] = {}
        selected: dict[_SampleIdentity, bool] = {}
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
            selected[identity] = sample_matches(data, classes, categorical)
        return indexed, selected

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

    gt_indices, gt_selected = identity_index(
        gt_loader,
        name,
        gt_classes,
        gt_categorical,
    )
    pred_indices, pred_selected = identity_index(
        pred_loader,
        predictions,
        pred_classes,
        pred_categorical,
    )
    gt_identities = set(gt_indices)
    pred_identities = set(pred_indices)
    report_unpaired(
        {
            identity
            for identity in gt_identities - pred_identities
            if gt_selected[identity]
        },
        description="Missing prediction samples",
    )
    report_unpaired(
        {
            identity
            for identity in pred_identities - gt_identities
            if pred_selected[identity]
        },
        description="Extra prediction samples",
    )
    shared_identities = sorted(
        gt_identities & pred_identities, key=lambda item: gt_indices[item]
    )
    if not shared_identities:
        raise ValueError(
            "The ground-truth and prediction datasets have no samples in "
            "common by source filename."
        )
    shared = [
        identity
        for identity in shared_identities
        if gt_selected[identity] or pred_selected[identity]
    ]
    if not shared:
        raise ValueError("No paired samples matched the comparison filters.")

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

    # ``--save`` renders the same frames headlessly, so the viewer — and the
    # screen probe that opens a window to find the display size — is only built
    # for the interactive path.
    viewer = None if save is not None else Viewer()
    screen = viewer.screen if viewer is not None else None
    # The metrics panel is a fixed pixel width; reserve room for it when fitting.
    panel_reserve = 0.0 if plain else 400.0

    def display_size(width: int, height: int) -> tuple[int, int] | None:
        """Fit ``width`` x ``height`` to the screen (or apply the multiplier).

        There is no screen to fit to when saving, so ``auto`` falls through to
        ``None`` and keeps the source resolution.
        """
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

    def source_scene(
        image: np.ndarray,
        result: "ComparisonResult",
    ) -> "Renderable":
        """Draw one image source's verdict scene, without the metrics panel."""
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
            display.add(class_legend.model_copy())
        return display

    def build_frame(
        images: "Mapping[str, np.ndarray]",
        gt_records: "Mapping[str, DatasetRecord]",
        pred_records: "Mapping[str, DatasetRecord]",
    ) -> Frame:
        """Match GT vs predictions and compose every source into one `Frame`.

        A multi-source sample is tiled rather than given a window per source,
        so the metrics panel is attached once and the sources stay side by side.
        """
        result = match_sample(gt_records, pred_records)
        scenes = [
            source_scene(image.astype(np.uint8), result)
            for image in images.values()
        ]
        # One source or several, vizlab decides the arrangement -- and orders
        # positional source names, so a stereo pair stays left-then-right.
        composed: Renderable = combine(
            dict(zip(images, scenes, strict=True))
            if len(scenes) > 1
            else scenes[0],
            bg=viz_theme.background,
        )
        if plain:
            return composed.frame()
        metrics: dict[str, PanelData] = dict(result.summary())
        if per_class and len(result.per_class) > 1:
            metrics["by class"] = result.per_class_panel()
        return composed.with_panel(metrics, title="Comparison").frame()

    def paired(
        identity: "_SampleIdentity",
    ) -> "tuple[LoaderOutput, Mapping[str, DatasetRecord], Mapping[str, DatasetRecord]]":
        """Load one paired sample: its images, then both sides' records."""
        gt_data = gt_loader[gt_indices[identity]]
        pred_data = pred_loader[pred_indices[identity]]
        return (
            gt_data,
            records_for(gt_data, gt_classes, gt_categorical),
            records_for(pred_data, pred_classes, pred_categorical),
        )

    if save is not None:

        def compared_frames() -> "Iterable[tuple[str, Renderable]]":
            """Build every comparison scene in view order, headlessly.

            `Frame.image` unwraps the composed scene from the hover/click maps
            that only a `Viewer` consumes — and that a saved file has no way to
            carry anyway.
            """
            for identity in shared:
                gt_data, gt_records, pred_records = paired(identity)
                yield (
                    next(iter(gt_data.images), "image"),
                    build_frame(
                        gt_data.images, gt_records, pred_records
                    ).image,
                )

        _write_renders(
            compared_frames(),
            save,
            image_format=save_format,
            fps=fps,
            background=viz_theme.background,
            empty_note="No paired samples matched the comparison filters.",
        )
        return

    assert viewer is not None
    for identity in shared:
        gt_data, gt_records, pred_records = paired(identity)
        viewer.show(
            name, build_frame(gt_data.images, gt_records, pred_records)
        )
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

    # Only spatial annotations are plotted. Metadata can have class counts, but
    # it has no meaningful health plot and should not create a window. A spatial
    # task whose annotations yielded no heatmap points still has a class
    # distribution worth showing, so the task types decide, not the heatmaps.
    all_task_names = sorted(
        {
            task_name
            for task_name, by_type in chain(
                stats["class_distributions"].items(), stats["heatmaps"].items()
            )
            if not HEATMAP_TASK_TYPES.isdisjoint(by_type)
        }
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
        if HEATMAP_TASK_TYPES.isdisjoint(
            set(class_dist_by_type) | set(heatmaps_by_type)
        ):
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
