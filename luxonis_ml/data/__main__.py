import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import chain
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
)

import numpy as np
import rich.box
from cyclopts import App, Group, Parameter, validators
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
from luxonis_ml.data.utils.data_utils import HEATMAP_TASK_TYPES
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.data.utils.inspection import (
    InspectionAnnotationType,
    NameFilterMode,
    SampleFilterConfig,
    SampleIdentity,
    identity_index,
    identity_label,
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
        Frame,
        Renderable,
        Theme,
    )
    from luxonis_ml.vizlab.adapters.instances import ColorBy
    from luxonis_ml.vizlab.layout.panel import PanelData
    from luxonis_ml.vizlab.viewer import LayerState, ViewerSample

    _InspectionSample: TypeAlias = tuple[
        LoaderOutput,
        dict[str, DatasetRecord],
        dict[str, PanelData],
    ]

_T = TypeVar("_T")

app = App(help="Dataset utilities.")


BucketStorageT: TypeAlias = Annotated[BucketStorage, Parameter(alias="-b")]
#: Where the viewer's ``s`` key collects saved frames, under a per-dataset
#: subdirectory. Created on the first save, so a session that never saves
#: leaves nothing behind.
_SAVE_ROOT = Path("luxonis-inspect")
#: Said once per interactive session: clicking is the only interaction the
#: panel's control list cannot show, since every row there is keyed by a key.
_PICK_HINT = (
    "[dim]Click an annotation to print its JSON here"
    " (and copy it to the clipboard).[/dim]"
)
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
#: The class-color schemes ``--palette`` offers: the colorblind-safe sets in
#: `luxonis_ml.utils.color.PALETTES`, the unbounded ``cvd`` search behind them,
#: and ``default`` for whichever scheme the chosen theme already carries.
#: Spelled out (rather than read off the registry) because this module must
#: import without the ``viz`` extra installed; a test keeps the two in step.
_PaletteName: TypeAlias = Literal[
    "default",
    "cvd",
    "okabe-ito",
    "tol-bright",
    "tol-high-contrast",
    "tol-muted",
    "tol-vibrant",
]
_ClassMappings: TypeAlias = dict[str, dict[str, int]]
_NO_SAMPLE_FILTERS = SampleFilterConfig()
_DATASET_OPTIONS = Group("Dataset options", sort_key=10)
_SAMPLE_FILTERS = Group("Sample filters", sort_key=20)
_AUGMENTATION_OPTIONS = Group("Augmentation options", sort_key=30)
_MATCHING_OPTIONS = Group("Matching options", sort_key=30)
_VISUALIZATION_OPTIONS = Group("Visualization options", sort_key=40)
# Options that only bite on one kind of label, listed after the ones that apply
# to every scene: a dataset without keypoints (or masks, or arrays) can skip the
# whole panel rather than read past its flags.
_KEYPOINT_OPTIONS = Group("Keypoint options", sort_key=41)
_SEGMENTATION_OPTIONS = Group("Segmentation options", sort_key=42)
_ARRAY_OPTIONS = Group("Array options", sort_key=43)
_VIEWER_OPTIONS = Group("Viewer options", sort_key=50)
_REPORT_OPTIONS = Group("Reporting options", sort_key=50)
_OUTPUT_OPTIONS = Group("Output options", sort_key=60)


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


def _sample_stem(sample_metadata: "Params") -> "str | None":
    """Name a sample after its source image(s), for an interactive save.

    The loader reports ``filenames`` as a ``{source_name: basename}`` mapping
    (see `_flatten_filenames`), so a saved frame can carry the name of the image
    it came from rather than the window's. Multi-source samples join their
    stems, since one frame tiles all of them.

    Args:
        sample_metadata: One sample's metadata.

    Returns:
        The joined image stems, or ``None`` when the loader reported no
        filenames (the viewer then falls back to the window name).

    """
    files = sample_metadata.get("filenames")
    if not isinstance(files, Mapping) or not files:
        return None
    return "-".join(Path(str(name)).stem for name in files.values())


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


def _with_lookahead(
    items: "Iterable[_T]",
) -> "Iterator[tuple[_T, _T | None]]":
    """Yield each item paired with the one after it (``None`` at the end).

    Keeps two items alive at once rather than the whole stream, which matters
    when each one holds a sample's decoded pixels.

    Args:
        items: The stream to walk.

    Yields:
        ``(item, next_item)`` pairs, in order.

    """
    held: _T | None = None
    for item in items:
        if held is not None:
            yield held, item
        held = item
    if held is not None:
        yield held, None


def _write_renders(
    renders: "Iterable[tuple[str, Renderable]]",
    destination: Path,
    *,
    image_format: str,
    fps: float,
    background: "Color",
    theme: "Theme",
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
        image_format: Extension for the per-render files, ``png``, ``svg``, or
            ``html``. Unused by the clip form, which takes its encoder from the
            extension.
        fps: Frame rate for the clip form.
        background: Color shown behind renders that do not fill the clip.
        theme: Theme the renders used, so an HTML index matches them.
        empty_note: What to print when ``renders`` turns out to be empty.

    """
    from luxonis_ml.vizlab import VideoWriter, is_video_path
    from luxonis_ml.vizlab.scene.html import Nav

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
        pages: list[tuple[str, str]] = []
        as_pages = image_format in {"html", "htm"}

        def page_name(index: int, source_name: str) -> str:
            stem = f"{index:04d}_{Path(source_name).stem or 'image'}"
            return f"{stem}.{image_format}"

        # A page has to link the one after it, whose name is only known once
        # the next render arrives — so hold one behind rather than materialize
        # every sample just to count them.
        for (source_name, render), following in _with_lookahead(renders):
            name = page_name(written, source_name)
            if as_pages:
                nav = Nav(
                    index="index.html",
                    previous=pages[-1][0] if pages else None,
                    following=(
                        page_name(written + 1, following[0])
                        if following is not None
                        else None
                    ),
                    position=written + 1,
                )
                (destination / name).write_text(
                    render.render_html(nav=nav, title=source_name),
                    encoding="utf-8",
                )
            else:
                render.save(destination / name)
            pages.append((name, source_name))
            written += 1
        summary = f"{written} render(s)"
        if pages and as_pages:
            # A directory of numbered pages is not something anyone opens one
            # at a time; this is the file you actually hand someone.
            from luxonis_ml.vizlab.scene.html import index_document

            (destination / "index.html").write_text(
                index_document(pages, theme, title=destination.name),
                encoding="utf-8",
            )
            summary += " and an index"
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
    skeletons: Annotated[
        bool,
        Parameter(negative="", group=_KEYPOINT_OPTIONS),
    ] = False,
    keypoint_labels: Annotated[
        Literal["none", "numbers", "names", "full"],
        Parameter(group=_KEYPOINT_OPTIONS),
    ] = "none",
    legend: Annotated[
        bool,
        Parameter(group=_VISUALIZATION_OPTIONS),
    ] = True,
    show_background: Annotated[
        bool,
        Parameter(alias="-bg", negative="", group=_SEGMENTATION_OPTIONS),
    ] = False,
    array_viz: Annotated[
        bool,
        Parameter(alias="-av", negative="", group=_ARRAY_OPTIONS),
    ] = False,
    array_mode: Annotated[
        Literal["tile", "overlay"],
        Parameter(group=_ARRAY_OPTIONS),
    ] = "tile",
    array_kind: Annotated[
        "list[tuple[str, _ArrayKindName]] | None",
        Parameter(group=_ARRAY_OPTIONS),
    ] = None,
    array_gradient: Annotated[
        str | None,
        Parameter(group=_ARRAY_OPTIONS),
    ] = None,
    array_vmin: Annotated[
        float | None,
        Parameter(group=_ARRAY_OPTIONS),
    ] = None,
    array_vmax: Annotated[
        float | None,
        Parameter(group=_ARRAY_OPTIONS),
    ] = None,
    array_ignore: Annotated[
        float | None,
        Parameter(group=_ARRAY_OPTIONS),
    ] = None,
    array_source: Annotated[
        str | None,
        Parameter(group=_ARRAY_OPTIONS),
    ] = None,
    array_colorbar: Annotated[
        bool,
        Parameter(negative="--no-array-colorbar", group=_ARRAY_OPTIONS),
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
    palette: Annotated[
        _PaletteName,
        Parameter(alias="-pl", group=_VISUALIZATION_OPTIONS),
    ] = "default",
    plain: Annotated[
        bool | None,
        Parameter(alias="-p", group=_VISUALIZATION_OPTIONS),
    ] = None,
    save: Annotated[
        Path | None,
        Parameter(alias="-o", group=_OUTPUT_OPTIONS),
    ] = None,
    save_format: Annotated[
        Literal["png", "svg", "html"],
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
    - ``s`` — save the current view as a PNG under
      ``luxonis-inspect/<dataset>/``, named after the sample's source image and
      numbered so saving the same sample twice does not overwrite.
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
        palette: Which colors the classes are drawn in, picked independently of
            ``--theme``. ``default`` keeps the theme's own distinct-hue scheme,
            which spreads hues as widely as it can but promises nothing about
            how they look to a colorblind viewer. Every other value does:
            ``okabe-ito``, ``tol-bright``, ``tol-high-contrast``,
            ``tol-vibrant``, and ``tol-muted`` are published qualitative sets,
            short by nature (three to nine colors) and continued past their last
            color by the search that ``cvd`` runs from the first one — which
            picks every color to stay as far as it can from the ones already
            given out, under normal vision and all three dichromacies. Reach for
            a named set with a handful of classes, and for ``cvd`` with dozens.
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
            (raster), ``svg`` (annotations and metadata panel as crisp vectors
            over the embedded photo, scalable to any zoom), or ``html`` (that
            same vector render in a self-contained page whose annotations stay
            **hoverable**, so a saved sample keeps the tooltips the interactive
            viewer shows). Ignored when ``--save`` names a clip, whose extension
            already picks the encoder. Every form follows ``--plain`` for the
            side panel.
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
            Hints,
            MaskOutline,
            Palette,
            RenderOptions,
            SampleComposer,
            resolve_generator,
            resolve_gradient,
            set_default_options,
        )
        from luxonis_ml.vizlab.viewer import (
            LayerState,
            Viewer,
            ViewerSample,
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
        viz_theme = viz_theme.with_style(
            viz_theme.style.merge(mask_outline=MaskOutline.NONE, shadow=False)
        )
    color_generator = (
        viz_theme.palette.generator
        if palette == "default"
        else resolve_generator(palette)
    )
    # The palette is pinned to the dataset's class order for stable colors; it
    # lives on the theme, and the whole bundle is the render options.
    class_palette = Palette(class_names, generator=color_generator)
    options = RenderOptions(
        theme=viz_theme.with_palette(class_palette),
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

    legend_tasks = (
        dataset.get_task_names()
        if legend and effective_color_by == "task"
        else []
    )
    composer = SampleComposer(
        options=options,
        # Identity coloring deliberately has its own sequence: the first
        # instance or task should receive the first generated color regardless
        # of how many class names were registered in the dataset palette above.
        # Same scheme, though — --palette is about how the colors are chosen,
        # not about what they name.
        identity_palette=Palette(generator=color_generator),
        color_by=effective_color_by,
        array_class_names=array_class_names,
        legend=legend,
        blend_all=blend_all,
        panel=not plain,
        scale=size_multiplier,
        # The legend reserves width for the dataset's longest class name
        # (capped, so one very long outlier does not blow the panel out),
        # keeping the panel a stable width sample to sample.
        reserve_class=max(class_names, key=len, default="")[:24],
        reserve_task=max(legend_tasks, key=len, default="")[:24],
    )

    def build_panel(sample_metadata: "Params") -> "dict[str, PanelData]":
        tracked_augmentations = get_tracked_augmentations(sample_metadata)
        if tracked_augmentations is not None:
            sample_metadata = {
                key: value
                for key, value in sample_metadata.items()
                if key != "augmentations"
            }
        panel = (
            dict(_present_sample_metadata(sample_metadata))
            if sample_metadata
            else {}
        )
        if tracked_augmentations:
            panel["augmentations"] = Hints(
                tuple(tracked_augmentations.items())
            )
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
            panel = build_panel(data.metadata)
            yield data, records, panel

    def sample_color_mode(
        records: "Mapping[str, DatasetRecord]",
    ) -> "ColorBy":
        """Resolve per-sample fallback for unsupported instance coloring."""
        color_by = composer.fallback_color_by(records)
        if color_by != effective_color_by:
            print(
                "[yellow]Warning: Instance coloring is not supported for this "
                "sample. Falling back to class colors.[/yellow]"
            )
        return color_by

    if save is not None:

        def headless_renders() -> "Iterable[tuple[str, Renderable]]":
            """Build one render per sample, paired with a name for its file.

            Decluttering follows ``--show-all``. Shared by both save forms so
            a clip and a directory show exactly the same pixels.
            """
            layers = LayerState(declutter=not show_all)
            for data, records, panel in prepared_samples():
                layers.update_classes(_present_classes(records.values()))
                arrays = _array_labels(
                    data.labels, task_filter, filters.task_name_mode
                )
                layers.has_arrays = bool(arrays)
                viz = composer.render(
                    data.images,
                    arrays,
                    records,
                    panel,
                    layers,
                    sample_color_mode(records),
                )
                if viz is not None:
                    yield next(iter(data.images), "image"), viz

        _write_renders(
            headless_renders(),
            save,
            image_format=save_format,
            fps=fps,
            background=viz_theme.background,
            theme=viz_theme,
            empty_note="No samples matched the inspection filters.",
        )
        return

    # vizlab owns layout, screen-fit sizing, hover hit-testing, prefetch, and
    # the interactive window loop; this command only prepares data and hands
    # per-sample frame builders to the viewer.
    # The controls live in the side panel (see `SampleComposer.sidebar`), so
    # the viewer does not also float its HUD over the image.
    viewer = Viewer(hud=False, save_dir=_SAVE_ROOT / name)
    print(_PICK_HINT)
    # Decluttering hides tiny detections in crowded scenes by default; --show-all
    # starts with it off (the `d` key still toggles it live either way).
    viewer.layers.declutter = not show_all
    composer = replace(composer, screen=viewer.screen)

    def viewer_samples() -> "Iterable[ViewerSample]":
        """Bind each sample to its own frame builder, in loader order."""
        for data, records, panel in prepared_samples():
            arrays = _array_labels(
                data.labels, task_filter, filters.task_name_mode
            )
            color_by = sample_color_mode(records)

            def render(
                layers: "LayerState",
                *,
                sources: "Mapping[str, np.ndarray]" = data.images,
                sample_arrays: "Mapping[str, np.ndarray]" = arrays,
                sample_records: "Mapping[str, DatasetRecord]" = records,
                sample_panel: "Mapping[str, PanelData]" = panel,
                identity: "ColorBy" = color_by,
            ) -> "Frame":
                # Default-bound values keep the window attached to its own
                # sample when layer toggles trigger a re-render.
                return composer.frame(
                    sources,
                    sample_arrays,
                    sample_records,
                    sample_panel,
                    layers,
                    identity,
                )

            yield ViewerSample(
                render=render,
                classes=_present_classes(records.values()),
                has_arrays=bool(arrays),
                save_as=_sample_stem(data.metadata),
                wait=bool(data.images),
            )

    try:
        shown = viewer.present(name, viewer_samples(), prefetch=prefetch)
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
            group=_KEYPOINT_OPTIONS,
        ),
    ] = True,
    keypoint_labels: Annotated[
        Literal["none", "numbers", "names", "full"],
        Parameter(group=_KEYPOINT_OPTIONS),
    ] = "none",
    legend: Annotated[
        bool,
        Parameter(alias="-lg", negative="", group=_VISUALIZATION_OPTIONS),
    ] = False,
    show_background: Annotated[
        bool,
        Parameter(alias="-bg", negative="", group=_SEGMENTATION_OPTIONS),
    ] = False,
    theme: Annotated[
        Literal["dark", "light"],
        Parameter(alias="-t", group=_VISUALIZATION_OPTIONS),
    ] = "dark",
    palette: Annotated[
        _PaletteName,
        Parameter(alias="-pl", group=_VISUALIZATION_OPTIONS),
    ] = "default",
    plain: Annotated[
        bool | None,
        Parameter(alias="-p", group=_VISUALIZATION_OPTIONS),
    ] = None,
    save: Annotated[
        Path | None,
        Parameter(alias="-o", group=_OUTPUT_OPTIONS),
    ] = None,
    save_format: Annotated[
        Literal["png", "svg", "html"],
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
    red miss, dashed-amber false alarm, orange class error — with a metrics side
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
        palette: Which colors the class legend is drawn in, picked
            independently of ``--theme``: ``default`` keeps the theme's own
            distinct-hue scheme, while ``okabe-ito``, ``tol-bright``,
            ``tol-high-contrast``, ``tol-vibrant``, ``tol-muted``, and ``cvd``
            are colorblind-safe (see ``luxonis_ml data inspect --help``). It
            colors classes, not outcomes: the verdict colors mean one thing
            each, and ``dual`` layout keys its two panels to per-match identity
            colors, so neither follows this.
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
        save_format: File format for the directory form of ``--save``: ``png``,
            ``svg``, or ``html`` — the vector render in a self-contained page
            that keeps each match's verdict tooltip hoverable. Ignored when
            ``--save`` names a clip.
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
            ComparisonComposer,
            ComparisonReport,
            Legend,
            Palette,
            RenderOptions,
            Verdict,
            confusion_matrix_figure,
            resolve_generator,
            set_default_options,
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
    # As in `inspect`: the color scheme is picked independently of the theme,
    # and `default` keeps whichever scheme the theme itself carries.
    class_palette = Palette(
        class_names,
        generator=(
            viz_theme.palette.generator
            if palette == "default"
            else resolve_generator(palette)
        ),
    )
    options = RenderOptions(
        theme=viz_theme.with_palette(class_palette),
        skeletons=gt_dataset.get_skeletons(),
        keypoint_label_mode=keypoint_labels,
        draw_skeletons=skeletons,
    )
    set_default_options(options)
    class_legend = (
        Legend(entries=class_names, palette=class_palette, title="classes")
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

    composer = ComparisonComposer(
        options=options,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_aware=not class_agnostic,
        show=layout_show,
        verdicts=(
            {Verdict.FP, Verdict.FN, Verdict.CLASS_ERROR}
            if errors_only
            else None
        ),
        legend=class_legend,
        per_class=per_class,
        panel=not plain,
        scale=size_multiplier,
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

    def report_unpaired(
        identities: "set[SampleIdentity]", *, description: str
    ) -> None:
        if not identities:
            return
        ordered = sorted(identity_label(identity) for identity in identities)
        preview = ", ".join(ordered[:10])
        remainder = len(ordered) - 10
        if remainder > 0:
            preview += f", and {remainder} more"
        print(f"[yellow]{description} ({len(ordered)}): {preview}.[/yellow]")

    def index_side(
        loader: LuxonisLoader,
        dataset_name: str,
        classes: _ClassMappings,
        categorical: _ClassMappings,
    ) -> "tuple[dict[SampleIdentity, int], dict[SampleIdentity, bool]]":
        """Index one side's identities, applying the sample filters."""
        return identity_index(
            loader,
            dataset_name,
            matches=(
                (lambda data: sample_matches(data, classes, categorical))
                if query.active
                else None
            ),
        )

    gt_indices, gt_selected = index_side(
        gt_loader,
        name,
        gt_classes,
        gt_categorical,
    )
    pred_indices, pred_selected = index_side(
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
                composer.match(
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

    def paired(
        identity: "SampleIdentity",
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
            a `Viewer` consumes. The scene keeps its own tooltips either way —
            live on its annotations, or reattached by `Image.with_hitmap` where
            a layout was baked to pixels — so ``--save-format html`` still
            resolves every hover region from the scene itself.
            """
            for identity in shared:
                gt_data, gt_records, pred_records = paired(identity)
                yield (
                    next(iter(gt_data.images), "image"),
                    composer.frame(
                        gt_data.images, gt_records, pred_records
                    ).image,
                )

        _write_renders(
            compared_frames(),
            save,
            image_format=save_format,
            fps=fps,
            background=viz_theme.background,
            theme=viz_theme,
            empty_note="No paired samples matched the comparison filters.",
        )
        return

    # The viewer — and the screen probe that opens a window to find the display
    # size — exists only on this interactive path; ``--save`` renders the same
    # frames headlessly above.
    viewer = Viewer(save_dir=_SAVE_ROOT / f"{name}_vs_{predictions}")
    print(_PICK_HINT)
    composer = replace(composer, screen=viewer.screen)
    for identity in shared:
        gt_data, gt_records, pred_records = paired(identity)
        viewer.show(
            name,
            composer.frame(gt_data.images, gt_records, pred_records),
            # The window is titled with the dataset; a saved frame is named
            # after the pair of images it actually shows.
            save_as="-".join(Path(file).stem for _, file in identity),
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
    palette: Annotated[
        _PaletteName,
        Parameter(alias="-pl"),
    ] = "default",
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
        palette: Which colors the class bars, chips, and per-class heatmaps are
            drawn in, picked independently of ``--theme``: ``default`` keeps the
            theme's own distinct-hue scheme, while ``okabe-ito``,
            ``tol-bright``, ``tol-high-contrast``, ``tol-vibrant``,
            ``tol-muted``, and ``cvd`` are colorblind-safe (see
            ``luxonis_ml data inspect --help``). Distinct from ``--gradient``,
            which colors the heatmaps' *values*.
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
        from luxonis_ml.vizlab.viewer import Cv2Backend, show_fitted
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
    if palette != "default":
        # A named scheme replaces the theme's; class colors are then assigned in
        # the order the plots first ask for them, as they are by default.
        plot_theme = plot_theme.with_palette(palette)
    # One backend for the whole series: it provides the best-effort screen
    # size, presents each task's window, and closes them all at the end.
    backend = Cv2Backend()
    screen = backend.screen_size() if not save_dir else None

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

        if save_dir:
            render_grid(scale).save(
                f"{save_dir}/dataset_health_{task_name}.png"
            )
            continue

        window = (
            f"dataset health: {task_name}" if task_name else "dataset health"
        )
        console.print(
            "[info]Press any key for the next task, or 'q' to quit.[/info]"
        )
        key = show_fitted(
            window, render_grid, scale=scale, screen=screen, backend=backend
        )
        if key == ord("q"):
            break

    if not save_dir:
        backend.close()


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
