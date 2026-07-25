import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

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

if TYPE_CHECKING:
    from luxonis_ml.ldf import DatasetRecord, Detection
    from luxonis_ml.vizlab import Palette

app = App(help="Dataset utilities.")

MouseCallback: TypeAlias = Callable[[int, int, int, int, object], None]


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


# --- Metadata hover tooltips -------------------------------------------------

HoverItem: TypeAlias = tuple[tuple[float, float, float, float], dict]


@dataclass
class _HoverWindow:
    """Per-window state for drawing metadata tooltips on mouse hover.

    Attributes:
        base: The rendered BGR frame without any tooltip.
        items: ``(rect, info)`` pairs; ``rect`` is a box in frame pixels and
            ``info`` is the detection's metadata (plus a ``"_title"`` header).
        frame: Composed frame size ``(w, h)`` (the image passed to ``imshow``).
        win: On-screen window size ``(w, h)``; used to map mouse coordinates
            back into frame pixels when the window was scaled.
        hover_idx: Index of the item under the cursor, or ``None``.
        mouse: Cursor position in frame pixels.
        dirty: Whether the window needs a redraw.

    """

    base: "np.ndarray"
    items: list[HoverItem]
    frame: tuple[int, int]
    win: tuple[int, int]
    hover_idx: int | None = None
    mouse: tuple[int, int] = (0, 0)
    dirty: bool = False


def _collect_hover_items(
    detection: "Detection",
    width: int,
    height: int,
    items: list[HoverItem],
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    text_key: str | None = None,
    palette: "Palette | None" = None,
) -> None:
    """Append ``(box_rect, info)`` for a detection (and its sub-detections).

    Only detections that carry both a bounding box and metadata contribute an
    item; the box is resolved to pixel coordinates on a ``width`` x ``height``
    image and shifted by ``(offset_x, offset_y)`` (the tile origin in a grid).
    ``text_key`` metadata is omitted here since it is shown on the label chip.
    When a ``palette`` is given, the tooltip title is tinted with the detection's
    class color (matching how its box is drawn), stored under ``"_color"`` as a
    BGR tuple.
    """
    box = detection.boundingbox
    info = {
        key: value
        for key, value in (detection.metadata or {}).items()
        if key != text_key
    }
    if box is not None and info:
        rect = (
            offset_x + box.x * width,
            offset_y + box.y * height,
            offset_x + (box.x + box.w) * width,
            offset_y + (box.y + box.h) * height,
        )
        title = detection.class_name or "object"
        if detection.instance_id is not None:
            title = f"{title} #{detection.instance_id}"
        info = {"_title": title, **info}
        if palette is not None and detection.class_name is not None:
            color = palette.color_for(detection.class_name)
            info["_color"] = (color.b, color.g, color.r)
        items.append((rect, info))
    for sub in detection.sub_detections.values():
        _collect_hover_items(
            sub, width, height, items, offset_x, offset_y, text_key, palette
        )


def _hover_items(
    records: "dict[str, DatasetRecord]",
    width: int,
    height: int,
    text_key: str | None = None,
    palette: "Palette | None" = None,
) -> list[HoverItem]:
    """Build the hover hit-list for all detections in ``records`` (no offset)."""
    items: list[HoverItem] = []
    for record in records.values():
        for detection in record._annotations():
            _collect_hover_items(
                detection,
                width,
                height,
                items,
                text_key=text_key,
                palette=palette,
            )
    return items


def _hit_test(items: list[HoverItem], x: float, y: float) -> int | None:
    """Return the index of the smallest box containing ``(x, y)``, or ``None``."""
    best: int | None = None
    best_area: float | None = None
    for i, (rect, _) in enumerate(items):
        left, top, right, bottom = rect
        if left <= x <= right and top <= y <= bottom:
            area = (right - left) * (bottom - top)
            if best_area is None or area < best_area:
                best, best_area = i, area
    return best


def _draw_tooltip(
    frame: "np.ndarray", info: dict, at: tuple[int, int]
) -> None:
    """Draw a translucent metadata card near ``at`` on the BGR ``frame``."""
    title = info.get("_title")
    title_color = info.get("_color", (120, 200, 255))
    rows = ([title] if title else []) + [
        f"{key}: {value}"
        for key, value in info.items()
        if not key.startswith("_")
    ]
    if not rows:
        return
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    pad, gap = 8, 6
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in rows]
    text_h = max(h for _, h in sizes)
    line_h = text_h + gap
    box_w = max(w for w, _ in sizes) + 2 * pad
    box_h = 2 * pad + line_h * len(rows) - gap
    height, width = frame.shape[:2]
    if box_w >= width or box_h >= height:
        return
    x = max(0, min(int(at[0]) + 16, width - box_w - 1))
    y = max(0, min(int(at[1]) + 16, height - box_h - 1))

    roi = frame[y : y + box_h, x : x + box_w]
    background = roi.copy()
    background[:] = (32, 32, 32)
    cv2.addWeighted(background, 0.72, roi, 0.28, 0, roi)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (90, 90, 90), 1)

    baseline = y + pad + text_h
    for i, text in enumerate(rows):
        color = title_color if title and i == 0 else (238, 238, 238)
        cv2.putText(
            frame,
            text,
            (x + pad, baseline),
            font,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )
        baseline += line_h


def _make_hover_callback(window: _HoverWindow) -> MouseCallback:
    """Build a mouse callback that tracks the hovered box for ``window``."""

    def callback(
        event: int, x: int, y: int, flags: int, param: object
    ) -> None:
        if event != cv2.EVENT_MOUSEMOVE or not window.items:
            return
        frame_w, frame_h = window.frame
        win_w, win_h = window.win
        fx = x * frame_w / win_w if win_w else x
        fy = y * frame_h / win_h if win_h else y
        idx = _hit_test(window.items, fx, fy)
        pos = (int(fx), int(fy))
        if idx != window.hover_idx or (
            idx is not None and pos != window.mouse
        ):
            window.hover_idx = idx
            window.mouse = pos
            window.dirty = True

    return callback


def _wait_with_hover(windows: dict[str, _HoverWindow]) -> int:
    """Poll for a key while redrawing hover tooltips; return the key pressed."""
    while True:
        key = cv2.waitKey(20)
        if key != -1:
            return key & 0xFF
        for name, window in windows.items():
            if not window.dirty:
                continue
            window.dirty = False
            if window.hover_idx is None:
                cv2.imshow(name, window.base)
            else:
                frame = window.base.copy()
                _draw_tooltip(
                    frame, window.items[window.hover_idx][1], window.mouse
                )
                cv2.imshow(name, frame)


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
    legend: Annotated[
        bool,
        Parameter(alias="-lg", negative=""),
    ] = False,
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
        legend: Draw a class-color legend on each image.
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
            Image,
            Legend,
            Palette,
            Skeleton,
            VizConfig,
            visualize_record,
        )
        from luxonis_ml.vizlab.compose import grid_placed
        from luxonis_ml.vizlab.convert import (
            detection_to_annotations,
            metadata_annotations,
        )
    except ImportError as e:
        raise SystemExit(
            "Visualization requires the 'viz' extra. "
            "Install it with `pip install luxonis-ml[viz]`."
        ) from e

    # Pre-seed a palette with every class name (in id order per task) so class
    # colors stay stable across samples. Names are stripped to match the labels
    # the converter produces (see ``loader_output_to_records``).
    class_names: list[str] = []
    for task_classes in classes.values():
        for class_name in sorted(task_classes, key=task_classes.__getitem__):
            stripped = class_name.strip()
            if stripped not in class_names:
                class_names.append(stripped)
    config = VizConfig(
        palette=Palette(class_names),
        skeletons={
            task: Skeleton.from_ldf(labels, edges)
            for task, (labels, edges) in keypoint_skeletons.items()
        },
        keypoint_label_mode=keypoint_labels,
        draw_skeletons=skeletons,
    )
    # A class-color key, colored from the shared palette so it matches the
    # boxes. "background" is bookkeeping, not a real class, so it is left out.
    has_legend = legend and any(name != _BACKGROUND for name in class_names)
    class_legend = (
        Legend(
            entries=[n for n in class_names if n != _BACKGROUND],
            palette=config.palette,
            title="classes",
        )
        if has_legend
        else None
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

    def show(
        source_name: str, viz: Image
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
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
        return out, (out_w, out_h), (win_w, win_h)

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
        hover_windows: dict[str, _HoverWindow] = {}
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
                    if class_legend is not None:
                        viz.add(class_legend)
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
            placements: list[tuple[int, int, int, int]] | None = None
            if blend_all or len(records) <= 1:
                size = target_size(width, height, reserve=reserve)
                viz = Image(image, config=config).render_at(size)
                for record in records.values():
                    for detection in record._annotations():
                        for annotation in detection_to_annotations(
                            detection, config, task_name=record.task_name
                        ):
                            viz.add(annotation)
                # Box-less metadata has nothing to hover, so show it as a card
                # (recognized text is rendered prominently on its own).
                for overlay in metadata_annotations(
                    [d for r in records.values() for d in r._annotations()],
                    text_key=config.text_metadata_key,
                ):
                    viz.add(overlay)
            else:
                # Fit the whole grid of tiles within the screen.
                cols = max(1, math.ceil(math.sqrt(len(records))))
                rows = math.ceil(len(records) / cols)
                size = target_size(
                    width, height, reserve=reserve, cols=cols, rows=rows
                )
                # visualize_record adds each tile's own box-less metadata card.
                tiles = [
                    visualize_record(record, image, config=config, size=size)
                    for record in records.values()
                ]
                viz, placements = grid_placed(
                    tiles, ncols=cols, titles=list(records)
                )
            if class_legend is not None:
                viz.add(class_legend)
            if panel:
                viz = viz.with_panel(panel, title="metadata")
            out, frame_size, win_size = show(source_name, viz)

            # Hover tooltips: hit-test the drawn detection boxes. In the grid
            # view each tile has its own offset in the composite (from
            # grid_placed); in the single/blend view the image sits at the
            # frame's top-left.
            disp_w, disp_h = size or (width, height)
            text_key = config.text_metadata_key
            items: list[HoverItem] = []
            if placements is not None:
                for (px, py, _, _), record in zip(
                    placements, records.values(), strict=True
                ):
                    for detection in record._annotations():
                        _collect_hover_items(
                            detection,
                            disp_w,
                            disp_h,
                            items,
                            px,
                            py,
                            text_key,
                            config.palette,
                        )
            else:
                items = _hover_items(
                    records, disp_w, disp_h, text_key, config.palette
                )
            window = _HoverWindow(
                base=out, items=items, frame=frame_size, win=win_size
            )
            hover_windows[source_name] = window
            cv2.setMouseCallback(source_name, _make_hover_callback(window))

        prev_windows = current_windows

        if quit_requested:
            break
        key = (
            _wait_with_hover(hover_windows) if hover_windows else cv2.waitKey()
        )
        if key == ord("q"):
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
