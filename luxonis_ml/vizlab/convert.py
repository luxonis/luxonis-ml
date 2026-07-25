"""Native rendering of Luxonis Data Format (LDF) objects.

vizlab renders LDF objects directly: pass a
`Detection` (or a whole
`DatasetRecord`) to
`Image.add`, or use `visualize_record` to
render a record onto an image in one call.

The per-annotation mapping lives with each render class as a ``from_ldf``
constructor (see `BBox.from_ldf` and
friends). This module wires those constructors together: it walks a
``Detection`` tree into a vizlab annotation tree, aggregates record-level
semantic segmentation and classification, and threads the rendering context
(class palette, skeletons, keypoint label mode) through a `VizConfig`.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .annotations import (
    Annotation,
    BBox,
    Classification,
    Corner,
    InfoCard,
    Keypoints,
    Mask,
    SemanticMask,
)
from .style import DEFAULT_STYLE, Palette, Theme

#: Prominent style for recognized-text captions (larger, bold).
_TEXT_STYLE = DEFAULT_STYLE.merge(
    font_size=DEFAULT_STYLE.font_size * 1.6, font_weight=700
)

if TYPE_CHECKING:
    from luxonis_ml.ldf import (
        DatasetRecord,
        Detection,
        KeypointAnnotation,
        SegmentationAnnotation,
    )

    from .image import Image

KeypointLabelMode = Literal["none", "numbers", "names", "full"]

SkeletonDef = tuple[list[str], list[tuple[int, int]]]
"""A keypoint skeleton as ``(labels, edges)`` — ``get_skeletons()``'s shape."""


@dataclass
class VizConfig:
    """Rendering context for turning LDF objects into vizlab annotations.

    LDF annotations carry only data; everything needed to *draw* them that is
    not on the annotation itself is supplied here. The same config renders a
    live ``LuxonisDataset`` sample (from the ``inspect`` CLI) and hand-built
    `Detection` objects. Reuse one config across a dataset so palette assignments
    and skeleton lookup remain consistent.

    Attributes:
        palette: Palette used to color classes. Pre-seed it with the dataset's
            class names in a fixed order (``Palette(class_names)``) for
            deterministic, stable colors across images.
        skeletons: Keypoint skeletons keyed by task name, in LDF's own
            ``(labels, edges)`` shape — pass ``LuxonisDataset.get_skeletons()``
            directly.
        keypoint_label_mode: How to label keypoints
            (``"none"``/``"numbers"``/``"names"``/``"full"``).
        draw_skeletons: Whether to draw skeleton limbs between keypoints.
        theme: Theme supplying default style/palette; ``None`` uses the default.
        text_metadata_key: Metadata key holding a recognized text string (e.g.
            OCR). It is rendered prominently — on a boxed detection's label chip,
            or as a large caption for a box-less one — instead of as a small
            metadata row. ``None`` disables the special treatment.

    Examples:
        >>> from luxonis_ml.vizlab import Palette, VizConfig
        >>> config = VizConfig(
        ...     palette=Palette(["person"]),
        ...     skeletons={"pose": (["left", "right"], [(0, 1)])},
        ...     keypoint_label_mode="names",
        ...     draw_skeletons=True,
        ... )
        >>> config.skeletons["pose"]
        (['left', 'right'], [(0, 1)])

    """

    palette: Palette | None = None
    skeletons: dict[str, SkeletonDef] = field(default_factory=dict)
    keypoint_label_mode: KeypointLabelMode = "numbers"
    draw_skeletons: bool = False
    theme: Theme | None = None
    text_metadata_key: str | None = "text"


def _spatial_annotations(
    detection: "Detection", config: VizConfig, task_name: str
) -> list[Annotation]:
    """Build the spatial annotations for one detection (no record-level parts).

    The bounding box becomes the root; keypoints and the instance mask attach as
    its children so they share its derived color. ``sub_detections`` recurse as
    children. Semantic segmentation and pure classification are handled at the
    record level (see `visualize_record`), not here.
    """
    palette = config.palette
    label = detection.class_name
    root: Annotation | None = None
    tops: list[Annotation] = []

    if detection.boundingbox is not None:
        root = BBox.from_ldf(
            detection.boundingbox, label=label, palette=palette
        )
        # Show recognized text (e.g. OCR) prominently on the box's label chip.
        text = _text_value(detection, config.text_metadata_key)
        if text is not None:
            root.payload = text

    # Keypoints and the instance mask are children of the box (deriving its
    # color), or top-level when there is no box; they carry no label then.
    child_label = None if root is not None else label
    if detection.keypoints is not None:
        _attach(
            root,
            tops,
            _keypoints_annotation(
                detection.keypoints, config, task_name, child_label
            ),
        )
    if detection.instance_segmentation is not None:
        _attach(
            root,
            tops,
            Mask.from_ldf(
                detection.instance_segmentation,
                label=child_label,
                palette=palette,
            ),
        )
    for name, sub in detection.sub_detections.items():
        for child in _spatial_annotations(sub, config, f"{task_name}/{name}"):
            _attach(root, tops, child)

    return ([root] if root is not None else []) + tops


def _attach(
    root: "Annotation | None", tops: list[Annotation], annotation: Annotation
) -> None:
    """Add ``annotation`` as a child of ``root``, or to the top-level list."""
    if root is not None:
        root.add(annotation)
    else:
        tops.append(annotation)


def _keypoints_annotation(
    keypoints: "KeypointAnnotation",
    config: VizConfig,
    task_name: str,
    label: str | None,
) -> Keypoints:
    """Build a `Keypoints` annotation, resolving its skeleton from the config."""
    label_mode = config.keypoint_label_mode
    # A skeleton is needed to draw limbs and to resolve joint names.
    needs_skeleton = config.draw_skeletons or label_mode in ("names", "full")
    skeleton = config.skeletons.get(task_name) if needs_skeleton else None
    names, edges = skeleton if skeleton is not None else (None, [])
    return Keypoints.from_ldf(
        keypoints,
        edges=edges,
        keypoint_names=names,
        point_labels=label_mode,
        label=label,
        palette=config.palette,
    )


def detection_to_annotations(
    detection: "Detection",
    config: VizConfig | None = None,
    *,
    task_name: str = "",
) -> list[Annotation]:
    """Convert a single LDF `Detection` into vizlab annotations.

    Includes the spatial annotations (box, keypoints, instance mask, nested
    sub-detections) plus, for a standalone detection, its semantic mask (as a
    single-class `Mask`) and an
    image-level classification chip when the detection carries only a class name.

    Args:
        detection: The LDF detection to render.
        config: Rendering context; a default is used when ``None``.
        task_name: Task name this detection belongs to (used to look up its
            skeleton in ``config.skeletons``).

    Returns:
        The vizlab annotations to draw for this detection.

    """
    config = config or VizConfig()
    annotations = _spatial_annotations(detection, config, task_name)
    if detection.segmentation is not None:
        annotations.append(
            Mask.from_ldf(
                detection.segmentation,
                label=detection.class_name,
                palette=config.palette,
            )
        )
    if detection.class_name is not None and not annotations:
        annotations.append(
            Classification.from_ldf(
                [detection.class_name], palette=config.palette
            )
        )
    return annotations


def _text_value(detection: "Detection", text_key: str | None) -> str | None:
    """Return the detection's recognized-text metadata value, if any."""
    if text_key is None or not detection.metadata:
        return None
    value = detection.metadata.get(text_key)
    return None if value is None else str(value)


def _collect_boxless(
    detection: "Detection",
    text_key: str | None,
    texts: list[str],
    rows: list[str],
) -> None:
    """Split a box-less detection's metadata into text and other rows.

    Recognized text (``text_key``) goes to ``texts`` (rendered prominently); all
    other metadata goes to ``rows`` (a plain card). Recurses sub-detections.
    """
    if detection.boundingbox is None and detection.metadata:
        text = _text_value(detection, text_key)
        if text is not None:
            texts.append(text)
        other = {
            key: value
            for key, value in detection.metadata.items()
            if text_key is None or key != text_key
        }
        if other:
            rows.extend(_boxless_rows(detection, other))
    for sub in detection.sub_detections.values():
        _collect_boxless(sub, text_key, texts, rows)


def _boxless_rows(detection: "Detection", other: dict) -> list[str]:
    """Format a box-less detection's non-text metadata as card rows."""
    rows: list[str] = []
    prefix = ""
    if detection.class_name:
        rows.append(detection.class_name)
        prefix = "  "
    for key, value in other.items():
        rows.append(f"{prefix}{key}: {value}")
    return rows


def metadata_annotations(
    detections: "Iterable[Detection]", *, text_key: str | None = "text"
) -> list[Annotation]:
    """Build in-image cards for box-less detections' metadata.

    A detection that carries metadata but no bounding box has nothing to anchor a
    hover tooltip to, so its metadata is surfaced as a corner card. Recognized
    text (``text_key``) is rendered prominently in its own larger card; the
    remaining key/value metadata goes in a smaller ``"metadata"`` card. Boxed
    detections contribute nothing here (their metadata is shown on hover, and
    their text on the label chip).

    Args:
        detections: The detections to scan; their sub-detections are included.
        text_key: Metadata key holding recognized text, or ``None`` to treat all
            metadata uniformly.

    Returns:
        The overlay annotations to draw (empty when there is no box-less
        metadata).

    """
    texts: list[str] = []
    rows: list[str] = []
    for detection in detections:
        _collect_boxless(detection, text_key, texts, rows)

    annotations: list[Annotation] = []
    if texts:
        annotations.append(
            InfoCard(rows=texts, corner=Corner.TOP_LEFT, style=_TEXT_STYLE)
        )
    if rows:
        annotations.append(
            InfoCard(rows=rows, title="metadata", corner=Corner.BOTTOM_LEFT)
        )
    return annotations


def to_render_annotations(
    obj: object, config: VizConfig | None = None
) -> list[Annotation]:
    """Convert one supported LDF object into render annotations.

    Accepts a `DatasetRecord`, a `Detection`, or a single annotation model
    (`BBoxAnnotation`, `KeypointAnnotation`, `InstanceSegmentationAnnotation`,
    `SegmentationAnnotation`). This is the conversion used by `Image.add`.
    For record-level aggregation of classification and semantic-segmentation
    labels, use `visualize_record`.

    Args:
        obj: The LDF object to render.
        config: Rendering context; a default is used when ``None``.

    Returns:
        The vizlab annotations to draw.

    Raises:
        TypeError: If ``obj`` is not a renderable LDF type.

    """
    from luxonis_ml.ldf import (
        BBoxAnnotation,
        DatasetRecord,
        Detection,
        InstanceSegmentationAnnotation,
        KeypointAnnotation,
        SegmentationAnnotation,
    )

    config = config or VizConfig()
    if isinstance(obj, DatasetRecord):
        annotations: list[Annotation] = []
        for detection in obj._annotations():
            annotations.extend(
                detection_to_annotations(
                    detection, config, task_name=obj.task_name
                )
            )
        annotations.extend(
            metadata_annotations(
                obj._annotations(), text_key=config.text_metadata_key
            )
        )
        return annotations
    if isinstance(obj, Detection):
        return detection_to_annotations(obj, config)
    annotation: Annotation
    if isinstance(obj, BBoxAnnotation):
        annotation = BBox.from_ldf(obj, palette=config.palette)
    elif isinstance(obj, KeypointAnnotation):
        annotation = Keypoints.from_ldf(obj, palette=config.palette)
    elif isinstance(
        obj, (InstanceSegmentationAnnotation, SegmentationAnnotation)
    ):
        annotation = Mask.from_ldf(obj, palette=config.palette)
    else:
        raise TypeError(
            f"Image.add() does not know how to render an LDF object of type "
            f"{type(obj).__name__!r}"
        )
    return [annotation]


def visualize_record(
    record: "DatasetRecord",
    image: object,
    *,
    config: VizConfig | None = None,
    theme: Theme | None = None,
    panel: dict | None = None,
    size: tuple[int, int] | None = None,
) -> "Image":
    """Build one complete record visualization over its source image.

    Draws every detection's spatial annotations, aggregates all top-level
    semantic-segmentation masks into a single
    `SemanticMask`, collects
    class-only detections into one classification overlay, and attaches a
    metadata side-panel built from the record's ``sample_metadata``, any array
    shapes, and an optional extra ``panel`` mapping. Extra panel values override
    record metadata with the same key.

    Args:
        record: The LDF record to visualize (its ``annotation`` may be a single
            `Detection` or a list of them).
        image: The base image (any source accepted by `Image`).
        config: Rendering context; a default is used when ``None``.
        theme: Theme override; falls back to ``config.theme``.
        panel: Extra key/value entries to show in the metadata side-panel.
        size: Optional ``(width, height)`` display size for the image; the panel
            (when present) is drawn crisply beside the scaled image. ``None``
            keeps the source resolution. See `Image.render`.

    Returns:
        A new visualization `Image`. When panel content exists, the returned
        image includes the rendered source and its side panel.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.ldf import DatasetRecord, Detection
        >>> from luxonis_ml.vizlab import visualize_record
        >>> record = DatasetRecord.model_construct(
        ...     files={},
        ...     annotation=[
        ...         Detection(
        ...             class_name="car",
        ...             boundingbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        ...         )
        ...     ],
        ...     task_name="objects",
        ... )
        >>> visualize_record(
        ...     record, np.zeros((32, 48, 3), np.uint8)
        ... ).render().shape
        (32, 48, 4)

    """
    from .image import Image

    config = config or VizConfig()
    img = Image(
        image, theme=theme or config.theme, config=config, render_size=size
    )
    task_name = record.task_name

    segmentations: list[tuple[str | None, SegmentationAnnotation]] = []
    class_tags: list[str] = []
    array_shapes: dict[str, list[int]] = {}

    for detection in record._annotations():
        _scan_detection(
            detection,
            config,
            task_name,
            img,
            segmentations,
            class_tags,
            array_shapes,
        )

    if segmentations:
        img.add(SemanticMask.from_ldf(segmentations, palette=config.palette))
    if class_tags:
        img.add(Classification.from_ldf(class_tags, palette=config.palette))
    for overlay in metadata_annotations(
        record._annotations(), text_key=config.text_metadata_key
    ):
        img.add(overlay)

    panel_data = _panel_data(record, array_shapes, panel)
    if panel_data:
        img = img.with_panel(panel_data, title="metadata")
    return img


def _is_pure_classification(detection: "Detection") -> bool:
    """Report whether a detection carries only a class name (an image-level tag)."""
    return (
        detection.class_name is not None
        and detection.boundingbox is None
        and detection.keypoints is None
        and detection.instance_segmentation is None
        and detection.segmentation is None
        and not detection.sub_detections
    )


def _scan_detection(
    detection: "Detection",
    config: VizConfig,
    task_name: str,
    img: "Image",
    segmentations: "list[tuple[str | None, SegmentationAnnotation]]",
    class_tags: list[str],
    array_shapes: dict[str, list[int]],
) -> None:
    """Add a detection's spatial annotations and collect its record-level parts."""
    for annotation in _spatial_annotations(detection, config, task_name):
        img.add(annotation)
    if detection.segmentation is not None:
        segmentations.append((detection.class_name, detection.segmentation))
    if _is_pure_classification(detection) and detection.class_name is not None:
        class_tags.append(detection.class_name)
    if detection.array is not None:
        array_shapes[task_name or "array"] = list(
            detection.array.to_numpy().shape
        )


def _panel_data(
    record: "DatasetRecord",
    array_shapes: dict[str, list[int]],
    panel: dict | None,
) -> dict:
    """Merge sample metadata, array shapes, and an extra panel into panel data."""
    data: dict = dict(record.sample_metadata) if record.sample_metadata else {}
    if array_shapes:
        data["arrays"] = array_shapes
    if panel:
        data.update(panel)
    return data
