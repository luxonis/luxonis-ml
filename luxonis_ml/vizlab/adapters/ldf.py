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
(class palette, skeletons, keypoint label mode) through `RenderOptions`.
"""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, TypeAlias

from luxonis_ml.vizlab.annotations import (
    Annotation,
    BBox,
    Classification,
    Corner,
    InfoCard,
    Keypoints,
    Mask,
    SemanticMask,
)
from luxonis_ml.vizlab.options import RenderOptions
from luxonis_ml.vizlab.tooltip import Tooltip

if TYPE_CHECKING:
    from luxonis_ml.ldf import (
        BBoxAnnotation,
        DatasetRecord,
        Detection,
        InstanceSegmentationAnnotation,
        KeypointAnnotation,
        SegmentationAnnotation,
    )
    from luxonis_ml.typing import ParamValue
    from luxonis_ml.vizlab.image import Image, Renderable
    from luxonis_ml.vizlab.io import ImageSource
    from luxonis_ml.vizlab.panel import PanelData

    #: Any LDF object `Image.add`/`to_render_annotations` renders: a whole
    #: record, a detection tree, or a single spatial annotation model.
    RenderableLDF: TypeAlias = (
        DatasetRecord
        | Detection
        | BBoxAnnotation
        | KeypointAnnotation
        | InstanceSegmentationAnnotation
        | SegmentationAnnotation
    )


def _spatial_annotations(
    detection: "Detection", options: RenderOptions, task_name: str
) -> list[Annotation]:
    """Build the spatial annotations for one detection (no record-level parts).

    The bounding box becomes the root; keypoints and the instance mask attach as
    its children so they share its derived color. ``sub_detections`` recurse as
    children. Semantic segmentation and pure classification are handled at the
    record level (see `visualize_record`), not here.
    """
    palette = options.theme.palette
    label = detection.class_name
    root: Annotation | None = None
    tops: list[Annotation] = []

    if detection.boundingbox is not None:
        root = BBox.from_ldf(
            detection.boundingbox, label=label, palette=palette
        )
        # The box's metadata rides along as hover content, if enabled.
        if options.hover_metadata:
            root.tooltip = _detection_tooltip(detection, options)

    # Keypoints and the instance mask are children of the box (deriving its
    # color), or top-level when there is no box; they carry no label then.
    child_label = None if root is not None else label
    if detection.keypoints is not None:
        _attach(
            root,
            tops,
            _keypoints_annotation(
                detection.keypoints, options, task_name, child_label
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
        for child in _spatial_annotations(sub, options, f"{task_name}/{name}"):
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
    options: RenderOptions,
    task_name: str,
    label: str | None,
) -> Keypoints:
    """Build a `Keypoints` annotation, resolving its skeleton from the options."""
    label_mode = options.keypoint_label_mode
    # A skeleton is needed to draw limbs and to resolve joint names.
    needs_skeleton = options.draw_skeletons or label_mode in ("names", "full")
    skeleton = options.skeletons.get(task_name) if needs_skeleton else None
    names, edges = skeleton if skeleton is not None else (None, [])
    return Keypoints.from_ldf(
        keypoints,
        edges=edges,
        keypoint_names=names,
        point_labels=label_mode,
        label=label,
        palette=options.theme.palette,
    )


def detection_to_annotations(
    detection: "Detection",
    options: RenderOptions | None = None,
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
        options: Rendering context; a default is used when ``None``.
        task_name: Task name this detection belongs to (used to look up its
            skeleton in ``options.skeletons``).

    Returns:
        The vizlab annotations to draw for this detection.

    """
    options = options or RenderOptions()
    annotations = _spatial_annotations(detection, options, task_name)
    if detection.segmentation is not None:
        annotations.append(
            Mask.from_ldf(
                detection.segmentation,
                label=detection.class_name,
                palette=options.theme.palette,
            )
        )
    if detection.class_name is not None and not annotations:
        annotations.append(
            Classification.from_ldf(
                [detection.class_name], palette=options.theme.palette
            )
        )
    return annotations


def blend_records_to_annotations(
    records: "Iterable[DatasetRecord]",
    options: RenderOptions | None = None,
) -> list[Annotation]:
    """Merge several records' detections into one flat annotation list.

    Use this when several tasks are drawn onto the *same* image (e.g. the
    ``--blend-all`` inspect view of a multitask dataset). Unlike converting each
    record independently, this suppresses the image-level classification chip
    once any spatial annotation (box, keypoints, mask) is present: a standalone
    class tag only reads correctly as the sole content of an image, so blending a
    classification task together with detection/segmentation tasks would leave a
    redundant corner chip. When nothing but class tags is present, the chips are
    kept.

    It likewise drops a segmentation mask's label chip when a box already labels
    that same class (see `_suppress_redundant_mask_labels`), so a dataset carrying
    both a detection and a semantic-segmentation task for the same classes is not
    labeled twice per class.

    Args:
        records: The records whose detections are drawn together; each record's
            task name is used to look up its detections' skeletons in ``options``.
        options: Rendering context; a default is used when ``None``.

    Returns:
        The vizlab annotations to draw, with redundant classification chips
        dropped when other annotations are present, and redundant mask chips
        dropped when a box already labels their class.

    """
    options = options or RenderOptions()
    annotations = [
        annotation
        for record in records
        for detection in record._annotations()
        for annotation in detection_to_annotations(
            detection, options, task_name=record.task_name
        )
    ]
    if any(not isinstance(a, Classification) for a in annotations):
        annotations = [
            a for a in annotations if not isinstance(a, Classification)
        ]
    _suppress_redundant_mask_labels(annotations)
    return annotations


def _suppress_redundant_mask_labels(annotations: list[Annotation]) -> None:
    """Drop a mask's label chip when a box already labels that same class.

    A semantic-segmentation mask paints one region per class; blended next to a
    detection task (the ``--blend-all`` view of a dataset with both a detection
    and a segmentation task), each class then carries two chips — the box's and
    the mask's identical restatement. Only the mask's redundant chip is hidden
    (via ``label_chip``); its label is kept, so the fill and contour still take
    the class color and the class focus still recognizes it. The class stays
    labeled on the box, and a segmentation class no box shows (e.g. ``road``)
    keeps its own chip. Mutates ``annotations`` in place.

    Args:
        annotations: The blended annotations to prune, edited in place.

    """
    boxed = {
        a.label for a in annotations if a.label and not isinstance(a, Mask)
    }
    for a in annotations:
        if isinstance(a, Mask) and a.label in boxed:
            a.label_chip = False


def _detection_tooltip(
    detection: "Detection", options: RenderOptions
) -> Tooltip | None:
    """Build a hover `Tooltip` from a boxed detection's metadata.

    The class name — with the instance id when present — becomes the title,
    tinted with the class color, and every metadata entry becomes a row. Returns
    ``None`` when there is no metadata worth hovering.
    """
    rows = tuple(
        (str(key), str(value))
        for key, value in (detection.metadata or {}).items()
    )
    if not rows:
        return None
    title = detection.class_name or "object"
    if detection.instance_id is not None:
        title = f"{title} #{detection.instance_id}"
    tint = (
        options.theme.palette.color_for(detection.class_name)
        if options.theme.palette is not None
        and detection.class_name is not None
        else None
    )
    return Tooltip(title=title, rows=rows, tint=tint)


def _append_meta(detection: "Detection", rows: list[str]) -> None:
    """Append one detection's own metadata to ``rows``."""
    if detection.metadata:
        rows.extend(_meta_rows(detection, detection.metadata))


def _collect_boxless(detection: "Detection", rows: list[str]) -> None:
    """Collect a box-less detection's metadata as card rows.

    A box-less detection has nothing to hover, so its metadata goes to ``rows``
    (a plain card). Recurses sub-detections.
    """
    if detection.boundingbox is None:
        _append_meta(detection, rows)
    for sub in detection.sub_detections.values():
        _collect_boxless(sub, rows)


def _meta_rows(
    detection: "Detection",
    metadata: "Mapping[str, int | float | str]",
) -> list[str]:
    """Format a detection's metadata as card rows."""
    rows: list[str] = []
    prefix = ""
    if detection.class_name:
        rows.append(detection.class_name)
        prefix = "  "
    for key, value in metadata.items():
        rows.append(f"{prefix}{key}: {value}")
    return rows


def metadata_annotations(
    detections: "Iterable[Detection]",
    *,
    lone_object_card: bool = False,
) -> list[Annotation]:
    """Build in-image cards for metadata that has nothing to hover.

    A detection that carries metadata but no bounding box has nothing to anchor a
    hover tooltip to, so its metadata is surfaced as a ``"metadata"`` corner card.
    Boxed detections normally contribute nothing here (their metadata is shown on
    hover).

    Args:
        detections: The detections to scan; their sub-detections are included.
        lone_object_card: When the image holds exactly one detection, also card
            that lone object's metadata even if it is boxed — a single object
            needs no hover. With more than one object, boxed metadata stays
            hover-only to keep dense scenes uncluttered.

    Returns:
        The overlay annotations to draw (empty when there is nothing to card).

    """
    detections = list(detections)
    rows: list[str] = []
    for detection in detections:
        _collect_boxless(detection, rows)
    if (
        lone_object_card
        and len(detections) == 1
        and detections[0].boundingbox is not None
    ):
        _append_meta(detections[0], rows)

    if rows:
        return [
            InfoCard(rows=rows, title="metadata", corner=Corner.BOTTOM_LEFT)
        ]
    return []


def to_render_annotations(
    obj: "RenderableLDF", options: RenderOptions | None = None
) -> list[Annotation]:
    """Convert one supported LDF object into render annotations.

    Accepts a `DatasetRecord`, a `Detection`, or a single annotation model
    (`BBoxAnnotation`, `KeypointAnnotation`, `InstanceSegmentationAnnotation`,
    `SegmentationAnnotation`). This is the conversion used by `Image.add`.
    For record-level aggregation of classification and semantic-segmentation
    labels, use `visualize_record`.

    Args:
        obj: The LDF object to render.
        options: Rendering context; a default is used when ``None``.

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

    options = options or RenderOptions()
    if isinstance(obj, DatasetRecord):
        annotations: list[Annotation] = []
        for detection in obj._annotations():
            annotations.extend(
                detection_to_annotations(
                    detection, options, task_name=obj.task_name
                )
            )
        annotations.extend(metadata_annotations(obj._annotations()))
        return annotations
    if isinstance(obj, Detection):
        return detection_to_annotations(obj, options)
    annotation: Annotation
    if isinstance(obj, BBoxAnnotation):
        annotation = BBox.from_ldf(obj, palette=options.theme.palette)
    elif isinstance(obj, KeypointAnnotation):
        annotation = Keypoints.from_ldf(obj, palette=options.theme.palette)
    elif isinstance(
        obj, (InstanceSegmentationAnnotation, SegmentationAnnotation)
    ):
        annotation = Mask.from_ldf(obj, palette=options.theme.palette)
    else:
        raise TypeError(
            f"Image.add() does not know how to render an LDF object of type "
            f"{type(obj).__name__!r}"
        )
    return [annotation]


def visualize_record(
    record: "DatasetRecord",
    image: "ImageSource",
    *,
    options: RenderOptions | None = None,
    panel: "Mapping[str, PanelData] | None" = None,
    size: tuple[int, int] | None = None,
) -> "Renderable":
    """Build one complete record visualization over its source image.

    Draws every detection's spatial annotations, aggregates all nested
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
        options: Render options (theme, palette, LDF behavior); a default is used
            when ``None``.
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
    from luxonis_ml.vizlab.image import Image

    options = options or RenderOptions()
    img = Image(image, options=options, render_size=size)
    task_name = record.task_name

    segmentations: list[tuple[str | None, SegmentationAnnotation]] = []
    class_tags: list[str] = []
    array_shapes: dict[str, list[int]] = {}

    for detection in record._annotations():
        _scan_detection(
            detection,
            options,
            task_name,
            img,
            segmentations,
            class_tags,
            array_shapes,
        )

    if segmentations:
        img.add(
            SemanticMask.from_ldf(segmentations, palette=options.theme.palette)
        )
    if class_tags:
        img.add(
            Classification.from_ldf(class_tags, palette=options.theme.palette)
        )
    for overlay in metadata_annotations(record._annotations()):
        img.add(overlay)

    panel_data = _panel_data(record, array_shapes, panel)
    if panel_data:
        img = img.with_panel(panel_data, title="Sample Metadata")
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
    options: RenderOptions,
    task_name: str,
    img: "Image",
    segmentations: "list[tuple[str | None, SegmentationAnnotation]]",
    class_tags: list[str],
    array_shapes: dict[str, list[int]],
) -> None:
    """Add a detection's spatial annotations and collect its record-level parts."""
    for annotation in _spatial_annotations(detection, options, task_name):
        img.add(annotation)
    _collect_record_annotations(
        detection,
        task_name,
        segmentations,
        class_tags,
        array_shapes,
    )


def _collect_record_annotations(
    detection: "Detection",
    task_name: str,
    segmentations: "list[tuple[str | None, SegmentationAnnotation]]",
    class_tags: list[str],
    array_shapes: dict[str, list[int]],
) -> None:
    """Collect record-level annotations from one complete detection tree."""
    if detection.segmentation is not None:
        segmentations.append((detection.class_name, detection.segmentation))
    if _is_pure_classification(detection) and detection.class_name is not None:
        class_tags.append(detection.class_name)
    if detection.array is not None:
        array_shapes[task_name or "array"] = list(
            detection.array.to_numpy().shape
        )
    for name, sub_detection in detection.sub_detections.items():
        _collect_record_annotations(
            sub_detection,
            f"{task_name}/{name}",
            segmentations,
            class_tags,
            array_shapes,
        )


def _panel_data(
    record: "DatasetRecord",
    array_shapes: dict[str, list[int]],
    panel: "Mapping[str, PanelData] | None",
) -> "dict[str, PanelData]":
    """Merge sample metadata, array shapes, and an extra panel into panel data."""
    data: dict[str, PanelData] = {
        key: _metadata_to_panel_data(value)
        for key, value in record.sample_metadata.items()
    }
    if array_shapes:
        data["arrays"] = array_shapes
    if panel:
        data.update(panel)
    return data


def _metadata_to_panel_data(value: "ParamValue") -> "PanelData":
    """Normalize JSON-like metadata to the panel's string-keyed data model."""
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_to_panel_data(item)
            for key, item in value.items()
        }
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return [_metadata_to_panel_data(item) for item in value]
    return value
