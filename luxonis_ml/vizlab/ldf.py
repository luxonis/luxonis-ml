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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .annotations import (
    Annotation,
    BBox,
    Classification,
    Keypoints,
    Mask,
    SemanticMask,
    Skeleton,
)
from .style import Palette, Theme

if TYPE_CHECKING:
    from luxonis_ml.ldf import (
        DatasetRecord,
        Detection,
        SegmentationAnnotation,
    )

    from .image import Image

KeypointLabelMode = Literal["none", "numbers", "names", "full"]


@dataclass
class VizConfig:
    """Rendering context for turning LDF objects into vizlab annotations.

    LDF annotations carry only data; everything needed to *draw* them that is
    not on the annotation itself is supplied here. The same config renders a
    live ``LuxonisDataset`` sample (from the ``inspect`` CLI) and hand-built
    ``Detection`` objects (tests, library use).

    Attributes:
        palette: Palette used to color classes. Pre-seed it with the dataset's
            class names in a fixed order (``Palette(class_names)``) for
            deterministic, stable colors across images.
        skeletons: Keypoint skeletons keyed by task name, in LDF's own
            ``(labels, edges)`` shape as returned by
            ``LuxonisDataset.get_skeletons`` — build them with
            `Skeleton.from_ldf`.
        keypoint_label_mode: How to label keypoints
            (``"none"``/``"numbers"``/``"names"``/``"full"``).
        draw_skeletons: Whether to draw skeleton limbs between keypoints.
        theme: Theme supplying default style/palette; ``None`` uses the default.

    """

    palette: Palette | None = None
    skeletons: dict[str, Skeleton] = field(default_factory=dict)
    keypoint_label_mode: KeypointLabelMode = "numbers"
    draw_skeletons: bool = False
    theme: Theme | None = None


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

    if detection.keypoints is not None:
        want_names = config.keypoint_label_mode in ("names", "full")
        skeleton = (
            config.skeletons.get(task_name)
            if config.draw_skeletons or want_names
            else None
        )
        keypoints = Keypoints.from_ldf(
            detection.keypoints,
            skeleton=skeleton,
            show_names=want_names,
            label=None if root is not None else label,
            palette=palette,
        )
        (root.add(keypoints) if root is not None else tops.append(keypoints))

    if detection.instance_segmentation is not None:
        mask = Mask.from_ldf(
            detection.instance_segmentation,
            label=None if root is not None else label,
            palette=palette,
        )
        (root.add(mask) if root is not None else tops.append(mask))

    for name, sub in detection.sub_detections.items():
        for child in _spatial_annotations(sub, config, f"{task_name}/{name}"):
            (root.add(child) if root is not None else tops.append(child))

    return ([root] if root is not None else []) + tops


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


def to_render_annotations(
    obj: object, config: VizConfig | None = None
) -> list[Annotation]:
    """Convert an LDF object into vizlab annotations (dispatch by type).

    Accepts a `DatasetRecord`, a `Detection`, or a single annotation model
    (`BBoxAnnotation`, `KeypointAnnotation`, `InstanceSegmentationAnnotation`,
    `SegmentationAnnotation`).

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
) -> "Image":
    """Render a `DatasetRecord` (and its image) into a composed `Image`.

    Draws every detection's spatial annotations, aggregates all top-level
    semantic-segmentation masks into a single
    `SemanticMask`, collects
    class-only detections into one classification overlay, and attaches a
    metadata side-panel built from the record's ``sample_metadata``, any array
    shapes, and an optional extra ``panel`` mapping.

    Args:
        record: The LDF record to visualize (its ``annotation`` may be a single
            `Detection` or a list of them).
        image: The base image (any source accepted by `Image`).
        config: Rendering context; a default is used when ``None``.
        theme: Theme override; falls back to ``config.theme``.
        panel: Extra key/value entries to show in the metadata side-panel.

    Returns:
        The rendered `Image`; when there is panel content, the returned
        image includes the side-panel.

    """
    from .image import Image

    config = config or VizConfig()
    img = Image(image, theme=theme or config.theme, config=config)
    task_name = record.task_name

    segmentations: list[tuple[str | None, SegmentationAnnotation]] = []
    class_tags: list[str] = []
    array_shapes: dict[str, list[int]] = {}

    for detection in record._annotations():
        for annotation in _spatial_annotations(detection, config, task_name):
            img.add(annotation)
        if detection.segmentation is not None:
            segmentations.append(
                (detection.class_name, detection.segmentation)
            )
        if (
            detection.class_name is not None
            and detection.boundingbox is None
            and detection.keypoints is None
            and detection.instance_segmentation is None
            and detection.segmentation is None
            and not detection.sub_detections
        ):
            class_tags.append(detection.class_name)
        if detection.array is not None:
            array_shapes[task_name or "array"] = list(
                detection.array.to_numpy().shape
            )

    if segmentations:
        img.add(SemanticMask.from_ldf(segmentations, palette=config.palette))
    if class_tags:
        img.add(Classification.from_ldf(class_tags, palette=config.palette))

    panel_data: dict = (
        dict(record.sample_metadata) if record.sample_metadata else {}
    )
    if array_shapes:
        panel_data["arrays"] = array_shapes
    if panel:
        panel_data.update(panel)
    if panel_data:
        img = img.with_panel(panel_data, title="metadata")
    return img
