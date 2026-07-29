"""Render LDF spatial detections with colors keyed by instance identity.

The standard LDF adapter colors annotations by class. This module provides the
alternate presentation used by ``data inspect --per-instance``: all spatial
detections remain in one scene, but every instance receives its own stable color
and a hover tooltip describing the source detection.
"""

from collections.abc import Iterable, Sequence
from typing import TypeAlias

from luxonis_ml.ldf import DatasetRecord, Detection
from luxonis_ml.vizlab.annotations import Annotation
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.options import RenderOptions
from luxonis_ml.vizlab.style import Palette
from luxonis_ml.vizlab.tooltip import Tooltip

from .ldf import detection_to_annotations

InstanceDetection: TypeAlias = tuple[str, Detection]
"""A task name paired with one spatial LDF detection."""


def spatial_instances(
    records: Iterable[DatasetRecord],
) -> list[InstanceDetection]:
    """Collect detections that have instance-level spatial annotations.

    Args:
        records: LDF records to inspect.

    Returns:
        ``(task_name, detection)`` pairs for detections carrying a bounding box,
        keypoints, or an instance-segmentation mask.

    """
    return [
        (record.task_name, detection)
        for record in records
        for detection in record._annotations()
        if detection.boundingbox is not None
        or detection.keypoints is not None
        or detection.instance_segmentation is not None
    ]


def instances_to_annotations(
    instances: Sequence[InstanceDetection],
    *,
    options: RenderOptions,
    palette: Palette,
) -> list[Annotation]:
    """Convert spatial detections using stable per-instance colors and tooltips.

    A non-negative LDF ``instance_id`` is stable within its task and therefore
    forms the color key. Directly constructed detections can retain the schema's
    default ``-1`` ID; those use their sequence position so separate detections
    still receive separate colors.

    Args:
        instances: Task/detection pairs to convert.
        options: Render options passed to the standard LDF adapter.
        palette: Palette dedicated to instance identities. Reuse it across
            samples when the same task/instance ID should keep its color.

    Returns:
        Vizlab annotations with an explicit instance color and tooltip on every
        part of each converted annotation tree.

    """
    annotations: list[Annotation] = []
    for ordinal, (task_name, detection) in enumerate(instances):
        identity = _instance_identity(task_name, detection, ordinal)
        color = palette.color_for(identity)
        tooltip = _instance_tooltip(task_name, detection, color)
        converted = detection_to_annotations(
            detection,
            options,
            task_name=task_name,
        )
        for annotation in converted:
            _style_instance_tree(annotation, color, tooltip)
        annotations.extend(converted)
    return annotations


def _instance_identity(
    task_name: str,
    detection: Detection,
    ordinal: int,
) -> str:
    """Return the palette key for one detection."""
    if detection.instance_id >= 0:
        return f"{task_name}:{detection.instance_id}"
    return f"{task_name}:ordinal:{ordinal}"


def _instance_tooltip(
    task_name: str,
    detection: Detection,
    color: Color,
) -> Tooltip:
    """Build the per-instance hover card for one spatial detection."""
    annotation_types: list[str] = []
    if detection.boundingbox is not None:
        annotation_types.append("bounding box")
    if detection.keypoints is not None:
        annotation_types.append("keypoints")
    if detection.instance_segmentation is not None:
        annotation_types.append("instance segmentation")
    if detection.sub_detections:
        annotation_types.append("sub-detections")

    class_name = detection.class_name or "(unlabeled)"
    instance_id = str(detection.instance_id)
    rows = [
        ("instance_id", instance_id),
        ("class", class_name),
        ("task", task_name or "(default)"),
        ("annotations", ", ".join(annotation_types)),
    ]
    rows.extend(
        (str(key), str(value)) for key, value in detection.metadata.items()
    )
    return Tooltip(
        title=f"{class_name} #{instance_id}",
        rows=tuple(rows),
        tint=color,
    )


def _style_instance_tree(
    annotation: Annotation,
    color: Color,
    tooltip: Tooltip,
) -> None:
    """Apply one instance identity to every spatial part of its tree."""
    annotation.color = color
    annotation.tooltip = tooltip
    for child in annotation.children:
        _style_instance_tree(child, color, tooltip)
