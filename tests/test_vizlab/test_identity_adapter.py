"""Color-identity coverage for LDF record rendering."""

from pathlib import Path

import numpy as np

from luxonis_ml.ldf import (
    BBoxAnnotation,
    DatasetRecord,
    Detection,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
)
from luxonis_ml.vizlab import Palette, RenderOptions
from luxonis_ml.vizlab.adapters import records_to_colored_annotations


def _record(
    task_name: str,
    detections: list[Detection],
) -> DatasetRecord:
    return DatasetRecord.model_construct(
        files={"image": Path("frame.jpg")},
        annotation=detections,
        task_name=task_name,
    )


def _box(instance_id: int, *, metadata: bool = False) -> Detection:
    return Detection(
        class_name="car",
        instance_id=instance_id,
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2),
        metadata={"quality": "good"} if metadata else {},
    )


def test_class_coloring_keeps_palette_driven_class_identity() -> None:
    annotations = records_to_colored_annotations(
        [_record("left", [_box(1)]), _record("right", [_box(2)])],
        color_by="class",
        options=RenderOptions(),
        identity_palette=Palette(),
    )
    assert len(annotations) == 2
    assert annotations[0].label == annotations[1].label == "car"
    assert annotations[0].color is None
    assert annotations[1].color is None


def test_instance_coloring_distinguishes_same_class_instances() -> None:
    annotations = records_to_colored_annotations(
        [_record("objects", [_box(1), _box(2)])],
        color_by="instance",
        options=RenderOptions(),
        identity_palette=Palette(),
    )
    assert len(annotations) == 2
    assert annotations[0].color != annotations[1].color
    assert annotations[0].tooltip is not None
    assert annotations[0].tooltip.rows[0] == ("instance_id", "1")


def test_task_coloring_is_stable_within_task_and_distinct_across_tasks() -> (
    None
):
    annotations = records_to_colored_annotations(
        [
            _record("left", [_box(1, metadata=True), _box(2)]),
            _record("right", [_box(3)]),
        ],
        color_by="task",
        options=RenderOptions(hover_metadata=True),
        identity_palette=Palette(),
    )
    first, second, third = annotations
    assert first.color == second.color
    assert first.color != third.color
    assert first.tooltip is not None
    assert first.tooltip.tint == first.color


def test_instance_coloring_describes_and_styles_the_whole_instance() -> None:
    detection = Detection(
        class_name=None,
        instance_id=-1,
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.4, h=0.4),
        keypoints=KeypointAnnotation(keypoints=[(0.2, 0.2, 2)]),
        instance_segmentation=InstanceSegmentationAnnotation(
            mask=np.ones((8, 8), np.uint8)  # type: ignore[arg-type]
        ),
        sub_detections={
            "part": Detection(
                class_name="wheel",
                boundingbox=BBoxAnnotation(x=0.2, y=0.2, w=0.1, h=0.1),
            )
        },
        metadata={"source": "manual"},
    )
    (root,) = records_to_colored_annotations(
        [_record("", [detection])],
        color_by="instance",
        options=RenderOptions(),
        identity_palette=Palette(),
    )

    assert root.tooltip is not None
    assert root.tooltip.title == "(unlabeled) #-1"
    assert ("task", "(default)") in root.tooltip.rows
    assert (
        "annotations",
        "bounding box, keypoints, instance segmentation, sub-detections",
    ) in root.tooltip.rows
    assert ("source", "manual") in root.tooltip.rows
    assert root.children
    assert all(child.color == root.color for child in root.children)
    assert all(child.tooltip == root.tooltip for child in root.children)


def test_task_coloring_styles_nested_children_without_a_tooltip() -> None:
    detection = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.4, h=0.4),
        keypoints=KeypointAnnotation(keypoints=[(0.2, 0.2, 2)]),
    )
    (root,) = records_to_colored_annotations(
        [_record("pose", [detection])],
        color_by="task",
        options=RenderOptions(hover_metadata=False),
        identity_palette=Palette(),
    )

    assert root.tooltip is None
    assert root.children
    assert all(child.color == root.color for child in root.children)
