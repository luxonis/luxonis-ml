"""Color-identity coverage for LDF record rendering."""

from pathlib import Path

from luxonis_ml.ldf import BBoxAnnotation, DatasetRecord, Detection
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
