"""Tests for `VizConfig.hover_metadata` attaching tooltips in convert."""

from luxonis_ml.ldf import Detection
from luxonis_ml.vizlab import BBox, Palette, VizConfig
from luxonis_ml.vizlab.convert import detection_to_annotations


def _first_box(det: Detection, config: VizConfig) -> BBox:
    annotations = detection_to_annotations(det, config)
    boxes = [a for a in annotations if isinstance(a, BBox)]
    assert boxes, "expected a BBox annotation"
    return boxes[0]


def _car(**metadata: object) -> Detection:
    data: dict[str, object] = {
        "class_name": "car",
        "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        "instance_id": 3,
    }
    if metadata:
        data["metadata"] = metadata
    return Detection.model_validate(data)


def test_hover_metadata_attaches_tooltip() -> None:
    palette = Palette(["car"])
    config = VizConfig(palette=palette, hover_metadata=True)
    box = _first_box(_car(text="AB123", track_id=7), config)
    tip = box.tooltip
    assert tip is not None
    assert tip.title == "car #3"  # class name + instance id
    assert tip.rows == (("track_id", "7"),)  # the OCR "text" stays on the chip
    assert tip.tint == palette.color_for("car")


def test_hover_metadata_off_by_default() -> None:
    box = _first_box(_car(track_id=7), VizConfig(palette=Palette(["car"])))
    assert box.tooltip is None


def test_no_tooltip_when_only_text_metadata() -> None:
    # Recognized text is shown on the chip; with nothing else, no hover.
    box = _first_box(_car(text="AB123"), VizConfig(hover_metadata=True))
    assert box.tooltip is None


def test_no_tooltip_without_metadata() -> None:
    box = _first_box(_car(), VizConfig(hover_metadata=True))
    assert box.tooltip is None
