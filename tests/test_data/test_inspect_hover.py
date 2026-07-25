"""Coverage for the ``data inspect`` metadata hover helpers."""

from pathlib import Path

import numpy as np

from luxonis_ml.data.__main__ import (
    _collect_hover_items,
    _draw_tooltip,
    _hit_test,
    _hover_items,
)
from luxonis_ml.ldf import BBoxAnnotation, DatasetRecord, Detection


def _records() -> dict[str, DatasetRecord]:
    with_meta = Detection(
        class_name="car",
        instance_id=3,
        boundingbox=BBoxAnnotation(x=0.1, y=0.2, w=0.4, h=0.3),
        metadata={"track_id": 42, "weather": "sunny"},
    )
    no_meta = Detection(
        class_name="person",
        boundingbox=BBoxAnnotation(x=0.6, y=0.6, w=0.2, h=0.3),
    )
    record = DatasetRecord.model_construct(
        files={"image": Path("x")},
        annotation=[with_meta, no_meta],
        task_name="obj",
    )
    return {"obj": record}


def test_hover_items_only_for_metadata_boxes() -> None:
    items = _hover_items(_records(), 1000, 500)
    # Only the detection carrying metadata contributes an item.
    assert len(items) == 1
    rect, info = items[0]
    assert rect == (100.0, 100.0, 500.0, 250.0)  # normalized * (1000, 500)
    assert info["_title"] == "car #3"
    assert info["track_id"] == 42
    assert info["weather"] == "sunny"


def test_collect_hover_recurses_sub_detections() -> None:
    parent = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.0, y=0.0, w=0.5, h=0.5),
        sub_detections={
            "plate": Detection(
                class_name="plate",
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.1, h=0.05),
                metadata={"text": "AB123"},
            )
        },
    )
    items: list = []
    _collect_hover_items(parent, 100, 100, items)
    # Parent has no metadata; only the sub-detection contributes.
    assert len(items) == 1
    assert items[0][1]["text"] == "AB123"


def test_collect_hover_applies_tile_offset() -> None:
    """Grid tiles shift box rects by the tile's composite offset."""
    detection = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.0, y=0.0, w=0.5, h=0.5),
        metadata={"id": 1},
    )
    items: list = []
    _collect_hover_items(detection, 200, 100, items, 320.0, 40.0)
    rect, _ = items[0]
    # (0,0,0.5,0.5) * (200,100) shifted by (320, 40).
    assert rect == (320.0, 40.0, 420.0, 90.0)


def test_hover_title_uses_class_color() -> None:
    """The tooltip title is tinted with the detection's class color (BGR)."""
    from luxonis_ml.vizlab import Palette

    palette = Palette(["car", "person"])
    detection = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2),
        metadata={"id": 1},
    )
    items: list = []
    _collect_hover_items(detection, 100, 100, items, palette=palette)
    color = palette.color_for("car")
    assert items[0][1]["_color"] == (color.b, color.g, color.r)


def test_hover_title_no_color_without_palette() -> None:
    detection = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2),
        metadata={"id": 1},
    )
    items: list = []
    _collect_hover_items(detection, 100, 100, items)
    assert "_color" not in items[0][1]


def test_hit_test_picks_smallest_containing_box() -> None:
    items = [
        ((0.0, 0.0, 100.0, 100.0), {"_title": "big"}),
        ((10.0, 10.0, 40.0, 40.0), {"_title": "small"}),
    ]
    assert _hit_test(items, 25, 25) == 1  # inside both -> smaller wins
    assert _hit_test(items, 80, 80) == 0  # inside only the big box
    assert _hit_test(items, 200, 200) is None  # outside all


def test_draw_tooltip_modifies_frame_and_clamps() -> None:
    frame = np.full((200, 300, 3), 40, np.uint8)
    before = frame.copy()
    # Cursor near the bottom-right corner: the card must stay in-bounds.
    _draw_tooltip(frame, {"_title": "car #3", "track_id": 42}, (295, 195))
    assert not np.array_equal(frame, before)


def test_draw_tooltip_empty_info_is_noop() -> None:
    frame = np.full((100, 100, 3), 40, np.uint8)
    before = frame.copy()
    _draw_tooltip(frame, {}, (10, 10))
    assert np.array_equal(frame, before)
