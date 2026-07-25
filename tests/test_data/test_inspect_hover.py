"""Coverage for the ``data inspect`` metadata hover helpers."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import luxonis_ml.data.__main__ as data_main
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


def test_per_instance_inspect_attaches_augmentation_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    record = DatasetRecord.model_construct(
        files={},
        annotation=[
            Detection(
                class_name="car",
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.4, h=0.4),
            )
        ],
        task_name="objects",
    )

    class _Dataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def get_classes(self) -> dict[str, dict[str, int]]:
            return {"objects": {"car": 0}}

        def get_class_names(self) -> dict[str, list[str]]:
            return {"objects": ["car"]}

        def get_categorical_encodings(self) -> dict[str, object]:
            return {}

        def get_skeletons(self) -> dict[str, object]:
            return {}

    class _Loader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self._augmentations = None

        def __getitem__(
            self, _index: int
        ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
            return {"image": image}, {}

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                images={"image": image},
                labels={},
                metadata={},
            )

        def _init_augmentations(self, **_kwargs: object) -> object:
            return object()

    class _Collector:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get_applied_augmentations(self) -> list[str]:
            return ["HorizontalFlip"]

    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import Image

    panels: list[object] = []

    def capture_panel(self: Image, data: object, **_kwargs: object) -> Image:
        panels.append(data)
        return self

    monkeypatch.setattr(data_main, "check_exists", lambda *_args: None)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(data_main, "AugmentationsCollector", _Collector)
    monkeypatch.setattr(data_main, "_screen_size", lambda: None)
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_args, **_kwargs: {"objects": record},
    )
    monkeypatch.setattr(Image, "with_panel", capture_panel)
    monkeypatch.setattr(data_main.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(data_main.cv2, "resizeWindow", lambda *_args: None)
    monkeypatch.setattr(data_main.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(data_main.cv2, "waitKey", lambda *_args: ord("q"))

    from luxonis_ml.vizlab import (
        DARK_THEME,
        LIGHT_THEME,
        get_default_theme,
        set_default_theme,
    )

    aug_config = tmp_path / "augmentations.json"
    aug_config.write_text("[]")
    try:
        data_main.inspect(
            "dataset",
            aug_config=aug_config,
            per_instance=True,
            list_augmentations=True,
            theme="light",
        )
        # The chosen theme becomes the process default for the visualization.
        assert get_default_theme() is LIGHT_THEME
    finally:
        set_default_theme(DARK_THEME)

    assert panels == [{"augmentations": ["HorizontalFlip"]}]
