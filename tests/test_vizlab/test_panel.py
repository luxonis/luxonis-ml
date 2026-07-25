"""Coverage for the metadata sidebar panel."""

import numpy as np

from luxonis_ml.vizlab import BBox, Image, with_panel
from luxonis_ml.vizlab.panel import _format_tree, _wrap


def _img(w: int = 120, h: int = 80) -> Image:
    return Image(np.full((h, w, 3), 30, np.uint8)).add(
        BBox(x=0.05, y=0.05, w=0.9, h=0.9, label="obj")
    )


def test_format_tree_nested_dict_list_scalar() -> None:
    lines = _format_tree({"a": 1, "b": [2, 3], "c": {"d": True}})
    assert lines == [
        (0, "a: ", True, "1"),
        (0, "b:", True, ""),
        (1, "• ", False, "2"),
        (1, "• ", False, "3"),
        (0, "c:", True, ""),
        (1, "d: ", True, "true"),
    ]


def test_format_tree_scalars_and_types() -> None:
    assert _format_tree(None) == [(0, "", False, "null")]
    assert _format_tree(3.14159)[0][3] == "3.142"
    assert _format_tree([{"x": 1}]) == [
        (0, "•", False, ""),
        (1, "x: ", True, "1"),
    ]


def test_format_tree_truncates_long_string() -> None:
    body = _format_tree("z" * 800)[0][3]
    assert body.endswith("…")
    assert len(body) == 500


def test_wrap_splits_and_handles_empty() -> None:
    assert _wrap("", 12.0, 400, 100.0) == [""]
    wrapped = _wrap("word " * 40, 13.5, 400, 80.0)
    assert len(wrapped) > 1


def test_with_panel_right_widens_output() -> None:
    img = _img(120, 80)
    out = with_panel(img, {"source": "a.jpg", "split": "train"}, title="meta")
    rendered = out.render()
    assert rendered.shape[1] > 120  # widened by the panel
    assert rendered.shape[0] >= 80


def test_with_panel_sides() -> None:
    img = _img(100, 60)
    data = {"aug": ["flip", "blur"], "tags": {"hard": True}}
    right = with_panel(img, data, side="right").render()
    left = with_panel(img, data, side="left").render()
    bottom = with_panel(img, data, side="bottom").render()
    assert right.shape[1] > 100
    assert left.shape[1] == right.shape[1]
    assert bottom.shape[0] > 60


def test_with_panel_explicit_width_and_method() -> None:
    img = _img(100, 60)
    out = img.with_panel({"k": "v"}, width=180.0)
    assert out.render().shape[1] == 100 + 180


def test_with_panel_wraps_long_value() -> None:
    img = _img(100, 60)
    out = with_panel(img, {"note": "lorem ipsum dolor sit amet " * 6})
    # Renders without error and stays a valid RGBA image.
    assert out.render().shape[2] == 4
