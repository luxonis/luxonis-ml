"""Coverage for the metadata sidebar panel."""

import numpy as np

from luxonis_ml.vizlab import BBox, Image, with_panel
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.panel import (
    _PAD,
    _PANEL_SIZE,
    _build_ops,
    _format_tree,
    _metrics,
    _wrap,
)


def test_value_ops_are_monospace_keys_are_not() -> None:
    """Values render in JetBrains Mono; keys/bullets stay in Inter (sans)."""
    m = _metrics(1.0, Color(24, 24, 28))
    lines = _format_tree({"speed": 12.4})
    ops, _ = _build_ops(lines, content_w=200.0, m=m)
    by_text = {op[2]: op[5] for op in ops}  # text -> mono flag
    assert by_text["speed: "] is False  # the key is sans
    assert by_text["12.4"] is True  # the value is mono


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


def test_metrics_scale_proportionally() -> None:
    from luxonis_ml.vizlab.color import Color

    m = _metrics(2.0, Color(24, 24, 28))
    assert m.size == _PANEL_SIZE * 2.0
    assert m.pad == _PAD * 2.0


def test_metrics_colors_adapt_to_background() -> None:
    """Panel text is a light purple on dark and a deeper purple on light."""
    from luxonis_ml.vizlab.color import Color

    dark = _metrics(1.0, Color(24, 24, 28))
    light = _metrics(1.0, Color(240, 240, 244))
    # Both on-brand blue-purple (not plain white/black), lighter on the dark bg.
    assert dark.value.b > dark.value.r
    assert light.value.b > light.value.r
    assert sum(dark.value.rgb) > sum(light.value.rgb)


def test_panel_type_scales_with_image_size() -> None:
    """A larger image gets a larger (not shrinking) panel, so type stays legible."""
    data = {
        "source": "frame.jpg",
        "note": "a longer value that may wrap around",
    }
    small = with_panel(_img(200, 200), data).render()
    large = with_panel(_img(1600, 1600), data).render()
    small_panel_w = small.shape[1] - 200
    large_panel_w = large.shape[1] - 1600
    assert large_panel_w > small_panel_w


def test_with_panel_sides() -> None:
    img = _img(100, 60)
    data = {"aug": ["flip", "blur"], "tags": {"hard": True}}
    right = with_panel(img, data, side="right").render()
    left = with_panel(img, data, side="left").render()
    bottom = with_panel(img, data, side="bottom").render()
    assert right.shape[1] > 100
    assert left.shape[1] == right.shape[1]
    assert bottom.shape[0] > 60


def test_bottom_panel_keeps_image_above_panel() -> None:
    img = _img(100, 60)
    source = img.render()
    bottom = with_panel(img, {"aug": "flip"}, side="bottom").render()

    assert np.array_equal(bottom[:59, :100], source[:59])


def test_with_panel_explicit_width_and_method() -> None:
    img = _img(100, 60)
    out = img.with_panel({"k": "v"}, width=180.0)
    assert out.render().shape[1] == 100 + 180


def test_with_panel_wraps_long_value() -> None:
    img = _img(100, 60)
    out = with_panel(img, {"note": "lorem ipsum dolor sit amet " * 6})
    # Renders without error and stays a valid RGBA image.
    assert out.render().shape[2] == 4
