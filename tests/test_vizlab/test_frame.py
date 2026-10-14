"""Tests for the `Frame` value type (image + hover map bundle)."""

import numpy as np

from luxonis_ml.vizlab import BBox, Frame, HitMap, Image, Style, Tooltip
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render.capture import ClickMap


def _tooltip_image() -> Image:
    return Image(np.zeros((80, 120, 3), np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.5, tooltip=Tooltip(title="car"))
    )


def test_frame_holds_image_and_hitmap() -> None:
    image = _tooltip_image()
    hits = HitMap.empty()
    frame = Frame(image, hits)
    assert frame.image is image
    assert frame.hitmap is hits


def test_frame_defaults_to_empty_hitmap() -> None:
    assert Frame(_tooltip_image()).hitmap.items == []


def test_image_frame_captures_hover_regions() -> None:
    frame = _tooltip_image().frame()
    assert isinstance(frame, Frame)
    # The box carries a tooltip, so its region is captured.
    assert frame.hitmap.hit(120 * 0.3, 80 * 0.3) is not None


def test_frame_render_matches_image_render() -> None:
    image = _tooltip_image()
    frame = image.frame()
    assert np.array_equal(frame.render(), image.render())


def test_frame_keeps_the_style_scope_used_to_capture_interactions() -> None:
    image = _tooltip_image()
    with Style.override(stroke_width=9.0, shadow=False):
        expected = image.render()
        frame = image.frame()

    assert np.array_equal(frame.render(), expected)


def test_with_image_keeps_hitmap() -> None:
    frame = Frame(
        _tooltip_image(),
        HitMap.empty(),
        ClickMap([(Rect(1, 2, 3, 4), "key:m")]),
    )
    replacement = Image(np.zeros((80, 120, 3), np.uint8))
    swapped = frame.with_image(replacement)
    assert swapped.image is replacement
    assert swapped.hitmap is frame.hitmap
    assert swapped.clickmap is frame.clickmap
