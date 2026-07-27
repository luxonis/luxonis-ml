"""Tests for `Image.render_hits` and the render capture seam."""

import numpy as np

from luxonis_ml.vizlab import BBox, Image, Tooltip


def _blank(h: int = 60, w: int = 100) -> Image:
    return Image(np.zeros((h, w, 3), np.uint8))


def test_render_hits_pixels_match_render() -> None:
    img = _blank().add(
        BBox(x=0.1, y=0.2, w=0.4, h=0.5, label="car", score=0.9)
    )
    plain = img.render()
    captured, _ = img.render_hits()
    assert np.array_equal(plain, captured)


def test_render_hits_pixels_match_render_when_scaled() -> None:
    img = _blank().add(BBox(x=0.1, y=0.2, w=0.4, h=0.5, label="car"))
    size = (200, 120)
    assert np.array_equal(img.render(size), img.render_hits(size)[0])


def test_box_without_tooltip_emits_no_hit() -> None:
    img = _blank().add(BBox(x=0.1, y=0.2, w=0.4, h=0.5, label="car"))
    _, hits = img.render_hits()
    assert hits.items == []


def test_tooltip_box_emits_hit_at_its_region() -> None:
    tip = Tooltip(title="car", rows=(("id", "7"),))
    img = _blank(60, 100).add(
        BBox(x=0.1, y=0.2, w=0.4, h=0.5, label="car", tooltip=tip)
    )
    _, hits = img.render_hits()
    assert len(hits.items) == 1
    # The box spans x in [10, 50] px and y in [12, 42] px on a 100x60 canvas;
    # its center must resolve to the tooltip.
    assert hits.hit(30, 27) is tip
    assert hits.hit(0, 0) is None


def test_hit_region_scales_with_render_size() -> None:
    tip = Tooltip(title="car")
    img = _blank(60, 100).add(BBox(x=0.1, y=0.2, w=0.4, h=0.5, tooltip=tip))
    _, hits = img.render_hits((200, 120))  # 2x each axis
    rect, _ = hits.items[0]
    # Box left/top of (0.1, 0.2) -> (20, 24) px at the doubled size.
    assert rect.left == 20.0
    assert rect.top == 24.0


def test_chip_less_box_with_tooltip_still_hits() -> None:
    # No label/score/payload -> no chip is drawn, but the region still hits.
    tip = Tooltip(rows=(("k", "v"),))
    img = _blank(60, 100).add(BBox(x=0.2, y=0.2, w=0.4, h=0.4, tooltip=tip))
    _, hits = img.render_hits()
    assert hits.hit(40, 30) is tip
