"""Tests for `Image.render_hits` and the render capture seam."""

import numpy as np

from luxonis_ml.vizlab import BBox, HitMap, Image, Keypoints, Tooltip
from luxonis_ml.vizlab.geometry import Rect


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


def test_carried_hit_region_scales_independently_on_each_axis() -> None:
    tip = Tooltip(title="car")
    image = Image(np.zeros((100, 100, 3), np.uint8)).with_hitmap(
        HitMap([(Rect(10, 20, 30, 40), tip)])
    )

    _, hits = image.render_hits((200, 50))

    rect, _ = hits.items[0]
    assert (rect.left, rect.top, rect.right, rect.bottom) == (
        20,
        10,
        60,
        20,
    )


def test_chip_less_box_with_tooltip_still_hits() -> None:
    # No label/score/payload -> no chip is drawn, but the region still hits.
    tip = Tooltip(rows=(("k", "v"),))
    img = _blank(60, 100).add(BBox(x=0.2, y=0.2, w=0.4, h=0.4, tooltip=tip))
    _, hits = img.render_hits()
    assert hits.hit(40, 30) is tip


def test_keypoints_without_tooltip_emit_no_hit() -> None:
    img = _blank(60, 100).add(
        Keypoints(keypoints=[(0.3, 0.3, 2), (0.6, 0.6, 2)], label="pose")
    )
    _, hits = img.render_hits()
    assert hits.items == []


def test_keypoints_tooltip_hits_over_the_joints() -> None:
    tip = Tooltip(title="pose", rows=(("id", "3"),))
    img = _blank(60, 100).add(
        Keypoints(keypoints=[(0.3, 0.3, 2), (0.6, 0.6, 2)], tooltip=tip)
    )
    _, hits = img.render_hits()
    assert len(hits.items) == 1
    # A joint at (0.3, 0.3) -> (30, 18) px on a 100x60 canvas is inside the
    # padded bounds; a far corner is not.
    assert hits.hit(30, 18) is tip
    assert hits.hit(2, 2) is None


def test_keypoints_hit_ignores_occluded_only_sets() -> None:
    # All points below the visibility threshold -> nothing visible -> no region.
    tip = Tooltip(title="pose")
    img = _blank(60, 100).add(
        Keypoints(
            keypoints=[(0.3, 0.3, 0), (0.6, 0.6, 0)],
            visibility_threshold=0.0,
            tooltip=tip,
        )
    )
    _, hits = img.render_hits()
    assert hits.items == []
