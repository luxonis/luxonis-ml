"""Coverage for oriented BBox behavior (normalized xywh + angle)."""

import math

import numpy as np
import pytest

from luxonis_ml.vizlab import BBox, Image
from luxonis_ml.vizlab.geometry import oriented_corners


def _img(w: int = 160, h: int = 120) -> Image:
    return Image(np.full((h, w, 3), 30, np.uint8))


def test_oriented_corners_axis_aligned_and_rotated() -> None:
    assert oriented_corners(0.0, 0.0, 2.0, 4.0, 0.0) == (
        (-1.0, -2.0),
        (1.0, -2.0),
        (1.0, 2.0),
        (-1.0, 2.0),
    )
    # 90° rotation swaps the extents: a 2x4 box becomes 4 wide, 2 tall.
    rotated = oriented_corners(0.0, 0.0, 2.0, 4.0, math.pi / 2)
    xs = [x for x, _ in rotated]
    ys = [y for _, y in rotated]
    assert max(xs) - min(xs) == pytest.approx(4.0)
    assert max(ys) - min(ys) == pytest.approx(2.0)


def test_bbox_axis_aligned_detection() -> None:
    assert BBox(x=0.1, y=0.2, w=0.3, h=0.4)._axis_aligned() is True
    assert BBox(x=0.1, y=0.2, w=0.3, h=0.4, angle=15)._axis_aligned() is False


def test_bbox_angle_rotates_square() -> None:
    # normalized 0.2x0.2 on 100x100 -> 20x20 px; rotated 90° stays 20x20.
    corners = BBox(x=0.4, y=0.4, w=0.2, h=0.2, angle=90)._corners(100, 100)
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    assert max(xs) - min(xs) == pytest.approx(20.0)
    assert max(ys) - min(ys) == pytest.approx(20.0)


def test_bbox_angle_rotates_rect() -> None:
    # normalized 0.2x0.1 on 200x100 -> 40x10 px; rotated 90° -> 10 wide, 40 tall.
    corners = BBox(x=0.4, y=0.45, w=0.2, h=0.1, angle=90)._corners(200, 100)
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    assert max(xs) - min(xs) == pytest.approx(10.0)
    assert max(ys) - min(ys) == pytest.approx(40.0)


def test_bbox_radians_angle_unit() -> None:
    rad = BBox(x=0.3, y=0.4, w=0.2, h=0.1, angle=math.pi / 2, angle_unit="rad")
    xs = [x for x, _ in rad._corners(100, 100)]
    # 0.2x0.1 on 100 -> 20x10 px; rotated 90° -> width 10.
    assert max(xs) - min(xs) == pytest.approx(10.0)


def test_bbox_oriented_renders_as_polygon() -> None:
    base = _img()
    plain = base.copy().render()
    rotated = (
        base.copy().add(BBox(x=0.3, y=0.3, w=0.4, h=0.3, angle=25)).render()
    )
    assert not np.array_equal(rotated, plain)


def test_bbox_axis_aligned_renders() -> None:
    base = _img()
    out = (
        base.copy()
        .add(BBox(x=0.1, y=0.1, w=0.5, h=0.5, color="#4c8dff"))
        .render()
    )
    assert not np.array_equal(out, base.copy().render())
