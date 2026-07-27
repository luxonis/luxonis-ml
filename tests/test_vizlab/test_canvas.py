"""Coverage for the Skia canvas primitives."""

import numpy as np

from luxonis_ml.vizlab.canvas import Canvas
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect

_RED = Color(220, 60, 60)


def _blank() -> Canvas:
    return Canvas.blank(40, 30)


def test_line_and_circle_draw() -> None:
    canvas = _blank()
    canvas.line((2, 2), (30, 20), _RED, width=3.0)
    canvas.circle((20, 15), 6.0, fill=_RED, stroke=Color(255, 255, 255))
    out = canvas.to_rgba()
    assert out.shape == (30, 40, 4)
    assert out[..., 3].max() > 0  # something was drawn


def test_gradient_line_fades_between_endpoints() -> None:
    # A wide horizontal gradient line from green to red: the left end reads
    # greener, the right end redder, and the middle sits between the two.
    canvas = Canvas.blank(60, 12)
    green, red = Color(40, 200, 120), Color(230, 60, 60)
    canvas.gradient_line((2, 6), (58, 6), green, red, width=10.0)
    out = canvas.to_rgba()[6]  # the center scanline

    def sample(x: int) -> tuple[int, int]:
        return int(out[x, 0]), int(out[x, 1])  # (r, g)

    left_r, left_g = sample(5)
    mid_r, mid_g = sample(30)
    right_r, right_g = sample(54)
    assert left_g > left_r  # green-dominant near the start
    assert right_r > right_g  # red-dominant near the end
    assert left_r < mid_r < right_r  # red rises left -> right
    assert left_g > mid_g > right_g  # green falls left -> right


def test_measure_and_text() -> None:
    canvas = _blank()
    metrics = canvas.measure_text("Ag", 16.0, weight=600)
    assert metrics.width > 0
    assert metrics.height == metrics.ascent + metrics.descent
    canvas.text((2, 20), "hi", size=14.0, color=_RED)
    assert canvas.to_rgba()[..., 3].max() > 0


def test_polygon_needs_two_points() -> None:
    canvas = _blank()
    canvas.polygon([(1.0, 1.0)], stroke=_RED)  # < 2 points: no-op
    assert canvas.to_rgba()[..., 3].max() == 0
    canvas.polygon(
        [(2.0, 2.0), (20.0, 2.0), (10.0, 20.0)], fill=_RED, stroke=_RED
    )
    assert canvas.to_rgba()[..., 3].max() > 0


def test_overlay_mask_bool_and_float() -> None:
    canvas = _blank()
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:15, 5:15] = True
    canvas.overlay_mask(mask, _RED, alpha=0.5)
    assert canvas.to_rgba()[10, 10, 3] > 0

    canvas2 = _blank()
    fmask = np.zeros((30, 40), dtype=np.float32)
    fmask[5:15, 5:15] = 0.9  # > 0.5 -> filled
    canvas2.overlay_mask(fmask, _RED)
    assert canvas2.to_rgba()[10, 10, 3] > 0


def test_blit_places_image() -> None:
    canvas = _blank()
    patch = np.zeros((8, 8, 4), dtype=np.uint8)
    patch[..., :] = (200, 100, 50, 255)
    canvas.blit(patch, 4, 4)
    assert tuple(canvas.to_rgba()[8, 8, :3]) == (200, 100, 50)


def test_rounded_rect_dashed_and_shadow() -> None:
    from luxonis_ml.vizlab.canvas import Shadow

    canvas = _blank()
    canvas.rounded_rect(
        Rect(3, 3, 35, 25),
        radius=5.0,
        fill=_RED.with_alpha(0.3),
        stroke=_RED,
        dash=(4.0, 3.0),
        shadow=Shadow(),
    )
    assert canvas.to_rgba()[..., 3].max() > 0
