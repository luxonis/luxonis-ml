"""Coverage for the Skia canvas primitives."""

import numpy as np
import pytest

from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render.canvas import Canvas, gaussian_blur
from luxonis_ml.vizlab.render.markup import Span

_RED = Color(220, 60, 60)


def test_gaussian_blur_softens_edges_and_is_a_noop_at_zero() -> None:
    # A hard black/white vertical edge: a blur spreads it into a ramp of
    # intermediate values; sigma <= 0 returns the image untouched.
    img = np.zeros((20, 20, 4), np.uint8)
    img[..., 3] = 255
    img[:, 10:, :3] = 255

    blurred = gaussian_blur(img, 3.0)
    edge = blurred[10, 6:14, 0]
    assert np.any((edge > 10) & (edge < 245))  # partial values -> softened
    assert blurred.shape == img.shape

    noop = gaussian_blur(img, 0.0)
    assert np.array_equal(noop, img)
    assert noop is not img  # a copy, safe to mutate


def _blank() -> Canvas:
    return Canvas.blank(40, 30)


def test_antialias_flag_controls_shape_edge_smoothing() -> None:
    # A filled circle: anti-aliased edges carry partial alpha; with the flag off
    # the edge is binary (only fully-inside or fully-outside pixels).
    def circle_alpha(antialias: bool) -> np.ndarray:
        canvas = Canvas.blank(40, 40, antialias=antialias)
        canvas.circle((20, 20), 12.0, fill=_RED)
        return canvas.to_rgba()[..., 3]

    soft = circle_alpha(antialias=True)
    hard = circle_alpha(antialias=False)
    assert np.any((soft > 0) & (soft < 255))  # smooth edge -> partial alpha
    assert not np.any(
        (hard > 0) & (hard < 255)
    )  # jagged edge -> 0 or 255 only


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


def test_blit_and_scaled_blit_clip_rounded_corners() -> None:
    patch = np.full((10, 10, 4), 255, dtype=np.uint8)
    direct = Canvas.blank(30, 20)
    direct.blit(patch, 2, 2, radius=4.0)
    scaled = Canvas.blank(30, 20)
    scaled.blit_scaled(patch, 2, 2, 20, 14, radius=5.0)

    assert direct.to_rgba()[7, 7, 3] > 0
    assert scaled.to_rgba()[9, 12, 3] > 0


def test_svg_records_vectors_and_embeds_rasters() -> None:
    canvas = Canvas.svg(120, 80)
    base = np.zeros((40, 60, 4), np.uint8)
    base[..., :] = (30, 40, 50, 255)
    canvas.draw_base(base)  # scaled to fill -> one embedded <image>
    canvas.rounded_rect(Rect(8, 8, 60, 50), radius=4, stroke=_RED)
    canvas.text((12, 40), "car", size=14, color=Color(255, 255, 255))
    svg = canvas.finish_svg().decode("utf-8")
    assert svg.startswith("<?xml")
    assert 'width="120"' in svg  # viewport size
    assert 'height="80"' in svg
    assert "<path" in svg  # the box stroke (and glyphs) are true vectors
    assert "<image" in svg or "base64" in svg  # the base raster is embedded


def test_svg_glyphs_are_paths_by_default_but_text_when_asked() -> None:
    def render(text_as_paths: bool) -> str:
        canvas = Canvas.svg(80, 40, text_as_paths=text_as_paths)
        canvas.text((5, 25), "hi", size=16, color=_RED)
        return canvas.finish_svg().decode("utf-8")

    assert "<text" not in render(True)  # glyphs become <path> outlines
    assert "<text" in render(False)  # ...unless selectable text is kept


def test_svg_canvas_rejects_raster_only_ops() -> None:
    canvas = Canvas.svg(20, 20)
    with pytest.raises(ValueError, match="SVG canvas"):
        canvas.to_rgba()
    with pytest.raises(ValueError, match="SVG canvas"):
        canvas.scaled(10, 10)


def test_finish_svg_rejects_a_raster_canvas() -> None:
    with pytest.raises(ValueError, match="SVG canvas"):
        _blank().finish_svg()


def test_rounded_rect_dashed_and_shadow() -> None:
    from luxonis_ml.vizlab.render.canvas import Shadow

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


def test_empty_spans_measure_and_draw_as_noop() -> None:
    canvas = _blank()
    assert canvas.measure_spans([], 14.0) == canvas.measure_text("", 14.0)
    canvas.draw_spans(
        (2.0, 15.0),
        [Span(""), Span("visible")],
        size=14.0,
        color=_RED,
    )
    assert canvas.to_rgba()[..., 3].max() > 0


def test_overlay_empty_mask_is_noop() -> None:
    canvas = _blank()
    canvas.overlay_mask(
        np.zeros((30, 40), dtype=np.uint8),
        _RED,
        alpha=0.5,
    )
    assert canvas.to_rgba()[..., 3].max() == 0
