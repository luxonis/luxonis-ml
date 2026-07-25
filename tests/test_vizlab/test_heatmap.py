"""Coverage for gradients and the Heatmap overlay."""

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    GRADIENTS,
    Color,
    Gradient,
    Heatmap,
    Image,
    resolve_gradient,
)
from luxonis_ml.vizlab.gradient import DEFAULT_GRADIENT


def _canvas(w: int = 40, h: int = 30) -> Image:
    return Image(np.full((h, w, 3), 30, np.uint8))


# --- Gradient ---------------------------------------------------------------


def test_gradient_from_colors_interpolates_endpoints_and_middle() -> None:
    g = Gradient.from_colors(["#000000", "#ff0000"])
    assert g.color_at(0.0) == Color(0, 0, 0)
    assert g.color_at(1.0) == Color(255, 0, 0)
    # Halfway is the linear midpoint of the two stops.
    assert g.color_at(0.5) == Color(128, 0, 0)


def test_gradient_clamps_out_of_range() -> None:
    g = Gradient.from_colors(["#000000", "#ffffff"])
    assert g.color_at(-1.0) == Color(0, 0, 0)
    assert g.color_at(2.0) == Color(255, 255, 255)


def test_gradient_custom_positions_and_sorting() -> None:
    # Positions may be given out of order; stops are sorted ascending.
    g = Gradient.from_colors(["#ffffff", "#000000"], positions=[1.0, 0.0])
    assert g.color_at(0.0) == Color(0, 0, 0)
    assert g.color_at(1.0) == Color(255, 255, 255)


def test_gradient_requires_two_colors() -> None:
    with pytest.raises(ValueError, match="at least two"):
        Gradient.from_colors(["#000000"])


def test_gradient_positions_length_must_match() -> None:
    with pytest.raises(ValueError, match="one entry per color"):
        Gradient.from_colors(["#000000", "#ffffff"], positions=[0.0])


def test_gradient_colorize_shape_and_dtype() -> None:
    field = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    rgb = GRADIENTS["viridis"].colorize(field)
    assert rgb.shape == (3, 4, 3)
    assert rgb.dtype == np.uint8


def test_resolve_gradient_by_name_and_instance() -> None:
    assert resolve_gradient("jet") is GRADIENTS["jet"]
    custom = Gradient.from_colors(["#000000", "#ffffff"])
    assert resolve_gradient(custom) is custom


def test_resolve_gradient_unknown_name_lists_options() -> None:
    with pytest.raises(KeyError, match="unknown gradient"):
        resolve_gradient("not-a-real-gradient")


def test_default_gradient_is_registered() -> None:
    assert DEFAULT_GRADIENT in GRADIENTS


# --- Heatmap ----------------------------------------------------------------


def test_heatmap_renders_over_image() -> None:
    field = np.linspace(0.0, 1.0, 40 * 30).reshape(30, 40)
    out = _canvas().add(Heatmap(values=field)).render()
    assert out.shape == (30, 40, 4)
    # The hot end of the field is colored over the base pixels.
    assert not np.array_equal(out[..., :3], _canvas().render()[..., :3])


def test_heatmap_none_values_is_noop() -> None:
    base = _canvas()
    plain = base.copy().render()
    with_none = base.copy().add(Heatmap(values=None)).render()
    assert np.array_equal(plain, with_none)


def test_heatmap_weight_by_value_fades_cold_regions() -> None:
    # Left half cold (0), right half hot (1).
    field = np.zeros((30, 40), dtype=np.float64)
    field[:, 20:] = 1.0
    base = _canvas()
    out = base.copy().add(Heatmap(values=field)).render()
    plain = base.copy().render()
    # Cold half keeps the original pixels; hot half is recolored.
    assert np.array_equal(out[:, :20, :3], plain[:, :20, :3])
    assert not np.array_equal(out[:, 20:, :3], plain[:, 20:, :3])


def test_heatmap_flat_alpha_covers_everywhere() -> None:
    field = np.zeros((30, 40), dtype=np.float64)
    base = _canvas()
    out = (
        base.copy()
        .add(Heatmap(values=field, weight_by_value=False, alpha=0.6))
        .render()
    )
    plain = base.copy().render()
    # A flat (all-zero) field still tints the whole image when not weighted.
    assert not np.array_equal(out[..., :3], plain[..., :3])


def test_heatmap_gradient_by_name_matches_instance() -> None:
    field = np.linspace(0.0, 1.0, 40 * 30).reshape(30, 40)
    by_name = _canvas().add(Heatmap(values=field, gradient="magma")).render()
    by_obj = (
        _canvas()
        .add(Heatmap(values=field, gradient=GRADIENTS["magma"]))
        .render()
    )
    assert np.array_equal(by_name, by_obj)


def test_heatmap_low_res_field_upsamples_to_canvas() -> None:
    # An 8x6 field on a 40x30 canvas resamples without error.
    field = np.linspace(0.0, 1.0, 8 * 6).reshape(6, 8)
    out = _canvas(40, 30).add(Heatmap(values=field)).render()
    assert out.shape == (30, 40, 4)
    assert out[..., 3].max() > 0


def test_heatmap_vmin_vmax_clip_range() -> None:
    field = np.array([[0.0, 5.0, 10.0]], dtype=np.float64)
    heat = Heatmap(values=field, vmin=0.0, vmax=10.0)
    normalized = heat._normalized(field)
    assert np.allclose(normalized, [[0.0, 0.5, 1.0]])


def test_heatmap_constant_field_normalizes_to_zero() -> None:
    # A flat field has no range; normalization must not divide by zero.
    field = np.full((4, 4), 7.0)
    assert np.array_equal(
        Heatmap(values=field)._normalized(field), np.zeros((4, 4))
    )
