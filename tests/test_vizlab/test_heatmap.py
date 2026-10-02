"""Coverage for gradients, the Heatmap overlay, and its color key."""

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    GRADIENTS,
    Color,
    ColorBar,
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


def test_heatmap_without_normalization_uses_unit_range() -> None:
    field = np.array([[-1.0, 0.5, 2.0]])
    normalized = Heatmap(normalize=False)._normalized(field)
    assert np.array_equal(normalized, [[0.0, 0.5, 1.0]])


def test_heatmap_extent_is_none() -> None:
    assert Heatmap(values=np.ones((2, 2))).extent() is None


# --- nodata and non-finite values -------------------------------------------


def test_heatmap_nan_does_not_poison_normalization() -> None:
    # A bare min()/max() propagates NaN through the whole field, which the
    # gradient then casts to arbitrary bytes -- silent, total corruption.
    clean = np.array([[1.0, 2.0, 3.0]])
    dirty = np.array([[1.0, 2.0, 3.0, np.nan]])
    heat = Heatmap(values=dirty)
    normalized = heat._normalized(dirty)
    assert np.isfinite(normalized).all()
    assert np.allclose(
        normalized[0, :3], Heatmap(values=clean)._normalized(clean)[0]
    )
    assert normalized[0, 3] == 0.0  # the NaN lands at zero, not at NaN


def test_heatmap_infinities_do_not_stretch_the_range() -> None:
    field = np.array([[0.0, 5.0, 10.0, np.inf, -np.inf]])
    assert Heatmap(values=field).value_range() == (0.0, 10.0)


def test_heatmap_ignore_value_is_excluded_from_the_auto_range() -> None:
    field = np.array([[0.0, 5.0, 10.0]])
    assert Heatmap(values=field).value_range() == (0.0, 10.0)
    assert Heatmap(values=field, ignore_value=0.0).value_range() == (5.0, 10.0)


def test_heatmap_ignore_value_pixels_stay_transparent() -> None:
    # The left half is the nodata sentinel, so those pixels must come through
    # untouched while the right half -- which carries a real range -- is
    # painted. weight_by_value is off so opacity isolates nodata from magnitude.
    field = np.zeros((30, 40))
    field[:, 20:] = np.linspace(1.0, 10.0, 20)[None, :]
    base = _canvas().render()
    out = (
        _canvas()
        .add(Heatmap(values=field, ignore_value=0.0, weight_by_value=False))
        .render()
    )
    assert np.array_equal(out[:, :18], base[:, :18])  # sentinel region intact
    assert not np.array_equal(out[:, 22:], base[:, 22:])  # data region painted


def test_heatmap_all_nodata_field_draws_nothing() -> None:
    field = np.full((30, 40), np.nan)
    assert np.array_equal(
        _canvas().add(Heatmap(values=field)).render(), _canvas().render()
    )


def test_heatmap_empty_field_draws_nothing() -> None:
    assert np.array_equal(
        _canvas().add(Heatmap(values=np.zeros(0))).render(), _canvas().render()
    )


def test_heatmap_without_ignore_value_is_unchanged() -> None:
    # Backwards-compatibility guard: the default path must not shift a pixel.
    field = np.linspace(0.0, 1.0, 30 * 40).reshape(30, 40)
    assert np.array_equal(
        _canvas().add(Heatmap(values=field)).render(),
        _canvas().add(Heatmap(values=field, ignore_value=None)).render(),
    )


def test_heatmap_value_range_reports_explicit_bounds() -> None:
    field = np.array([[0.0, 5.0, 10.0]])
    assert Heatmap(values=field, vmin=-1.0, vmax=4.0).value_range() == (
        -1.0,
        4.0,
    )
    assert Heatmap(values=field, vmax=4.0).value_range() == (0.0, 4.0)
    assert Heatmap(values=field, normalize=False).value_range() == (0.0, 1.0)


# --- ColorBar ---------------------------------------------------------------


def test_colorbar_for_heatmap_inherits_gradient_and_range() -> None:
    # The key must never be an approximation of the field it describes.
    heat = Heatmap(values=np.array([[0.0, 5.0, 10.0]]), gradient="magma")
    key = ColorBar.for_heatmap(heat, title="depth")
    assert key.gradient == heat.gradient
    assert (key.vmin, key.vmax) == heat.value_range()
    assert key.title == "depth"


def test_colorbar_for_heatmap_carries_an_unset_gradient_through() -> None:
    # Both None means both inherit the context gradient, so they cannot drift.
    key = ColorBar.for_heatmap(Heatmap(values=np.ones((2, 2))))
    assert key.gradient is None


def test_colorbar_for_heatmap_excludes_nodata_from_its_range() -> None:
    heat = Heatmap(values=np.array([[0.0, 5.0, 10.0]]), ignore_value=0.0)
    assert ColorBar.for_heatmap(heat).vmin == 5.0


def test_colorbar_renders_onto_the_image() -> None:
    base = _canvas(120, 90).render()
    out = _canvas(120, 90).add(ColorBar(vmin=0.0, vmax=291.0)).render()
    assert out.shape == base.shape
    assert not np.array_equal(out, base)


def test_colorbar_extent_is_none() -> None:
    assert ColorBar().extent() is None


def test_colorbar_tick_values_span_the_range() -> None:
    assert ColorBar(vmin=0.0, vmax=10.0)._tick_values() == [0.0, 10.0]
    assert ColorBar(vmin=0.0, vmax=10.0, ticks=3)._tick_values() == [
        0.0,
        5.0,
        10.0,
    ]


def test_colorbar_follows_the_gradient_it_is_given() -> None:
    a = _canvas(120, 90).add(ColorBar(gradient="viridis")).render()
    b = _canvas(120, 90).add(ColorBar(gradient="magma")).render()
    assert not np.array_equal(a, b)


def test_colorbar_stacks_with_a_legend_in_the_same_corner() -> None:
    # CornerStack reserves its rectangle, so two overlays anchored to one
    # corner offset rather than overlap.
    from luxonis_ml.vizlab import Corner, Legend

    legend = Legend(entries=["car"], corner=Corner.BOTTOM_RIGHT)
    key = ColorBar(vmin=0.0, vmax=1.0, corner=Corner.BOTTOM_RIGHT)
    both = _canvas(200, 150).add(legend).add(key).render()
    only = _canvas(200, 150).add(legend).render()
    assert not np.array_equal(both, only)
