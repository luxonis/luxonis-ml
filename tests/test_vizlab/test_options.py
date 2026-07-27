"""Tests for RenderOptions, scoped defaults, Style scopes, and palette pins."""

import numpy as np

from luxonis_ml.vizlab import (
    DARK_THEME,
    LIGHT_THEME,
    Color,
    Heatmap,
    Image,
    Palette,
    RenderOptions,
    Style,
    current_options,
    default_options,
    set_default_options,
)
from luxonis_ml.vizlab.style import (
    current_default_style,
    current_style_overrides,
)


def _field() -> np.ndarray:
    return np.linspace(0.0, 1.0, 6 * 8).reshape(6, 8)


def test_render_options_defaults_and_replace() -> None:
    opts = RenderOptions()
    assert opts.theme is DARK_THEME
    assert opts.hover_metadata is False
    assert opts.replace(hover_metadata=True).hover_metadata is True
    assert opts.hover_metadata is False  # frozen: original unchanged


def test_default_options_scope_restores_on_exit() -> None:
    before = current_options()
    with default_options(RenderOptions(hover_metadata=True)) as scoped:
        assert current_options() is scoped
        assert current_options().hover_metadata is True
    assert current_options() is before


def test_default_options_nesting() -> None:
    with default_options(RenderOptions(draw_skeletons=True)):
        with default_options(RenderOptions(hover_metadata=True)):
            assert current_options().hover_metadata is True
            assert current_options().draw_skeletons is False
        assert current_options().draw_skeletons is True


def test_set_default_options_persists_without_scope() -> None:
    original = current_options()
    try:
        set_default_options(RenderOptions(hover_metadata=True))
        assert current_options().hover_metadata is True
    finally:
        set_default_options(original)


def test_style_as_default_scope() -> None:
    bold = Style(stroke_width=9.0)
    assert current_default_style() is None
    with bold.as_default():
        assert current_default_style() is bold
    assert current_default_style() is None


def test_style_override_scope_merges_and_nests() -> None:
    assert current_style_overrides() == {}
    with Style.override(stroke_width=6.0, shadow=False):
        assert current_style_overrides() == {
            "stroke_width": 6.0,
            "shadow": False,
        }
        with Style.override(fill_alpha=0.5, shadow=True):
            merged = current_style_overrides()
            assert merged["stroke_width"] == 6.0  # inherited from outer
            assert merged["fill_alpha"] == 0.5  # added
            assert merged["shadow"] is True  # inner wins
    assert current_style_overrides() == {}


def test_palette_pins_exact_color_without_consuming_a_slot() -> None:
    palette = Palette(colors={"car": "#ff0000"})
    assert palette.color_for("car") == Color.parse("#ff0000")
    # A pin does not shift the generator: the first *unpinned* class is index 0.
    assert palette.color_for("bus") == Palette().color_for("bus")


def test_palette_pin_is_fluent() -> None:
    palette = Palette().pin("bike", Color(136, 255, 0))
    assert palette.color_for("bike") == Color(136, 255, 0)


def test_palette_with_colors_is_a_copy() -> None:
    base = Palette()
    base.color_for("car")  # assign a generated color
    derived = base.with_colors({"car": "#00ff00"})
    assert derived.color_for("car") == Color.parse("#00ff00")
    assert base.color_for("car") != Color.parse("#00ff00")  # original intact


def test_theme_with_class_colors_pins_via_palette() -> None:
    theme = DARK_THEME.with_class_colors({"car": "#123456"})
    assert theme.palette.color_for("car") == Color.parse("#123456")
    # DARK_THEME's shared palette is untouched.
    assert DARK_THEME.palette.color_for("car") != Color.parse("#123456")


def test_theme_with_style_and_palette() -> None:
    style = Style(stroke_width=1.0)
    palette = Palette(colors={"x": "#010203"})
    theme = DARK_THEME.with_style(style).with_palette(palette)
    assert theme.style is style
    assert theme.palette is palette
    assert theme.background == DARK_THEME.background


def test_image_renders_with_scoped_options_theme() -> None:
    image = Image(np.zeros((20, 30, 3), np.uint8))
    assert image.theme is DARK_THEME  # scope default outside any block
    with default_options(RenderOptions(theme=LIGHT_THEME)):
        assert image.theme is LIGHT_THEME  # resolved at access time
    assert image.theme is DARK_THEME


def test_explicit_image_options_win_over_scope() -> None:
    image = Image(
        np.zeros((20, 30, 3), np.uint8),
        options=RenderOptions(theme=LIGHT_THEME),
    )
    with default_options(RenderOptions(theme=DARK_THEME)):
        assert (
            image.theme is LIGHT_THEME
        )  # explicit options are not overridden


def test_heatmap_inherits_options_gradient() -> None:
    def render_with(gradient: str) -> np.ndarray:
        image = Image(np.zeros((6, 8, 3), np.uint8)).add(
            Heatmap(values=_field())
        )
        with default_options(RenderOptions(gradient=gradient)):
            return image.render()

    assert not np.array_equal(render_with("viridis"), render_with("magma"))


def test_explicit_heatmap_gradient_beats_options() -> None:
    image = Image(np.zeros((6, 8, 3), np.uint8)).add(
        Heatmap(values=_field(), gradient="viridis")
    )
    with default_options(RenderOptions(gradient="viridis")):
        pinned = image.render()
    with default_options(RenderOptions(gradient="magma")):
        under_magma = image.render()
    assert np.array_equal(pinned, under_magma)
