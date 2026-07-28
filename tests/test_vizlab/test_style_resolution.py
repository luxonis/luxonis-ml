"""Tests for layered style resolution (`Annotation.resolve_style`)."""

from luxonis_ml.vizlab import LIGHT_THEME, BBox, Style
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.canvas import Canvas
from luxonis_ml.vizlab.render import RenderEnvironment


def _ctx(style_scale: float = 1.0) -> RenderContext:
    # LIGHT_THEME.style sets fill_alpha=0.20 (vs Style()'s 0.16), so it is a
    # clear witness for "did the theme survive the override?".
    return RenderContext(
        canvas=Canvas.blank(2, 2), theme=LIGHT_THEME, style_scale=style_scale
    )


def test_styled_patch_keeps_the_rest_of_the_theme() -> None:
    style = (
        BBox(x=0, y=0, w=1, h=1).styled(stroke_width=5.0).resolve_style(_ctx())
    )
    assert style.stroke_width == 5.0  # overridden
    assert (
        style.fill_alpha == 0.20
    )  # theme value preserved (not Style()'s 0.16)


def test_full_style_replaces_the_theme() -> None:
    style = (
        BBox(x=0, y=0, w=1, h=1)
        .styled(Style(stroke_width=5.0))
        .resolve_style(_ctx())
    )
    assert style.stroke_width == 5.0
    assert style.fill_alpha == 0.16  # Style() default: theme not consulted


def test_styled_full_style_with_keyword_tweak() -> None:
    # A full Style base plus a keyword override applied on top.
    style = (
        BBox(x=0, y=0, w=1, h=1)
        .styled(Style(stroke_width=5.0), corner_radius=1.0)
        .resolve_style(_ctx())
    )
    assert style.stroke_width == 5.0  # from the full Style base
    assert style.corner_radius == 1.0  # keyword tweak layered on top


def test_styled_mapping_positional_layers() -> None:
    faded = {"label_alpha": 0.4, "fill_alpha": 0.05}
    box = BBox(x=0, y=0, w=1, h=1).styled(faded, stroke_width=2.0)
    assert box.style_overrides == {**faded, "stroke_width": 2.0}
    assert box.style is None  # a mapping layers; it does not replace


def test_override_scope_layers_on_a_full_style() -> None:
    # A full-Style base still picks up an enclosing Style.override scope.
    with Style.override(shadow=False):
        style = (
            BBox(x=0, y=0, w=1, h=1)
            .styled(Style(stroke_width=5.0))
            .resolve_style(_ctx())
        )
    assert style.stroke_width == 5.0  # from the full Style
    assert style.shadow is False  # scope layered on top


def test_style_override_scope_layers_over_theme() -> None:
    box = BBox(x=0, y=0, w=1, h=1)
    with Style.override(stroke_width=7.0):
        style = box.resolve_style(_ctx())
    assert style.stroke_width == 7.0
    assert style.fill_alpha == 0.20  # theme survives the scoped override


def test_per_annotation_override_beats_scope() -> None:
    with Style.override(stroke_width=7.0):
        style = (
            BBox(x=0, y=0, w=1, h=1)
            .styled(stroke_width=9.0)
            .resolve_style(_ctx())
        )
    assert style.stroke_width == 9.0


def test_as_default_replaces_the_base_style_in_scope() -> None:
    with Style(stroke_width=1.0, fill_alpha=0.9).as_default():
        style = BBox(x=0, y=0, w=1, h=1).resolve_style(_ctx())
    assert style.stroke_width == 1.0
    assert style.fill_alpha == 0.9


def test_render_environment_is_an_ambient_style_snapshot() -> None:
    with Style.override(stroke_width=7.0):
        environment = RenderEnvironment.current()

    ctx = RenderContext(
        canvas=Canvas.blank(2, 2),
        theme=LIGHT_THEME,
        environment=environment,
    )
    with Style.override(stroke_width=11.0):
        style = BBox(x=0, y=0, w=1, h=1).resolve_style(ctx)

    assert style.stroke_width == 7.0


def test_theme_base_is_scaled_but_overrides_are_display_pixels() -> None:
    # Theme stroke 3.0 scales to 6.0 at 2x; a styled override is taken as-is.
    scaled = BBox(x=0, y=0, w=1, h=1).resolve_style(_ctx(style_scale=2.0))
    assert scaled.stroke_width == 6.0
    override = (
        BBox(x=0, y=0, w=1, h=1)
        .styled(stroke_width=3.0)
        .resolve_style(_ctx(style_scale=2.0))
    )
    assert override.stroke_width == 3.0  # display pixels, not re-scaled
