"""Themes: the default style, palette, and background used when nothing overrides.

A `Theme` bundles the three global defaults a render falls back to — the
`Style` for un-styled annotations, the `Palette` that assigns class
colors, and the background used by compositing. The active theme is threaded
through the render (see `RenderContext`), so an
annotation's own ``style``/``color``/``palette`` still win; the theme only fills
the gaps.

Two presets ship: :data:`DARK_THEME` (the default) and :data:`LIGHT_THEME`. Set a
process-wide default with `set_default_theme`, or pass ``theme=`` to a single
`Image`.
"""

from dataclasses import dataclass, field

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.color import Color

from .palette import DEFAULT_PALETTE, GoldenRatioColors, Palette
from .style import DEFAULT_STYLE, Style

# On-brand composite backgrounds: a deep navy in the Luxonis "ink" family for
# dark mode, and the brand light-gray surface for light mode.
_DARK_BG = brand.BACKGROUND
_LIGHT_BG = brand.LIGHT_BACKGROUND


@dataclass(frozen=True)
class Theme:
    """The default style, palette, and background for a render.

    Attributes:
        style: Default style for annotations that set none.
        palette: Palette assigning class colors. Shared, so a class keeps its color
            across every image drawn with this theme.
        background: Background color used by compositing (stacks, grids, padding).

    Examples:
        >>> DARK_THEME.style.fill_alpha
        0.16
        >>> LIGHT_THEME.background.r
        242

    """

    style: Style = DEFAULT_STYLE
    palette: Palette = field(default=DEFAULT_PALETTE)
    background: Color = _DARK_BG


DARK_THEME = Theme()
"""The default theme: soft shadows, translucent fills, a dark composite background."""

LIGHT_THEME = Theme(
    style=Style(fill_alpha=0.20),
    # Slightly darker, punchier colors that hold up on light photos/backgrounds.
    palette=Palette(
        generator=GoldenRatioColors(lightness=0.5, saturation=0.72)
    ),
    background=_LIGHT_BG,
)
"""A light-background counterpart to :data:`DARK_THEME`."""

_default_theme = DARK_THEME


def get_default_theme() -> Theme:
    """Return the process-wide default theme.

    Returns:
        The current default `Theme`.

    """
    return _default_theme


def set_default_theme(theme: Theme) -> None:
    """Set the process-wide default theme used by images without an explicit one.

    Args:
        theme: The theme to make the default.

    """
    global _default_theme  # noqa: PLW0603
    _default_theme = theme
