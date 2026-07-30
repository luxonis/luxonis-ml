"""Themes: the default style, palette, and background used when nothing overrides.

A `Theme` bundles the three global defaults a render falls back to — the
`Style` for un-styled annotations, the `Palette` that assigns class
colors, and the background used by compositing. The active theme is threaded
through the render (see `RenderContext`), so an
annotation's own ``style``/``color``/``palette`` still win; the theme only fills
the gaps.

Two presets ship: :data:`DARK_THEME` (the default) and :data:`LIGHT_THEME`. A
theme is carried by a `luxonis_ml.vizlab.RenderOptions`: install one for a scope
with `luxonis_ml.vizlab.default_options` / `luxonis_ml.vizlab.set_default_options`,
or pass ``options=`` to a single `Image`. Build variants with `Theme.with_style`,
`Theme.with_palette`, and `Theme.with_class_colors`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.color import Color, ColorLike

from .palette import DEFAULT_PALETTE, GoldenRatioColors, Palette
from .style import DEFAULT_STYLE, Style

# On-brand composite backgrounds: a deep navy in the Luxonis "ink" family for
# dark mode, and the brand light-gray surface for light mode.
_DARK_BG = brand.BACKGROUND
_LIGHT_BG = brand.LIGHT_BACKGROUND


@dataclass(frozen=True)
class Theme:
    """The default style, palette, and background for a render.

    .. image:: TODO-HOST/themes.png
       :alt: The same scene rendered in the dark and light themes.

    Attributes:
        style: Default style for annotations that set none.
        palette: Palette assigning class colors. Shared, so a class keeps its color
            across every image drawn with this theme.
        background: Background color used by compositing (stacks, grids, padding).

    Examples:
        >>> DARK_THEME.style.fill_alpha
        0.16
        >>> LIGHT_THEME.background.r
        237

    """

    style: Style = DEFAULT_STYLE
    palette: Palette = field(default=DEFAULT_PALETTE)
    background: Color = _DARK_BG

    def with_style(self, style: Style) -> "Theme":
        """Return a copy of this theme using ``style`` as its default style."""
        return replace(self, style=style)

    def with_palette(self, palette: Palette) -> "Theme":
        """Return a copy of this theme using ``palette`` for class colors."""
        return replace(self, palette=palette)

    def with_class_colors(self, colors: Mapping[str, ColorLike]) -> "Theme":
        """Return a copy whose palette pins the given ``{class: color}`` map.

        Sugar for ``theme.with_palette(theme.palette.with_colors(colors))``: the
        listed classes get exactly those colors, every other class keeps the
        palette's generated color. The original theme and palette are untouched.

        Args:
            colors: ``{class_name: color}`` pins (any :data:`ColorLike`).

        Returns:
            A new `Theme` with the pinned palette.

        """
        return replace(self, palette=self.palette.with_colors(colors))


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
