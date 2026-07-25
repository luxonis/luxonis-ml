"""Styling: resolved per-annotation `Style` and class-color `Palette`."""

from .derive import derive_child_color, derive_child_style
from .palette import (
    DEFAULT_PALETTE,
    ColorGenerator,
    GoldenRatioColors,
    Palette,
)
from .style import DEFAULT_STYLE, LabelPlacement, Style
from .theme import (
    DARK_THEME,
    LIGHT_THEME,
    Theme,
    get_default_theme,
    set_default_theme,
)

__all__ = [
    "DARK_THEME",
    "DEFAULT_PALETTE",
    "DEFAULT_STYLE",
    "LIGHT_THEME",
    "ColorGenerator",
    "GoldenRatioColors",
    "LabelPlacement",
    "Palette",
    "Style",
    "Theme",
    "derive_child_color",
    "derive_child_style",
    "get_default_theme",
    "set_default_theme",
]
