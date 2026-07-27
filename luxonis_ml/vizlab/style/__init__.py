"""Configure annotation geometry, class colors, and render-wide defaults.

`Style` controls strokes, fills, typography, labels, keypoints, and masks.
`Palette` assigns stable colors in first-seen class order; pre-register class
names when the mapping must remain stable across images. `Theme` combines a
style, palette, and composition background. An annotation-level override wins
over its parent, then the active theme, then the library default.
"""

from .derive import derive_child_color, derive_child_style
from .palette import (
    BRAND_COLORS,
    DEFAULT_PALETTE,
    ColorGenerator,
    GoldenRatioColors,
    Palette,
    SequenceColors,
)
from .style import (
    DEFAULT_STYLE,
    LabelPlacement,
    Style,
    current_default_style,
    current_style_overrides,
)
from .theme import (
    DARK_THEME,
    LIGHT_THEME,
    Theme,
    get_default_theme,
    set_default_theme,
)

__all__ = [
    "BRAND_COLORS",
    "DARK_THEME",
    "DEFAULT_PALETTE",
    "DEFAULT_STYLE",
    "LIGHT_THEME",
    "ColorGenerator",
    "GoldenRatioColors",
    "LabelPlacement",
    "Palette",
    "SequenceColors",
    "Style",
    "Theme",
    "current_default_style",
    "current_style_overrides",
    "derive_child_color",
    "derive_child_style",
    "get_default_theme",
    "set_default_theme",
]
