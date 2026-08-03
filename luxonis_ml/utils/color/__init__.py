"""Color management for ``luxonis-ml``: the primitive, brand, and class palette.

This package is the single home for color handling across the stack, so no other
module reimplements it:

- `Color` — the immutable RGBA primitive with hex/name/HSL parsing and helpers.
- `brand` — the Luxonis brand colors and the UI-chrome palette built from them,
  for every non-label color (backgrounds, cards, dividers, verdict marks).
- `Palette` / `GoldenRatioColors` / `SequenceColors` — distinct per-class label
  colors from a pluggable index-based generator.

None of the above needs NumPy, so importing this package stays cheap for a base
``luxonis-ml[utils]`` install. `Gradient` (colormaps for scalar fields such as
heatmaps) lives in `luxonis_ml.utils.color.gradient` and is imported directly
from there rather than re-exported here; it also imports without NumPy, and only
its `Gradient.colorize` method — which takes a NumPy array in the first place —
needs it. NumPy is therefore never a dependency of importing any of this.
"""

from . import brand
from .base import RGB, RGBA, Color, ColorLike
from .palette import (
    BRAND_COLORS,
    DEFAULT_PALETTE,
    ColorGenerator,
    GoldenRatioColors,
    Palette,
    SequenceColors,
)

__all__ = [
    "BRAND_COLORS",
    "DEFAULT_PALETTE",
    "RGB",
    "RGBA",
    "Color",
    "ColorGenerator",
    "ColorLike",
    "GoldenRatioColors",
    "Palette",
    "SequenceColors",
    "brand",
]
