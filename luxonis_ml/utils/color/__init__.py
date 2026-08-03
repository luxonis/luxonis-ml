"""Color management for ``luxonis-ml``: the primitive, brand, and class palette.

This package is the single home for color handling across the stack, so no other
module reimplements it:

- `Color` — the immutable RGBA primitive with hex/name/HSL parsing and helpers.
- `brand` — the Luxonis brand colors and the UI-chrome palette built from them,
  for every non-label color (backgrounds, cards, dividers, verdict marks).
- `Palette` / `GoldenRatioColors` / `SequenceColors` — distinct per-class label
  colors from a pluggable index-based generator.
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
