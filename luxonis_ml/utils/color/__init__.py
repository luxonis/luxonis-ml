"""Color management for ``luxonis-ml``: the primitive, brand, and class palette.

This package is the single home for color handling across the stack, so no other
module reimplements it:

- `Color` — the immutable RGBA primitive with hex/name/HSL parsing and helpers.
- `brand` — the Luxonis brand colors and the UI-chrome palette built from them,
  for every non-label color (backgrounds, cards, dividers, verdict marks).
- `Palette` / `GoldenRatioColors` / `SequenceColors` — distinct per-class label
  colors from a pluggable index-based generator, plus the colorblind-safe named
  sets in :data:`PALETTES` and the `CVDDistinctColors` generator behind them.
- `cvd` — simulating color-vision deficiency and measuring perceptual distance,
  which is how the palettes above earn the word "safe".
"""

from . import brand, cvd
from .base import RGB, RGBA, Color, ColorLike
from .palette import (
    BRAND_COLORS,
    CVD_PALETTE,
    DEFAULT_PALETTE,
    PALETTES,
    ColorGenerator,
    ColormapColors,
    CVDDistinctColors,
    GoldenRatioColors,
    Palette,
    SequenceColors,
    resolve_generator,
)

__all__ = [
    "BRAND_COLORS",
    "CVD_PALETTE",
    "DEFAULT_PALETTE",
    "PALETTES",
    "RGB",
    "RGBA",
    "CVDDistinctColors",
    "Color",
    "ColorGenerator",
    "ColorLike",
    "ColormapColors",
    "GoldenRatioColors",
    "Palette",
    "SequenceColors",
    "brand",
    "cvd",
    "resolve_generator",
]
