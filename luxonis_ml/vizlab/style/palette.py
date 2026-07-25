"""Distinct class colors from a pluggable, index-based generator.

The `Palette` and its color generators are shared color logic and live in
`luxonis_ml.utils.color.palette`; the vizlab style package re-exports them here so
that ``from luxonis_ml.vizlab.style.palette import ...`` keeps working and vizlab
never reimplements color handling.
"""

from luxonis_ml.utils.color.palette import (
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
    "ColorGenerator",
    "GoldenRatioColors",
    "Palette",
    "SequenceColors",
]
