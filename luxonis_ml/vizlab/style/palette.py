"""Distinct class colors from a pluggable, index-based generator.

`Palette` and its color generators are shared color logic, defined in
`luxonis_ml.utils.color.palette` and re-exported here under the name the vizlab
style package uses internally.
"""

from luxonis_ml.utils.color.palette import (
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
    "CVDDistinctColors",
    "ColorGenerator",
    "ColormapColors",
    "GoldenRatioColors",
    "Palette",
    "SequenceColors",
    "resolve_generator",
]
