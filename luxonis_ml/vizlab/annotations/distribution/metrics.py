"""Shared metrics and colours for the distribution chart modes.

Padding, gaps and the semantic fills the painters and the annotation's cell
layout both need, so neither has to import the other.
"""

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.style import Style

_PAD = 10.0

_ROW_GAP = 6.0

_COL_GAP = 8.0

_PIE_KEY_GAP = 16.0

_WHITE = Color(255, 255, 255)

_OK = brand.SUCCESS

_BAD = brand.ERROR

_OTHER = brand.MUTED


def _clamp01(value: float) -> float:
    """Clamp a probability into ``[0, 1]``."""
    return max(0.0, min(1.0, value))


def _pct(prob: float) -> str:
    """Format a probability as a whole-percent string, e.g. ``"92%"``."""
    return f"{round(_clamp01(prob) * 100)}%"


def _edge_w(style: Style) -> float:
    """Hairline width for chart-element outlines, tracking the font size."""
    return max(1.0, style.font_size * 0.09)
