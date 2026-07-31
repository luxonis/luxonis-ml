"""Color gradients (colormaps) for scalar fields such as heatmaps.

`Gradient` is shared color logic, defined in `luxonis_ml.utils.color.gradient`
and re-exported here under the name vizlab uses internally.
"""

from luxonis_ml.utils.color.gradient import (
    DEFAULT_DIVERGING_GRADIENT,
    DEFAULT_GRADIENT,
    DIVERGING_GRADIENTS,
    GRADIENTS,
    Gradient,
    resolve_gradient,
)

__all__ = [
    "DEFAULT_DIVERGING_GRADIENT",
    "DEFAULT_GRADIENT",
    "DIVERGING_GRADIENTS",
    "GRADIENTS",
    "Gradient",
    "resolve_gradient",
]
