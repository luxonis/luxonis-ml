"""Color gradients (colormaps) for scalar fields such as heatmaps.

The `Gradient` model is shared color logic and lives in
`luxonis_ml.utils.color.gradient`; vizlab re-exports it here so that
``from luxonis_ml.vizlab.gradient import ...`` keeps working and vizlab never
reimplements color handling.
"""

from luxonis_ml.utils.color.gradient import (
    DEFAULT_GRADIENT,
    GRADIENTS,
    Gradient,
    resolve_gradient,
)

__all__ = [
    "DEFAULT_GRADIENT",
    "GRADIENTS",
    "Gradient",
    "resolve_gradient",
]
