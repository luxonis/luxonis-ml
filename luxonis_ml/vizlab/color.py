"""Color parsing and manipulation for vizlab.

The `Color` primitive is shared across ``luxonis-ml`` and lives in
`luxonis_ml.utils.color`; vizlab re-exports it here so that every
``from .color import Color`` inside the package keeps working and vizlab never
reimplements color handling.
"""

from luxonis_ml.utils.color import RGBA, Color, ColorLike

__all__ = ["RGBA", "Color", "ColorLike"]
