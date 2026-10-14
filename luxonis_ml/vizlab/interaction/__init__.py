"""The `Frame`: a drawn scene paired with the regions captured alongside it.

This sits above `luxonis_ml.vizlab.scene` — it is what a `Viewer` consumes. The
collectors themselves live in `luxonis_ml.vizlab.render`, because they are
filled during the render, before any of this exists.
"""

from .frame import Frame

__all__ = ["Frame"]
