"""Interactive display for vizlab images.

Import this subpackage explicitly (``from luxonis_ml.vizlab.viewer import
Viewer``) — it is deliberately *not* re-exported from ``luxonis_ml.vizlab`` so the
core rendering package stays free of any windowing/OpenCV import until a viewer is
actually used.
"""

from .backend import WindowBackend
from .cv2_backend import Cv2Backend
from .tooltip_render import draw_tooltip, render_tooltip_card
from .viewer import Viewer

__all__ = [
    "Cv2Backend",
    "Viewer",
    "WindowBackend",
    "draw_tooltip",
    "render_tooltip_card",
]
