"""Interactive display for vizlab images.

Import this subpackage explicitly (``from luxonis_ml.vizlab.viewer import
Viewer``) — it is deliberately *not* re-exported from ``luxonis_ml.vizlab`` so the
core rendering package stays free of any windowing/OpenCV import until a viewer is
actually used.
"""

from . import clipboard
from .backend import WindowBackend
from .cv2_backend import Cv2Backend
from .layers import LayerState
from .notebook_backend import NotebookBackend
from .prefetch import PrefetchIterator
from .static import show_fitted
from .tooltip_render import (
    TooltipCard,
    draw_tooltip,
    prepare_tooltip,
    render_tooltip_card,
)
from .viewer import PreparedFrame, Viewer, ViewerSample, report_pick

__all__ = [
    "Cv2Backend",
    "LayerState",
    "NotebookBackend",
    "PrefetchIterator",
    "PreparedFrame",
    "TooltipCard",
    "Viewer",
    "ViewerSample",
    "WindowBackend",
    "clipboard",
    "draw_tooltip",
    "prepare_tooltip",
    "render_tooltip_card",
    "report_pick",
    "show_fitted",
]
