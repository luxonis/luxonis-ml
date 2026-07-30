"""Render-time state: the drawing surface, the style snapshot, the collectors.

Everything here is filled or consulted *while* a scene draws itself, so nothing
in this package knows what a scene, a layout or a viewer is.
"""

from . import text_layout
from .capture import ClickMap, HitMap, InteractionCapture
from .context import RenderEnvironment

__all__ = [
    "ClickMap",
    "HitMap",
    "InteractionCapture",
    "RenderEnvironment",
    "text_layout",
]
