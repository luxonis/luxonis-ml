"""Compatibility façade for retained Vizlab scenes."""

from .scene import image as _impl
from .scene.image import Composite, Image, Renderable

_style_scale = _impl._style_scale

__all__ = ["Composite", "Image", "Renderable"]
