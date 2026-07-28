"""Compatibility façade for hover and click interaction maps."""

from .interaction import maps as _impl
from .interaction.maps import ClickMap, HitMap

InteractionCapture = _impl.InteractionCapture

__all__ = ["ClickMap", "HitMap"]
