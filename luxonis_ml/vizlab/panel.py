"""Compatibility façade for metadata panel layout."""

from .layout import panel as _impl
from .layout.panel import (
    Block,
    Controls,
    PanelData,
    Swatches,
    with_panel,
)

_MARGIN = _impl._MARGIN
_PAD = _impl._PAD
_PANEL_SIZE = _impl._PANEL_SIZE
_auto_width = _impl._auto_width
_build_ops = _impl._build_ops
_compose_panel = _impl._compose_panel
_format_sections = _impl._format_sections
_format_tree = _impl._format_tree
_layout_swatches = _impl._layout_swatches
_metrics = _impl._metrics
_middle_ellipsize = _impl._middle_ellipsize
_wrap = _impl._wrap

__all__ = [
    "Block",
    "Controls",
    "PanelData",
    "Swatches",
    "with_panel",
]
