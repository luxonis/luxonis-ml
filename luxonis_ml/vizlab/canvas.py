"""Compatibility façade for the rendering canvas implementation."""

from .render.canvas import Canvas, Shadow, TextMetrics, gaussian_blur

__all__ = ["Canvas", "Shadow", "TextMetrics", "gaussian_blur"]
