"""Shared drawing primitives for annotation cards."""

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render.canvas import Canvas, Shadow
from luxonis_ml.vizlab.style import Style


def draw_card_background(
    canvas: Canvas,
    rect: Rect,
    style: Style,
    chrome: brand.Chrome,
) -> None:
    """Paint the standard theme-aware annotation card surface."""
    canvas.rounded_rect(
        rect,
        radius=9.0,
        fill=chrome.card_bg,
        stroke=chrome.border,
        stroke_width=1.0 if chrome.border is not None else 0.0,
        shadow=Shadow(blur=6.0, dy=2.0) if style.shadow else None,
    )
