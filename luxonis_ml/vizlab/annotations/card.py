"""Shared drawing primitives for annotation cards."""

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render.canvas import Canvas, Shadow
from luxonis_ml.vizlab.style import Style

#: How opaque an annotation's card is over the image it labels. The shared
#: card fill is translucent because the *tooltip* card is drawn over a blurred
#: backdrop, where translucency is the whole effect. An annotation card has no
#: blur under it — it sits straight on the frame — so at that alpha a busy
#: scene reads through the text. It gets its own, nearly opaque, value.
_OVERLAY_ALPHA = 0.94


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
        fill=chrome.card_bg.with_alpha(_OVERLAY_ALPHA),
        stroke=chrome.border,
        stroke_width=1.0 if chrome.border is not None else 0.0,
        shadow=Shadow(blur=6.0, dy=2.0) if style.shadow else None,
    )
