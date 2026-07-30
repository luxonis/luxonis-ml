"""Render the interactive controls HUD as a small card.

Pure vizlab `Canvas`/NumPy — no windowing — so any backend can composite it. The
card matches the tooltip's brand look (rounded, translucent, soft shadow) and lays
the controls out as a little table: a keycap, the layer name, and its current
value, with the value tinted by state (mint when on, muted when off). The type
size is passed in by the viewer so the HUD scales with the displayed frame.
"""

import numpy as np

from luxonis_ml.vizlab.render import text_layout
from luxonis_ml.vizlab.render.canvas import Canvas

from .layers import Control


def render_controls_card(controls: list[Control], size: int) -> np.ndarray:
    """Render ``controls`` as an RGBA HUD card at body type ``size`` (pixels).

    Args:
        controls: The rows to show (see `Control`).
        size: Body type size in pixels; keycaps, paddings, and the title scale
            from it, so passing a size derived from the frame keeps the HUD
            proportional to the image.

    Returns:
        A ``(H, W, 4)`` ``uint8`` RGBA array (with a transparent margin for the
        drop shadow).

    """
    from luxonis_ml.utils.color import brand
    from luxonis_ml.vizlab.color import Color
    from luxonis_ml.vizlab.geometry import Rect
    from luxonis_ml.vizlab.render.canvas import Shadow

    name_color = Color(206, 214, 230)
    value_on = brand.MINT
    value_off = Color(120, 131, 150)
    keycap_fill = brand.PERIWINKLE.with_alpha(0.14)
    keycap_stroke = brand.PERIWINKLE.with_alpha(0.34)
    keycap_text = brand.CARD_TITLE
    title_color = brand.CARD_TITLE.with_alpha(0.72)

    title = "CONTROLS"
    title_size = size * 0.82
    row = text_layout.line_metrics(size, weight=600)
    row_h = round(row.height * 1.42)
    keycap_h = round(row.height * 1.12)
    kc_pad = round(size * 0.5)
    col_gap = round(size * 0.7)
    pad = round(size * 1.0)
    mg = round(size * 0.6)  # transparent margin for the shadow

    key_w = [
        text_layout.measure(c.key, size * 0.92, weight=700, mono=True).width
        for c in controls
    ]
    name_w = [
        text_layout.measure(c.name, size, weight=600).width for c in controls
    ]
    val_w = [
        text_layout.measure(c.value, size, weight=600, mono=True).width
        for c in controls
    ]
    keycap_w = round(max(key_w) + 2 * kc_pad)
    names_w = max(name_w)
    values_w = max(val_w)
    title_m = text_layout.measure(title, title_size, weight=700)

    content_w = max(
        keycap_w + col_gap + names_w + col_gap + values_w, title_m.width
    )
    title_h = title_m.height + round(size * 0.55)
    card_w = round(content_w + 2 * pad)
    card_h = round(2 * pad + title_h + len(controls) * row_h)

    canvas = Canvas.blank(card_w + 2 * mg, card_h + 2 * mg)
    canvas.rounded_rect(
        Rect(mg, mg, mg + card_w, mg + card_h),
        radius=round(size * 0.6),
        fill=brand.CARD_BG,
        shadow=Shadow(blur=size * 0.55, dy=size * 0.16),
    )

    left = mg + pad
    right = left + content_w
    canvas.text(
        (float(left), mg + pad + title_m.ascent),
        title,
        size=title_size,
        color=title_color,
        weight=700,
    )
    y = mg + pad + title_h
    for i, control in enumerate(controls):
        mid = y + i * row_h + row_h / 2
        cap = Rect(
            left, mid - keycap_h / 2, left + keycap_w, mid + keycap_h / 2
        )
        canvas.rounded_rect(
            cap,
            radius=round(size * 0.3),
            fill=keycap_fill,
            stroke=keycap_stroke,
            stroke_width=max(1.0, size * 0.07),
        )
        baseline = mid + (row.ascent - row.descent) / 2
        canvas.text(
            (left + (keycap_w - key_w[i]) / 2, baseline),
            control.key,
            size=size * 0.92,
            color=keycap_text,
            weight=700,
            mono=True,
        )
        canvas.text(
            (left + keycap_w + col_gap, baseline),
            control.name,
            size=size,
            color=name_color,
            weight=600,
        )
        tint = (
            value_on
            if control.active
            else value_off
            if control.active is False
            else name_color
        )
        canvas.text(
            (right - val_w[i], baseline),
            control.value,
            size=size,
            color=tint,
            weight=600,
            mono=True,
        )
    return canvas.to_rgba()
