"""Render a `Tooltip` as a native vizlab card and composite it onto a frame.

These helpers are pure vizlab `Canvas`/NumPy — no windowing or OpenCV — so they
are shared by any backend. The card matches the rest of the UI: a rounded,
translucent brand card with a soft shadow, a class-tinted title, periwinkle Inter
keys, and near-white JetBrains Mono values.
"""

import functools

import numpy as np

from luxonis_ml.vizlab.canvas import Canvas
from luxonis_ml.vizlab.tooltip import Tooltip


@functools.lru_cache(maxsize=1)
def _measure_canvas() -> Canvas:
    """Return a cached tiny canvas used only to measure tooltip text."""
    return Canvas.blank(2, 2)


def render_tooltip_card(tooltip: Tooltip, size: int) -> np.ndarray:
    """Render ``tooltip`` as an RGBA card at type ``size`` (in pixels).

    Args:
        tooltip: The hover content to draw.
        size: Body type size in pixels; the title and paddings scale from it.

    Returns:
        A ``(H, W, 4)`` ``uint8`` RGBA array of the card (with a transparent
        margin so its drop shadow has room).

    """
    from luxonis_ml.utils.color import brand
    from luxonis_ml.vizlab.canvas import Shadow
    from luxonis_ml.vizlab.geometry import Rect

    measure = _measure_canvas()
    title = tooltip.title
    title_color = (
        tooltip.tint if tooltip.tint is not None else brand.CARD_TITLE
    )
    pairs = [(f"{key}: ", value) for key, value in tooltip.rows]

    pad, gap = round(size * 0.7), round(size * 0.4)
    title_size = size * 1.06
    row = measure.measure_text("Ag", size, mono=True)
    rows = [
        (
            key,
            val,
            measure.measure_text(key, size, weight=600).width,
            measure.measure_text(val, size, weight=500, mono=True).width,
        )
        for key, val in pairs
    ]
    title_m = (
        measure.measure_text(title, title_size, weight=700) if title else None
    )
    content_w = max(
        [kw + vw for _, _, kw, vw in rows]
        + ([title_m.width] if title_m is not None else [0.0])
    )
    card_w = round(content_w + 2 * pad)
    title_h = title_m.height + gap if title_m is not None else 0.0
    card_h = round(2 * pad + title_h + len(rows) * row.height)
    mg = round(size * 0.5)  # transparent margin so the drop shadow has room

    canvas = Canvas.blank(card_w + 2 * mg, card_h + 2 * mg)
    canvas.rounded_rect(
        Rect(mg, mg, mg + card_w, mg + card_h),
        radius=round(size * 0.55),
        fill=brand.CARD_BG,
        shadow=Shadow(blur=size * 0.5, dy=size * 0.14),
    )
    x0, y = mg + pad, float(mg + pad)
    if title_m is not None:
        canvas.text(
            (x0, y + title_m.ascent),
            str(title),
            size=title_size,
            color=title_color,
            weight=700,
        )
        y += title_m.height + gap
    for key, val, kw, _vw in rows:
        base = y + row.ascent
        canvas.text(
            (x0, base), key, size=size, color=brand.CARD_KEY, weight=600
        )
        canvas.text(
            (x0 + kw, base),
            val,
            size=size,
            color=brand.CARD_TEXT,
            weight=500,
            mono=True,
        )
        y += row.height
    return canvas.to_rgba()


def blit_rgba_on_bgr(
    frame: np.ndarray, rgba: np.ndarray, x: int, y: int
) -> None:
    """Alpha-composite an RGBA card onto a BGR ``frame`` at ``(x, y)``."""
    fh, fw = frame.shape[:2]
    ch, cw = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + cw), min(fh, y + ch)
    if x1 <= x0 or y1 <= y0:
        return
    sub = rgba[y0 - y : y1 - y, x0 - x : x1 - x]
    roi = frame[y0:y1, x0:x1]
    alpha = sub[..., 3:4].astype(np.float32) / 255.0
    card_bgr = sub[..., 2::-1].astype(np.float32)  # RGB -> BGR
    roi[:] = (
        card_bgr * alpha + roi.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)


def draw_tooltip(
    frame: np.ndarray, tooltip: Tooltip, at: tuple[int, int]
) -> None:
    """Draw ``tooltip`` near ``at`` on the BGR ``frame`` (a no-op if empty).

    The card is clamped to stay fully in-bounds and skipped entirely if it would
    not fit within the frame.
    """
    if tooltip.is_empty:
        return
    height, width = frame.shape[:2]
    # Scale the type to the frame so it stays legible on large windows.
    size = int(min(24, max(13, round(min(width, height) / 48))))
    card = render_tooltip_card(tooltip, size)
    ch, cw = card.shape[:2]
    if cw >= width or ch >= height:
        return
    x = max(0, min(int(at[0]) + 16, width - cw))
    y = max(0, min(int(at[1]) + 16, height - ch))
    blit_rgba_on_bgr(frame, card, x, y)
