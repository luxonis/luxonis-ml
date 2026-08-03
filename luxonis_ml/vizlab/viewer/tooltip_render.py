"""Render a `Tooltip` as a native vizlab card and composite it onto a frame.

These helpers are pure vizlab `Canvas`/NumPy — no windowing or OpenCV — so they
are shared by any backend. The card matches the rest of the UI: a rounded,
translucent brand card with a soft shadow, a class-tinted title, periwinkle Inter
keys, and near-white JetBrains Mono values.

The title and rows are drawn as inline markup, like the rest of the library's
text; the adapters escape dataset metadata on the way in, so an arbitrary value
still renders verbatim (see `luxonis_ml.vizlab.render.markup.escape`).
"""

import numpy as np

from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render import text_layout
from luxonis_ml.vizlab.render.canvas import Canvas, TextMetrics
from luxonis_ml.vizlab.tooltip import Tooltip


def _draw_tooltip_header(
    canvas: Canvas,
    *,
    title: str | None,
    title_metrics: TextMetrics | None,
    title_size: float,
    title_color: Color,
    tint: Color | None,
    x: float,
    y: float,
    line_h: float,
    gap: int,
    swatch: int,
    swatch_gap: int,
) -> float:
    """Draw the optional tooltip swatch/title and return the next y offset."""
    if title_metrics is None and not swatch:
        return y
    text_x = x
    if tint is not None and swatch:
        swatch_y = y + (line_h - swatch) / 2
        canvas.rounded_rect(
            Rect(text_x, swatch_y, text_x + swatch, swatch_y + swatch),
            radius=round(swatch * 0.3),
            fill=tint,
        )
        text_x += swatch + swatch_gap
    if title_metrics is not None:
        canvas.markup(
            (text_x, y + title_metrics.ascent),
            str(title),
            size=title_size,
            color=title_color,
            weight=700,
        )
    return y + line_h + gap


def _draw_tooltip_rows(
    canvas: Canvas,
    rows: list[tuple[str, str, float, float]],
    *,
    x: float,
    y: float,
    size: int,
    metrics: TextMetrics,
) -> None:
    """Draw aligned tooltip key/value rows."""
    from luxonis_ml.utils.color import brand

    for key, value, key_width, _value_width in rows:
        baseline = y + metrics.ascent
        canvas.markup(
            (x, baseline),
            key,
            size=size,
            color=brand.CARD_KEY,
            weight=600,
        )
        canvas.markup(
            (x + key_width, baseline),
            value,
            size=size,
            color=brand.CARD_TEXT,
            weight=500,
            mono=True,
        )
        y += metrics.height


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
    from luxonis_ml.vizlab.render.canvas import Shadow

    title = tooltip.title
    tint = tooltip.tint
    title_color = tint if tint is not None else brand.CARD_TITLE
    # A key already carrying its own punctuation (a JSON-like tree's ``"key:"``
    # header or ``"• "`` bullet, see `Tooltip.resolved_rows`) is drawn as-is;
    # a plain key gets the usual ``": "`` separator.
    pairs = [
        (key if key.endswith((":", " ")) else f"{key}: ", value)
        for key, value in tooltip.resolved_rows
    ]

    pad, gap = round(size * 0.7), round(size * 0.4)
    title_size = size * 1.06
    row = text_layout.line_metrics(size, mono=True)
    rows = [
        (
            key,
            val,
            text_layout.measure_markup(key, size, weight=600).width,
            text_layout.measure_markup(val, size, weight=500, mono=True).width,
        )
        for key, val in pairs
    ]
    title_m = (
        text_layout.measure_markup(title, title_size, weight=700)
        if title
        else None
    )
    # A class-color swatch leads the header, so the tint reads even when there is
    # no title to tint (a rows-only tooltip). Its side tracks the header height.
    swatch = round(size * 0.95) if tint is not None else 0
    swatch_gap = round(size * 0.45) if swatch else 0
    header_line_h = title_m.height if title_m is not None else float(swatch)
    header_w = (swatch + swatch_gap if title_m is not None else swatch) + (
        title_m.width if title_m is not None else 0.0
    )
    content_w = max([kw + vw for _, _, kw, vw in rows] + [header_w])
    card_w = round(content_w + 2 * pad)
    title_h = header_line_h + gap if (title_m is not None or swatch) else 0.0
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
    y = _draw_tooltip_header(
        canvas,
        title=title,
        title_metrics=title_m,
        title_size=title_size,
        title_color=title_color,
        tint=tint,
        x=float(x0),
        y=y,
        line_h=header_line_h,
        gap=gap,
        swatch=swatch,
        swatch_gap=swatch_gap,
    )
    _draw_tooltip_rows(canvas, rows, x=x0, y=y, size=size, metrics=row)
    return canvas.to_rgba()


def blit_rgba_on_bgr(
    frame: np.ndarray, rgba: np.ndarray, x: int, y: int, *, blur: float = 0.0
) -> None:
    """Alpha-composite an RGBA card onto a BGR ``frame`` at ``(x, y)``.

    With ``blur > 0`` the frame behind the card's solid body is Gaussian-blurred
    first, so the translucent card reads as frosted glass; the soft drop shadow
    around the body is left sharp (see `_frost_backdrop`).
    """
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
    backdrop = (
        _frost_backdrop(roi, sub, blur)
        if blur > 0.0
        else roi.astype(np.float32)
    )
    roi[:] = (card_bgr * alpha + backdrop * (1.0 - alpha)).astype(np.uint8)


#: Card body color (max RGB channel, 0-255) at or above which the backdrop is
#: fully blurred. The card's soft drop shadow is pure black (max channel 0), so
#: keying the frost on color — not alpha — confines it to the colored body and
#: leaves the translucent shadow fringe sharp, avoiding a blurred halo around the
#: panel. Independent of the panel's opacity.
_FROST_BODY_MIN_RGB = 12.0


def _frost_backdrop(
    roi: np.ndarray, card: np.ndarray, blur: float
) -> np.ndarray:
    """Blur the frame region behind a card, confined to the card's colored body.

    The blur fills the card body (so the translucent panel reads as frosted
    glass) but not the drop shadow around it: the shadow is pure black, while the
    body fill and text carry real color, so the mask keys on the card's color
    rather than its alpha. That keeps the frost off the shadow's translucent
    fringe (no blurred halo) and holds at any panel opacity. Returns a float BGR
    array the caller composites the card over.

    Args:
        roi: The BGR frame region under the card (not mutated).
        card: The card's ``(H, W, 4)`` ``uint8`` RGBA, aligned to ``roi``.
        blur: Gaussian blur sigma in pixels.

    Returns:
        The ``(H, W, 3)`` ``float32`` backdrop: blurred under the body, original
        elsewhere.

    """
    blurred = _blur_bgr(roi, blur)
    # The soft shadow is black (max channel 0); the body fill and text are
    # colored, so this is ~1 over the whole body and 0 over the shadow, with a
    # one-pixel feather along the body's anti-aliased edge.
    body = card[..., :3].max(axis=2).astype(np.float32)
    mask = np.clip(body / _FROST_BODY_MIN_RGB, 0.0, 1.0)[..., None]
    return blurred * mask + roi.astype(np.float32) * (1.0 - mask)


def _blur_bgr(roi: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur a BGR region, returning ``float32`` BGR.

    A frosted backdrop is low-frequency, so for a large ``sigma`` the region is
    blurred at a reduced resolution and scaled back up — visually identical to a
    full-resolution blur, but with a cost that stays low and roughly flat as the
    sigma grows.
    """
    from luxonis_ml.vizlab.render.canvas import Canvas, gaussian_blur

    height, width = roi.shape[:2]
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    # Channel order is irrelevant to a symmetric blur, so the BGR data rides in
    # the RGB slots and returns in the same order; a full alpha keeps edges
    # opaque (the clamp tile-mode avoids a dark halo).
    rgba[..., :3] = roi
    rgba[..., 3] = 255
    scale = int(min(4, max(1, sigma // 3)))
    if scale > 1:
        sw, sh = max(1, width // scale), max(1, height // scale)
        small = Canvas.from_rgba(rgba).scaled(sw, sh).to_rgba()
        small = gaussian_blur(small, sigma / scale)
        rgba = Canvas.from_rgba(small).scaled(width, height).to_rgba()
    else:
        rgba = gaussian_blur(rgba, sigma)
    return rgba[..., :3].astype(np.float32)


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
    # A frosted-glass backdrop behind the translucent card, scaled with the type.
    blit_rgba_on_bgr(frame, card, x, y, blur=size * 0.7)
