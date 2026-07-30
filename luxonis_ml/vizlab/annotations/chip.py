"""The rounded label chip shared by boxes, masks, and classification tags.

A chip is a small rounded rectangle filled with an annotation's color, holding a
line of text (class, confidence, payload) in a readable contrast color. Keeping it
here means every annotation that shows a label draws an identical-looking one.
"""

from typing import TYPE_CHECKING

from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import XY, Rect
from luxonis_ml.vizlab.render.canvas import Canvas, Shadow, TextMetrics
from luxonis_ml.vizlab.style import Style

from .layout import LabelLayout, label_candidates

if TYPE_CHECKING:
    from .base import RenderContext


def compose_label(
    label: str | None, score: float | None, payload: str | float | None
) -> str:
    """Compose a chip string from a class label, score, and payload.

    Args:
        label: Class name, or ``None``.
        score: Confidence in ``[0, 1]``, or ``None``.
        payload: Arbitrary value, or ``None``.

    Returns:
        The chip text, for example "car  95%", or an empty string when there
        is nothing to show.

    Examples:
        >>> compose_label("car", 0.95, None)
        'car  95%'
        >>> compose_label(None, None, "HELLO")
        'HELLO'
        >>> compose_label(None, None, None)
        ''

    """
    parts: list[str] = []
    if label:
        parts.append(label)
    if score is not None:
        parts.append(f"{round(float(score) * 100)}%")
    text = "  ".join(parts)
    if payload is not None and payload != "":
        payload_text = str(payload)
        text = f"{text}  ·  {payload_text}" if text else payload_text
    return text


def chip_size(
    canvas: Canvas, text: str, style: Style
) -> tuple[float, float, TextMetrics]:
    """Measure the chip box for ``text`` under ``style``.

    Args:
        canvas: The canvas (for text measurement).
        text: The chip text, which may carry inline markup.
        style: The resolved style.

    Returns:
        A ``(width, height, metrics)`` tuple, all in pixels.

    """
    metrics = canvas.measure_markup(
        text, style.font_size, weight=style.font_weight
    )
    width = metrics.width + 2 * style.label_pad_x
    height = metrics.height + 2 * style.label_pad_y
    return width, height, metrics


def draw_chip(
    canvas: Canvas, top_left: XY, text: str, color: Color, style: Style
) -> Rect:
    """Draw a filled label chip with its top-left at ``top_left``.

    Args:
        canvas: The canvas to draw on.
        top_left: The chip's top-left ``(x, y)`` in pixels.
        text: The chip text (assumed non-empty).
        color: The chip fill color; text uses a readable contrast of it.
        style: The resolved style.

    Returns:
        The chip's `Rect`.

    """
    width, height, metrics = chip_size(canvas, text, style)
    x, y = top_left
    rect = Rect(x, y, x + width, y + height)
    alpha = style.label_alpha
    fill = color if alpha >= 1.0 else color.with_alpha(alpha)
    text_color = color.readable_text_color()
    if alpha < 1.0:
        text_color = text_color.with_alpha(alpha)
    canvas.rounded_rect(
        rect,
        radius=style.label_radius,
        fill=fill,
        shadow=Shadow(blur=4.0, dy=1.0)
        if style.shadow and alpha >= 1.0
        else None,
    )
    baseline_y = rect.top + style.label_pad_y + metrics.ascent
    canvas.markup(
        (rect.left + style.label_pad_x, baseline_y),
        text,
        size=style.font_size,
        color=text_color,
        weight=style.font_weight,
    )
    return rect


def place_label(
    ctx: "RenderContext",
    region: Rect,
    label: str | None,
    score: float | None,
    payload: str | float | None,
    color: Color,
    style: Style,
) -> None:
    """Place and draw an annotation's label chip near ``region``, avoiding overlap.

    Composes the chip text and, if non-empty, measures it, proposes candidate
    positions around ``region``, picks the least-overlapping one via the shared
    layout, and draws it. Used by every box/mask-shaped annotation.

    Args:
        ctx: The current render context (supplies the canvas and layout).
        region: The annotation's pixel bounds the chip labels.
        label: Class name, or ``None``.
        score: Confidence in ``[0, 1]``, or ``None``.
        payload: Arbitrary value, or ``None``.
        color: The chip fill color.
        style: The resolved style.

    """
    text = compose_label(label, score, payload)
    if not text:
        return
    canvas = ctx.canvas
    chip_w, chip_h, _ = chip_size(canvas, text, style)
    layout = ctx.layout or LabelLayout(canvas.width, canvas.height)
    candidates = label_candidates(
        region, chip_w, chip_h, style.label_placement
    )
    placement = layout.place(chip_w, chip_h, candidates, region=region)
    rect = placement.rect
    if placement.leader is not None:
        _draw_leader(canvas, placement.leader, rect, color, style)
    draw_chip(canvas, (rect.left, rect.top), text, color, style)


def _draw_leader(
    canvas: Canvas, anchor: XY, chip: Rect, color: Color, style: Style
) -> None:
    """Connect a pushed-out chip back to its box with a thin leader line.

    Drawn before the chip (so the chip sits on top), from ``anchor`` on the box to
    the nearest point on the chip, with a small dot pinning the box end.

    Args:
        canvas: The canvas to draw on.
        anchor: The point on the labeled box to lead from.
        chip: The placed chip rectangle.
        color: The chip color (the connector is a faded shade of it).
        style: The resolved style (sets the line weight).

    """
    ax, ay = anchor
    tip = (
        max(chip.left, min(ax, chip.right)),
        max(chip.top, min(ay, chip.bottom)),
    )
    line_color = color.with_alpha(0.55)
    canvas.line(
        anchor, tip, line_color, width=max(1.0, style.stroke_width * 0.4)
    )
    canvas.circle(anchor, max(1.5, style.stroke_width * 0.7), fill=line_color)
