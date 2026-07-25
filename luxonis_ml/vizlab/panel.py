"""Append a metadata sidebar ("second window") to an image.

`with_panel` renders an image and appends a panel that shows arbitrary
JSON-like metadata (augmentations, tags, source, filenames, ...) as an indented
key/value tree. Like the `compose` functions it renders the image at
native resolution and returns a new `Image`, so the panel
never occludes pixels or labels and the original is untouched.
"""

from collections.abc import Mapping, Sequence

from .canvas import Canvas
from .color import Color, ColorLike
from .geometry import Rect
from .image import Image
from .style import DEFAULT_STYLE, Style, get_default_theme

_PAD = 16.0
_INDENT = 14.0
_LINE_GAP = 5.0
_PANEL_SIZE = 15.0
_MIN_WIDTH = 220.0
_MAX_WIDTH = 400.0

_KEY = Color(150, 185, 235)
_VALUE = Color(222, 222, 228)
_TITLE = Color(245, 245, 248)
_DIVIDER = Color(255, 255, 255, 30)

_MEASURE = Canvas.blank(1, 1)

# One logical line: (depth, prefix, prefix_is_key, body).
Line = tuple[int, str, bool, str]


def _is_container(value: object) -> bool:
    """Whether a value is a nested container (mapping or non-string sequence)."""
    return isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    )


def _format_scalar(value: object) -> str:
    """Format a leaf value as a compact single-line string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= 500 else text[:499] + "…"


def _format_tree(data: object, depth: int = 0) -> list[Line]:
    """Flatten JSON-like data into indented key/value lines.

    Args:
        data: A mapping, sequence, or scalar (nested arbitrarily).
        depth: The current nesting depth (used for indentation).

    Returns:
        Logical lines ``(depth, prefix, prefix_is_key, body)``. Keys end in ``": "``
        (or ``":"`` for container headers); list items use a ``"• "`` bullet.

    Examples:
        >>> _format_tree({"a": 1, "b": [2, 3], "c": {"d": True}})
        [(0, 'a: ', True, '1'), (0, 'b:', True, ''), (1, '• ', False, '2'), \
(1, '• ', False, '3'), (0, 'c:', True, ''), (1, 'd: ', True, 'true')]

    """
    lines: list[Line] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            if _is_container(value):
                lines.append((depth, f"{key}:", True, ""))
                lines.extend(_format_tree(value, depth + 1))
            else:
                lines.append((depth, f"{key}: ", True, _format_scalar(value)))
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        for item in data:
            if _is_container(item):
                lines.append((depth, "•", False, ""))
                lines.extend(_format_tree(item, depth + 1))
            else:
                lines.append((depth, "• ", False, _format_scalar(item)))
    else:
        lines.append((depth, "", False, _format_scalar(data)))
    return lines


def _wrap(text: str, size: float, weight: int, max_width: float) -> list[str]:
    """Greedily wrap ``text`` to ``max_width`` using measured word widths."""
    if not text:
        return [""]
    wrapped: list[str] = []
    current = ""
    for word in text.split(" "):
        trial = f"{current} {word}".strip()
        if (
            not current
            or _MEASURE.measure_text(trial, size, weight=weight).width
            <= max_width
        ):
            current = trial
        else:
            wrapped.append(current)
            current = word
    wrapped.append(current)
    return wrapped


# One draw op: (y, x, text, weight, color).
_Op = tuple[float, float, str, int, Color]


def _build_ops(lines: list[Line], content_w: float) -> tuple[list[_Op], float]:
    """Lay out logical lines into positioned text ops; return them and total height."""
    metrics = _MEASURE.measure_text("Ag", _PANEL_SIZE)
    row_h = metrics.height + _LINE_GAP
    ascent = metrics.ascent
    ops: list[_Op] = []
    y = 0.0
    for depth, prefix, is_key, body in lines:
        x = depth * _INDENT
        weight = 600 if is_key else 400
        prefix_w = (
            _MEASURE.measure_text(prefix, _PANEL_SIZE, weight=weight).width
            if prefix
            else 0.0
        )
        body_lines = _wrap(
            body, _PANEL_SIZE, 400, max(24.0, content_w - x - prefix_w)
        )
        if prefix:
            ops.append(
                (y + ascent, x, prefix, weight, _KEY if is_key else _VALUE)
            )
        if body_lines[0]:
            ops.append((y + ascent, x + prefix_w, body_lines[0], 400, _VALUE))
        y += row_h
        for cont in body_lines[1:]:
            ops.append((y + ascent, x + prefix_w, cont, 400, _VALUE))
            y += row_h
    return ops, y


def _auto_width(lines: list[Line]) -> float:
    """Pick a panel width from the content, clamped to a sensible range."""
    widest = 0.0
    for depth, prefix, is_key, body in lines:
        weight = 600 if is_key else 400
        prefix_w = _MEASURE.measure_text(
            prefix, _PANEL_SIZE, weight=weight
        ).width
        body_w = _MEASURE.measure_text(body, _PANEL_SIZE, weight=400).width
        widest = max(widest, depth * _INDENT + prefix_w + body_w)
    return min(_MAX_WIDTH, max(_MIN_WIDTH, widest + 2 * _PAD))


def with_panel(
    image: Image,
    data: object,
    *,
    side: str = "right",
    width: float | None = None,
    title: str | None = None,
    style: Style | None = None,
    bg: ColorLike | None = None,
) -> Image:
    """Render ``image`` and append a metadata panel beside it.

    Args:
        image: The image to annotate. Rendered at native resolution.
        data: JSON-like metadata (mapping/sequence/scalar, nested arbitrarily).
        side: Which edge to attach the panel to: ``"right"`` (default), ``"left"``,
            or ``"bottom"``.
        width: Panel width in pixels; ``None`` auto-sizes from the content.
        title: Optional bold heading drawn above the tree.
        style: Style whose font is used (defaults to the library default).
        bg: Panel background color; defaults to the image's theme background.

    Returns:
        A new `Image` of the image plus the panel. The input is
        not mutated.

    """
    style = style or DEFAULT_STYLE
    base = image.render()
    img_h, img_w = base.shape[:2]
    background = (
        Color.parse(bg)
        if bg is not None
        else (image.theme or get_default_theme()).background
    )

    lines = _format_tree(data)
    panel_w = width if width is not None else _auto_width(lines)
    content_w = panel_w - 2 * _PAD
    ops, content_h = _build_ops(lines, content_w)

    title_metrics = (
        _MEASURE.measure_text(title, _PANEL_SIZE * 1.15, weight=700)
        if title is not None
        else None
    )
    title_h = title_metrics.height + _LINE_GAP * 2 if title_metrics else 0.0
    panel_h = 2 * _PAD + title_h + content_h

    horizontal = side in ("right", "left")
    if horizontal:
        out_w = int(img_w + panel_w)
        out_h = int(max(img_h, panel_h))
        panel_x = 0.0 if side == "left" else float(img_w)
        image_x = int(panel_w) if side == "left" else 0
        image_y = 0
    else:  # bottom
        out_w = int(max(img_w, panel_w))
        out_h = int(img_h + panel_h)
        panel_x = 0.0
        image_x = 0
        image_y = 0

    canvas = Canvas.blank(out_w, out_h)
    canvas.rounded_rect(Rect(0, 0, out_w, out_h), 0.0, fill=background)
    canvas.blit(base, image_x, image_y)
    _draw_divider(canvas, side, img_w, img_h, panel_w, out_w, out_h)

    panel_y = float(img_h) if not horizontal else 0.0
    _draw_panel(canvas, ops, title, title_metrics, panel_x, panel_y)
    return Image(canvas.to_rgba(), theme=image.theme)


def _draw_divider(
    canvas: Canvas,
    side: str,
    img_w: int,
    img_h: int,
    panel_w: float,
    out_w: int,
    out_h: int,
) -> None:
    """Draw a subtle line between the image and the panel."""
    if side == "right":
        canvas.line((img_w, 0), (img_w, out_h), _DIVIDER, width=1.0)
    elif side == "left":
        canvas.line((panel_w, 0), (panel_w, out_h), _DIVIDER, width=1.0)
    else:  # bottom
        canvas.line((0, img_h), (out_w, img_h), _DIVIDER, width=1.0)


def _draw_panel(
    canvas: Canvas,
    ops: list[_Op],
    title: str | None,
    title_metrics: object,
    panel_x: float,
    panel_y: float,
) -> None:
    """Draw the optional title and the laid-out key/value ops."""
    x0 = panel_x + _PAD
    y0 = panel_y + _PAD
    if title is not None and title_metrics is not None:
        canvas.text(
            (x0, y0 + title_metrics.ascent),  # type: ignore[attr-defined]
            title,
            size=_PANEL_SIZE * 1.15,
            color=_TITLE,
            weight=700,
        )
        y0 += title_metrics.height + _LINE_GAP * 2  # type: ignore[attr-defined]
    for op_y, op_x, text, weight, color in ops:
        canvas.text(
            (x0 + op_x, y0 + op_y),
            text,
            size=_PANEL_SIZE,
            color=color,
            weight=weight,
        )
