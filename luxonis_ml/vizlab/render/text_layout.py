"""Measuring and fitting text, independent of what is being drawn.

Every card the library draws — the metadata panel, the hover tooltip, the viewer's
controls HUD — has to answer the same three questions before it can lay anything
out: how tall is a line, how wide is this string, and what do I do when it does
not fit. Answering them needs a `Canvas` (Skia owns the font metrics) but not a
*drawing* canvas, so this module keeps one tiny scratch surface and exposes the
measurements over it.

Nothing here knows about panels, tooltips or annotations; callers pass a size and
a weight and get numbers or shortened strings back.
"""

from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.render.canvas import Canvas, TextMetrics

#: A 1x1 scratch surface used only for font metrics — never drawn to. Shared, so
#: the three cards that measure text do not each keep their own.
_MEASURE = Canvas.blank(1, 1)


def measure(
    text: str, size: float, *, weight: int = 400, mono: bool = False
) -> TextMetrics:
    """Measure ``text`` without drawing it.

    Args:
        text: The string to measure.
        size: Type size in pixels.
        weight: Font weight.
        mono: Measure in the monospace face rather than the sans one.

    Returns:
        The `TextMetrics` for ``text``.

    """
    return _MEASURE.measure_text(text, size, weight=weight, mono=mono)


def width(
    text: str, size: float, *, weight: int = 400, mono: bool = False
) -> float:
    """Return the advance width of ``text`` in pixels.

    Args:
        text: The string to measure.
        size: Type size in pixels.
        weight: Font weight.
        mono: Measure in the monospace face rather than the sans one.

    Returns:
        The width in pixels.

    """
    return measure(text, size, weight=weight, mono=mono).width


def line_metrics(
    size: float, *, weight: int = 400, mono: bool = False
) -> TextMetrics:
    """Return ascent/descent/height for a line of type at ``size``.

    Measured from ``"Ag"`` — an ascender and a descender — so the result
    describes the line box rather than whatever string happens to be drawn in it.

    Args:
        size: Type size in pixels.
        weight: Font weight.
        mono: Measure in the monospace face rather than the sans one.

    Returns:
        The `TextMetrics` of a full-height line.

    """
    return measure("Ag", size, weight=weight, mono=mono)


def wrap(
    text: str,
    max_width: float,
    size: float,
    *,
    weight: int = 400,
    mono: bool = False,
) -> list[str]:
    """Greedily wrap ``text`` to ``max_width`` using measured word widths.

    Args:
        text: The string to wrap; empty text yields one empty line.
        max_width: The width to wrap within, in pixels.
        size: Type size in pixels.
        weight: Font weight.
        mono: Measure in the monospace face rather than the sans one.

    Returns:
        The wrapped lines. A word wider than ``max_width`` is kept whole on its
        own line rather than broken.

    Examples:
        >>> wrap("", 100.0, 12.0)
        ['']

    """
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        trial = f"{current} {word}".strip()
        if (
            not current
            or width(trial, size, weight=weight, mono=mono) <= max_width
        ):
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def middle_ellipsize(
    text: str,
    max_width: float,
    size: float,
    *,
    weight: int = 400,
    mono: bool = True,
) -> str:
    """Trim ``text``'s middle with an ellipsis until it fits ``max_width``.

    Both ends are kept, which is what makes a long path or filename still
    identifiable after shortening.

    Args:
        text: The string to shorten.
        max_width: The width to fit within, in pixels.
        size: Type size in pixels.
        weight: Font weight.
        mono: Measure in the monospace face rather than the sans one.

    Returns:
        ``text`` unchanged if it already fits, otherwise a shortened form; in the
        worst case the ellipsis alone.

    """

    def fits(candidate: str) -> bool:
        return width(candidate, size, weight=weight, mono=mono) <= max_width

    if fits(text):
        return text
    for keep in range(len(text) - 1, 0, -1):
        head, tail = (keep + 1) // 2, keep // 2
        candidate = text[:head] + "…" + (text[-tail:] if tail else "")
        if fits(candidate):
            return candidate
    return "…"


def tracked_width(
    text: str, size: float, *, weight: int = 400, tracking: float = 0.0
) -> float:
    """Width of ``text`` drawn with ``tracking`` pixels between characters.

    Letter-spaced headings are drawn a character at a time, so their width is the
    sum of the glyph advances plus the tracking — not the measured width of the
    whole string.

    Args:
        text: The string to measure.
        size: Type size in pixels.
        weight: Font weight.
        tracking: Extra space after each character, in pixels.

    Returns:
        The total advance in pixels.

    """
    return sum(width(char, size, weight=weight) + tracking for char in text)


def draw_tracked(
    canvas: Canvas,
    text: str,
    x: float,
    baseline: float,
    size: float,
    color: Color,
    *,
    weight: int = 400,
    tracking: float = 0.0,
) -> None:
    """Draw ``text`` one character at a time with ``tracking`` between them.

    Args:
        canvas: The canvas to draw on.
        text: The string to draw.
        x: Left edge, in canvas pixels.
        baseline: Text baseline, in canvas pixels.
        size: Type size in pixels.
        color: Fill color for the glyphs.
        weight: Font weight.
        tracking: Extra space after each character, in pixels.

    """
    for char in text:
        canvas.text((x, baseline), char, size=size, color=color, weight=weight)
        x += width(char, size, weight=weight) + tracking
