"""Measuring and fitting text, independent of what is being drawn.

Every card the library draws — the metadata panel, the hover tooltip, the viewer's
controls HUD — has to answer the same three questions before it can lay anything
out: how tall is a line, how wide is this string, and what do I do when it does
not fit. Answering them needs a `Canvas` (Skia owns the font metrics) but not a
*drawing* canvas, so this module keeps one tiny scratch surface and exposes the
measurements over it.

Nothing here knows about panels, tooltips or annotations; callers pass a size and
a weight and get numbers or shortened strings back.

Most callers want the ``*_markup`` and ``*_spans`` variants: text vizlab draws is
inline markup (see `luxonis_ml.vizlab.render.markup`), so it must be measured and
shortened as styled runs rather than as a flat string — otherwise tags would be
counted as characters. The plain-string functions remain for vizlab's own fixed
strings (keycaps, section labels), which carry no markup by construction.
"""

from collections.abc import Sequence

from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.render.canvas import Canvas, TextMetrics
from luxonis_ml.vizlab.render.markup import Span, parse

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


def measure_markup(
    text: str,
    size: float,
    *,
    weight: int = 400,
    italic: bool = False,
    mono: bool = False,
) -> TextMetrics:
    """Measure inline-markup ``text`` as one line, without drawing it.

    The marked-up counterpart of `measure`: tags cost no width, and a run that
    changes size or family widens or heightens the result.

    Args:
        text: The (possibly tagged) string to measure.
        size: Type size in pixels.
        weight: Baseline font weight for untagged text.
        italic: Whether untagged text is italic.
        mono: Whether untagged text uses the monospace face.

    Returns:
        The `TextMetrics` for the line.

    Examples:
        >>> measure_markup("<b>hi</b>", 12.0).width == measure(
        ...     "hi", 12.0, weight=700
        ... ).width
        True

    """
    return _MEASURE.measure_markup(
        text, size, weight=weight, italic=italic, mono=mono
    )


def wrap_markup(
    text: str,
    max_width: float,
    size: float,
    *,
    weight: int = 400,
    italic: bool = False,
    mono: bool = False,
) -> list[list[Span]]:
    """Parse ``text`` as markup and wrap it to ``max_width``.

    Args:
        text: The (possibly tagged) string to wrap.
        max_width: The width to wrap within, in pixels.
        size: Type size in pixels.
        weight: Baseline font weight for untagged text.
        italic: Whether untagged text is italic.
        mono: Whether untagged text uses the monospace face.

    Returns:
        One list of `Span` per output line (at least one line).

    """
    spans = parse(text, weight=weight, italic=italic, mono=mono)
    return _MEASURE.wrap_spans(spans, size, max_width=max_width)


def _take(spans: "Sequence[Span]", count: int) -> list[Span]:
    """Return the first ``count`` characters of ``spans``, styling preserved."""
    kept: list[Span] = []
    for span in spans:
        if count <= 0:
            break
        piece = span.text[:count]
        if piece:
            kept.append(span.with_text(piece))
        count -= len(piece)
    return kept


def _take_end(spans: "Sequence[Span]", count: int) -> list[Span]:
    """Return the last ``count`` characters of ``spans``, styling preserved."""
    kept: list[Span] = []
    for span in reversed(spans):
        if count <= 0:
            break
        piece = span.text[-count:] if count < len(span.text) else span.text
        if piece:
            kept.append(span.with_text(piece))
        count -= len(piece)
    return list(reversed(kept))


def _with_ellipsis(kept: list[Span]) -> list[Span]:
    """Append ``…`` to a truncated span list, in the last run's own style."""
    if not kept:
        return [Span("…")]
    last = kept[-1]
    return [*kept[:-1], last.with_text(f"{last.text}…")]


def ellipsize_spans(
    spans: "Sequence[Span]", max_width: float, size: float
) -> list[Span]:
    """Trim styled ``spans`` from the end with ``…`` until they fit ``max_width``.

    The span-aware counterpart of the plain-string trimming the cards used to
    do: the surviving text keeps the weight, family, color, and size of the run
    it came from.

    Args:
        spans: The styled runs making up one line.
        max_width: The width to fit within, in pixels.
        size: Type size in pixels.

    Returns:
        ``spans`` unchanged if they already fit, otherwise a shortened list; in
        the worst case the ellipsis alone.

    """
    if _MEASURE.measure_spans(spans, size).width <= max_width:
        return list(spans)
    total = sum(len(span.text) for span in spans)
    for keep in range(total - 1, 0, -1):
        candidate = _with_ellipsis(_take(spans, keep))
        if _MEASURE.measure_spans(candidate, size).width <= max_width:
            return candidate
    return [Span("…")]


def middle_ellipsize_spans(
    spans: "Sequence[Span]", max_width: float, size: float
) -> list[Span]:
    """Trim styled ``spans`` in the middle with ``…`` until they fit ``max_width``.

    Both ends are kept, which is what makes a long path or filename still
    identifiable after shortening. The styling of each surviving run is kept.

    Args:
        spans: The styled runs making up one line.
        max_width: The width to fit within, in pixels.
        size: Type size in pixels.

    Returns:
        ``spans`` unchanged if they already fit, otherwise a shortened list; in
        the worst case the ellipsis alone.

    """
    if _MEASURE.measure_spans(spans, size).width <= max_width:
        return list(spans)
    total = sum(len(span.text) for span in spans)
    for keep in range(total - 1, 0, -1):
        head, tail = (keep + 1) // 2, keep // 2
        candidate = [
            *_with_ellipsis(_take(spans, head)),
            *_take_end(spans, tail),
        ]
        if _MEASURE.measure_spans(candidate, size).width <= max_width:
            return candidate
    return [Span("…")]


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


def upper_spans(spans: "Sequence[Span]") -> list[Span]:
    """Upper-case every run's text without disturbing its styling.

    Headings that shout are built this way rather than by upper-casing the
    marked-up string, which would also mangle ``<span>`` attribute values.

    Args:
        spans: The styled runs to upper-case.

    Returns:
        The runs with upper-cased text.

    """
    return [span.with_text(span.text.upper()) for span in spans]


def tracked_spans_width(
    spans: "Sequence[Span]", size: float, *, tracking: float = 0.0
) -> float:
    """Width of styled ``spans`` drawn with ``tracking`` between characters.

    Args:
        spans: The styled runs making up one line.
        size: Type size in pixels.
        tracking: Extra space after each character, in pixels.

    Returns:
        The total advance in pixels.

    """
    return sum(
        width(char, size * span.scale, weight=span.weight, mono=span.mono)
        + tracking
        for span in spans
        for char in span.text
    )


def draw_tracked_spans(
    canvas: Canvas,
    spans: "Sequence[Span]",
    x: float,
    baseline: float,
    size: float,
    color: Color,
    *,
    tracking: float = 0.0,
) -> None:
    """Draw styled ``spans`` a character at a time with ``tracking`` between them.

    The span-aware counterpart of `draw_tracked`, so a letterspaced heading can
    still carry inline markup.

    Args:
        canvas: The canvas to draw on.
        spans: The styled runs to draw in order.
        x: Left edge, in canvas pixels.
        baseline: Text baseline, in canvas pixels.
        size: Type size in pixels.
        color: Fill color for runs without an explicit one of their own.
        tracking: Extra space after each character, in pixels.

    """
    for span in spans:
        run_size = size * span.scale
        for char in span.text:
            canvas.draw_spans(
                (x, baseline), [span.with_text(char)], size=size, color=color
            )
            x += (
                width(char, run_size, weight=span.weight, mono=span.mono)
                + tracking
            )
