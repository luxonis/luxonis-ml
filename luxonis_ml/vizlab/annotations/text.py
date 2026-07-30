"""Image-level captions and a class-color legend.

`Caption` draws a short line of text as a card in a corner (optionally
larger, as a title). `Legend` draws a color key — a stack of swatch + name
rows inside a single card. Both are corner-stacked overlays (drawn on top of
everything, and reserved so box labels avoid them).
"""

import math
from collections.abc import Sequence

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render.canvas import Canvas, Shadow, TextMetrics
from luxonis_ml.vizlab.render.markup import Span, parse
from luxonis_ml.vizlab.style import Style

from .base import RenderContext
from .card import draw_card_background
from .overlay import (
    Cell,
    Corner,
    CornerStack,
    resolve_chrome,
    swatch_outline,
)

_CAPTION_BG = brand.CAPTION_BG

_PAD, _ROW_GAP, _SWATCH_GAP = 10.0, 6.0, 8.0

# A wrapped, measured text line: ``(styled spans, metrics)``. Card/caption/legend
# text is parsed as inline markup (``<b>``/``<i>``/``<code>``), so a line is a
# list of styled `Span` rather than a bare string.
_Line = tuple[list[Span], TextMetrics]
# A card row: ``(styled spans, metrics, swatch_color_or_None)``.
_Row = tuple[list[Span], TextMetrics, "Color | None"]


def _wrap_measured(
    canvas: Canvas,
    texts: list[str],
    size: float,
    weight: int,
    avail: float,
) -> list[_Line]:
    """Wrap each text to ``avail`` px and measure every resulting line.

    Text is parsed as inline markup, so ``<b>``/``<i>``/``<code>`` runs render
    bold/italic/monospace; ``weight`` sets the untagged baseline.
    """
    lines: list[_Line] = []
    for text in texts:
        spans = parse(text, weight=weight)
        wrapped = canvas.wrap_spans(spans, size, max_width=avail) or [
            [Span("")]
        ]
        lines.extend(
            (line, canvas.measure_spans(line, size)) for line in wrapped
        )
    return lines


def _card_cell(
    style: Style,
    title_lines: list[_Line],
    rows: list[_Row],
    swatch: float,
    chrome: brand.Chrome,
) -> Cell | None:
    """Build one card `Cell` from wrapped title lines and rows.

    ``swatch`` is the swatch square size (``0`` for a swatch-less card). ``chrome``
    supplies the (theme-aware) card fill/text colors. Shared by `InfoCard` (no
    swatch) and `Legend` (swatch on each entry's first line).
    """
    if not rows and not title_lines:
        return None
    size = style.font_size
    swatch_col = swatch + _SWATCH_GAP if swatch else 0.0
    row_h = max((m.height for _, m, _ in rows), default=size)
    title_line_h = max((m.height for _, m in title_lines), default=0.0)
    title_h = (
        len(title_lines) * title_line_h + _ROW_GAP if title_lines else 0.0
    )
    content_w = max(
        [swatch_col + m.width for _, m, _ in rows]
        + [m.width for _, m in title_lines],
        default=0.0,
    )
    card_w = content_w + 2 * _PAD
    card_h = (
        2 * _PAD
        + title_h
        + len(rows) * row_h
        + _ROW_GAP * max(0, len(rows) - 1)
    )

    def _draw(cv: Canvas, rect: Rect) -> None:
        _draw_card(
            cv,
            rect,
            style,
            title_lines,
            title_line_h,
            rows,
            row_h,
            swatch,
            chrome,
        )

    return Cell(card_w, card_h, _draw)


def _draw_card(
    cv: Canvas,
    rect: Rect,
    style: Style,
    title_lines: list[_Line],
    title_line_h: float,
    rows: list[_Row],
    row_h: float,
    swatch: float,
    chrome: brand.Chrome,
) -> None:
    """Paint a card background, its title lines, then its (optionally swatched) rows."""
    draw_card_background(cv, rect, style, chrome)
    title_size = style.font_size * 1.05
    y = rect.top + _PAD
    for line, metrics in title_lines:
        cv.draw_spans(
            (rect.left + _PAD, y + metrics.ascent),
            line,
            size=title_size,
            color=chrome.card_title,
        )
        y += title_line_h
    if title_lines:
        y += _ROW_GAP
    text_x = rect.left + _PAD + (swatch + _SWATCH_GAP if swatch else 0.0)
    for line, metrics, color in rows:
        if color is not None and swatch:
            sw_top = y + (row_h - swatch) / 2
            cv.rounded_rect(
                Rect(
                    rect.left + _PAD,
                    sw_top,
                    rect.left + _PAD + swatch,
                    sw_top + swatch,
                ),
                radius=3.0,
                fill=color,
                stroke=swatch_outline(chrome.card_bg),
                stroke_width=1.0,
            )
        cv.draw_spans(
            (text_x, y + metrics.ascent),
            line,
            size=style.font_size,
            color=chrome.card_text,
        )
        y += row_h + _ROW_GAP


def _swatch_rows(
    canvas: Canvas,
    items: "list[tuple[str, Color]]",
    size: float,
    weight: int,
    avail: float,
) -> list[_Row]:
    """Build swatched rows: each name wraps; its swatch sits on the first line."""
    rows: list[_Row] = []
    for name, color in items:
        spans = parse(name, weight=weight)
        wrapped = canvas.wrap_spans(spans, size, max_width=avail) or [
            [Span("")]
        ]
        for i, line in enumerate(wrapped):
            metrics = canvas.measure_spans(line, size)
            rows.append((line, metrics, color if i == 0 else None))
    return rows


def _ellipsize(
    canvas: Canvas, text: str, size: float, weight: int, max_w: float
) -> str:
    """Truncate ``text`` with a trailing ``…`` so it fits within ``max_w`` px."""
    if canvas.measure_text(text, size, weight=weight).width <= max_w:
        return text
    cut = text
    while (
        cut
        and canvas.measure_text(cut + "…", size, weight=weight).width > max_w
    ):
        cut = cut[:-1]
    return cut + "…" if cut else "…"


def _legend_columns_cell(
    canvas: Canvas,
    style: Style,
    title_lines: list[_Line],
    items: "list[tuple[str, Color]]",
    swatch: float,
    chrome: brand.Chrome,
    avail_w: float,
    avail_h: float,
) -> Cell:
    """Lay a legend that overflows one column into several, capped with "+N more".

    Entries flow column-major into as many columns as fit the width, each row a
    single (ellipsized) name; if even that cannot hold every class the last slot
    becomes a muted ``+N more`` so the count is never silently dropped.
    """
    size, weight = style.font_size, style.font_weight
    swatch_col = swatch + _SWATCH_GAP
    col_gap = _PAD * 1.5
    row_m = canvas.measure_text("Ag", size)
    row_h = row_m.height
    step = row_h + _ROW_GAP

    title_line_h = max((m.height for _, m in title_lines), default=0.0)
    title_h = (
        len(title_lines) * title_line_h + _ROW_GAP if title_lines else 0.0
    )
    inner_w = max(1.0, avail_w - 2 * _PAD)
    inner_h = max(1.0, avail_h - 2 * _PAD - title_h)
    rows_per_col = max(1, int((inner_h + _ROW_GAP) // step))

    min_name = canvas.measure_text("MMMM", size, weight=weight).width
    max_cols = max(
        1, int((inner_w + col_gap) // (swatch_col + min_name + col_gap))
    )
    n_cols = min(max(1, math.ceil(len(items) / rows_per_col)), max_cols)

    capacity = n_cols * rows_per_col
    shown = list(items)
    overflow = 0
    if len(items) > capacity:
        shown = shown[: capacity - 1]
        overflow = len(items) - len(shown)

    max_name = max(
        (canvas.measure_text(n, size, weight=weight).width for n, _ in shown),
        default=min_name,
    )
    name_w = max(
        min_name,
        min(max_name, inner_w / n_cols - swatch_col - col_gap),
    )
    col_w = swatch_col + name_w

    cells: list[tuple[str, Color | None]] = [
        (_ellipsize(canvas, name, size, weight, name_w), color)
        for name, color in shown
    ]
    if overflow:
        cells.append((f"+{overflow} more", None))
    per_col = max(1, math.ceil(len(cells) / n_cols))

    card_w = 2 * _PAD + n_cols * col_w + (n_cols - 1) * col_gap
    card_h = 2 * _PAD + title_h + per_col * step - _ROW_GAP

    def _draw(cv: Canvas, rect: Rect) -> None:
        draw_card_background(cv, rect, style, chrome)
        y0 = rect.top + _PAD
        for line, metrics in title_lines:
            cv.draw_spans(
                (rect.left + _PAD, y0 + metrics.ascent),
                line,
                size=style.font_size * 1.05,
                color=chrome.card_title,
            )
            y0 += title_line_h
        if title_lines:
            y0 += _ROW_GAP
        for i, (name, color) in enumerate(cells):
            x = rect.left + _PAD + (i // per_col) * (col_w + col_gap)
            y = y0 + (i % per_col) * step
            if color is not None:
                sw_top = y + (row_h - swatch) / 2
                cv.rounded_rect(
                    Rect(x, sw_top, x + swatch, sw_top + swatch),
                    radius=3.0,
                    fill=color,
                    stroke=swatch_outline(chrome.card_bg),
                    stroke_width=1.0,
                )
            cv.text(
                (x + swatch_col, y + row_m.ascent),
                name,
                size=size,
                color=chrome.card_text,
                weight=weight,
            )

    return Cell(card_w, card_h, _draw)


class Caption(CornerStack):
    """A short line of text drawn as a card in a corner.

    .. image:: TODO-HOST/overlays.png
       :alt: Captions among classification tags, a legend, and analytics.

    Attributes:
        text: The caption text (single line).
        background: Card fill color; the text uses a readable contrast of it.
        title: Draw larger and bolder, as a title.

    See `CornerStack` for
    ``corner``/``margin``/``gap``.

    Examples:
        >>> from luxonis_ml.vizlab import Caption, Corner
        >>> caption = Caption(
        ...     text="camera 1", corner=Corner.BOTTOM_LEFT, title=True
        ... )
        >>> caption.text
        'camera 1'

    """

    text: str = ""
    background: ColorLike = _CAPTION_BG
    title: bool = False

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        if not self.text:
            return []
        cap_style = (
            style.merge(font_size=style.font_size * 1.4, font_weight=700)
            if self.title
            else style
        )
        canvas = ctx.canvas
        size, weight = cap_style.font_size, cap_style.font_weight
        pad_x, pad_y = cap_style.label_pad_x, cap_style.label_pad_y
        # Wrap to keep the chip within the canvas (minus the corner margin).
        avail = max(1.0, canvas.width - 2 * self.margin - 2 * pad_x)
        spans = parse(self.text, weight=weight)
        lines = canvas.wrap_spans(spans, size, max_width=avail)
        if not lines:
            return []
        # An unset background follows the theme (white chip + purple text on
        # light, navy chip + near-white text on dark); an explicit one is honored
        # verbatim, with a readable contrast picked for its text.
        if self.background == _CAPTION_BG:
            chrome = resolve_chrome(ctx)
            fill, text_color = chrome.caption_bg, chrome.card_text
        else:
            fill = Color.parse(self.background)
            text_color = fill.readable_text_color()
        measured: list[_Line] = [
            (line, canvas.measure_spans(line, size)) for line in lines
        ]
        line_h = max(m.height for _, m in measured)
        content_w = max(m.width for _, m in measured)
        card_w = content_w + 2 * pad_x
        card_h = len(measured) * line_h + 2 * pad_y

        def _draw(cv: Canvas, rect: Rect) -> None:
            cv.rounded_rect(
                rect,
                radius=cap_style.label_radius,
                fill=fill,
                shadow=Shadow(blur=4.0, dy=1.0) if cap_style.shadow else None,
            )
            y = rect.top + pad_y
            for line, m in measured:
                cv.draw_spans(
                    (rect.left + pad_x, y + m.ascent),
                    line,
                    size=size,
                    color=text_color,
                )
                y += line_h

        return [Cell(card_w, card_h, _draw)]


class InfoCard(CornerStack):
    """A titled card of plain text rows, e.g. per-object metadata.

    Like `Legend` but without color swatches: a rounded card with an optional
    heading and one text row per line. Used to surface annotation metadata that
    has no bounding box to anchor a hover tooltip to.

    .. image:: TODO-HOST/typography.png
       :alt: Text cards with bundled fonts and inline bold/italic/mono markup.

    Attributes:
        rows: The text lines to show, top to bottom.
        title: Optional heading drawn above the rows.

    See `CornerStack` for ``corner``/``margin``/``gap``.

    Examples:
        >>> from luxonis_ml.vizlab import InfoCard
        >>> card = InfoCard(
        ...     rows=["track_id: 7", "speed: 12.4"], title="metadata"
        ... )
        >>> card.rows[0]
        'track_id: 7'

    """

    rows: list[str] = []
    title: str | None = None
    corner: Corner = Corner.TOP_LEFT

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        size, weight = style.font_size, style.font_weight
        # Wrap rows/title to keep the card within the canvas (minus the margin).
        avail = max(1.0, canvas.width - 2 * self.margin - 2 * _PAD)
        title_lines = (
            _wrap_measured(canvas, [self.title], size * 1.05, 700, avail)
            if self.title is not None
            else []
        )
        rows: list[_Row] = [
            (line, metrics, None)
            for line, metrics in _wrap_measured(
                canvas, self.rows, size, weight, avail
            )
        ]
        cell = _card_cell(style, title_lines, rows, 0.0, resolve_chrome(ctx))
        return [cell] if cell is not None else []


class Legend(CornerStack):
    """A class-color key: swatch + name rows inside one card.

    .. image:: TODO-HOST/overlays.png
       :alt: A class-color legend among captions, tags, and analytics.

    Attributes:
        entries: Class names (colored from the palette) or explicit
            ``(name, color)`` pairs.
        title: Optional heading drawn above the rows.

    See `CornerStack` for
    ``corner``/``margin``/``gap``.

    Examples:
        Class names use the active palette; tuples pin an explicit color:

        >>> from luxonis_ml.vizlab import Legend
        >>> legend = Legend(
        ...     entries=["car", ("road", "#555555")], title="classes"
        ... )
        >>> len(legend.entries)
        2

    """

    entries: Sequence[str | tuple[str, ColorLike]] = ()
    title: str | None = None
    corner: Corner = Corner.BOTTOM_RIGHT

    def _resolved_entries(self, ctx: RenderContext) -> list[tuple[str, Color]]:
        """Resolve each entry to a ``(name, color)`` pair."""
        palette = self.resolved_palette(ctx)
        items: list[tuple[str, Color]] = []
        for entry in self.entries:
            if isinstance(entry, str):
                items.append((entry, palette.color_for(entry)))
            else:
                name, color = entry
                items.append((name, Color.parse(color)))
        return items

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        size, weight = style.font_size, style.font_weight
        swatch = size
        # Wrap names to keep the card within the canvas; a wrapped name's swatch
        # is drawn on its first line, continuation lines sit under the text.
        name_avail = max(
            1.0,
            canvas.width - 2 * self.margin - 2 * _PAD - swatch - _SWATCH_GAP,
        )
        rows = _swatch_rows(
            canvas, self._resolved_entries(ctx), size, weight, name_avail
        )
        title_avail = max(1.0, canvas.width - 2 * self.margin - 2 * _PAD)
        title_lines = (
            _wrap_measured(canvas, [self.title], size * 1.05, 700, title_avail)
            if self.title is not None
            else []
        )
        chrome = resolve_chrome(ctx)
        cell = _card_cell(style, title_lines, rows, swatch, chrome)
        if cell is None:
            return []
        # A tall class list would run off the image; flow it into columns (and,
        # if still too many, cap with "+N more") so the card always fits.
        avail_h = canvas.height - 2 * self.margin
        if cell.height <= avail_h:
            return [cell]
        return [
            _legend_columns_cell(
                canvas,
                style,
                title_lines,
                self._resolved_entries(ctx),
                swatch,
                chrome,
                canvas.width - 2 * self.margin,
                avail_h,
            )
        ]
