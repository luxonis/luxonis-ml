"""Image-level captions and a class-color legend.

`Caption` draws a short line of text as a card in a corner (optionally
larger, as a title). `Legend` draws a color key — a stack of swatch + name
rows inside a single card. Both are corner-stacked overlays (drawn on top of
everything, and reserved so box labels avoid them).
"""

from luxonis_ml.vizlab.canvas import Canvas, Shadow, TextMetrics
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.style import Style

from .base import RenderContext
from .overlay import CARD_BG, CARD_TEXT, Cell, Corner, CornerStack

_CAPTION_BG = Color(22, 22, 26, 235)
_CARD_BG = CARD_BG
_CARD_TEXT = CARD_TEXT


class Caption(CornerStack):
    """A short line of text drawn as a card in a corner.

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
        lines = canvas.wrap_text(
            self.text, size, max_width=avail, weight=weight
        )
        if not lines:
            return []
        fill = Color.parse(self.background)
        text_color = fill.readable_text_color()
        measured = [
            (line, canvas.measure_text(line, size, weight=weight))
            for line in lines
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
                cv.text(
                    (rect.left + pad_x, y + m.ascent),
                    line,
                    size=size,
                    color=text_color,
                    weight=weight,
                )
                y += line_h

        return [Cell(card_w, card_h, _draw)]


class InfoCard(CornerStack):
    """A titled card of plain text rows, e.g. per-object metadata.

    Like `Legend` but without color swatches: a rounded card with an optional
    heading and one text row per line. Used to surface annotation metadata that
    has no bounding box to anchor a hover tooltip to.

    Attributes:
        rows: The text lines to show, top to bottom.
        title: Optional heading drawn above the rows.

    See `CornerStack` for ``corner``/``margin``/``gap``.

    Examples:
        >>> from luxonis_ml.vizlab import InfoCard
        >>> card = InfoCard(rows=["track_id: 7", "speed: 12.4"], title="metadata")
        >>> card.rows[0]
        'track_id: 7'

    """

    rows: list[str] = []
    title: str | None = None
    corner: Corner = Corner.TOP_LEFT

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        size, weight = style.font_size, style.font_weight
        title_size = size * 1.05
        pad, row_gap = 10.0, 6.0
        # Wrap rows/title to keep the card within the canvas (minus the margin).
        avail = max(1.0, canvas.width - 2 * self.margin - 2 * pad)

        lines: list[str] = []
        for row in self.rows:
            lines.extend(
                canvas.wrap_text(row, size, max_width=avail, weight=weight)
                or [""]
            )
        measured = [
            (line, canvas.measure_text(line, size, weight=weight))
            for line in lines
        ]
        title_lines = (
            canvas.wrap_text(
                self.title, title_size, max_width=avail, weight=700
            )
            if self.title is not None
            else []
        )
        title_measured = [
            (line, canvas.measure_text(line, title_size, weight=700))
            for line in title_lines
        ]
        if not measured and not title_measured:
            return []

        row_h = max((m.height for _, m in measured), default=size)
        content_w = max((m.width for _, m in measured), default=0.0)
        title_line_h = max((m.height for _, m in title_measured), default=0.0)
        title_h = (
            len(title_measured) * title_line_h + row_gap
            if title_measured
            else 0.0
        )
        if title_measured:
            content_w = max(content_w, *(m.width for _, m in title_measured))

        card_w = content_w + 2 * pad
        card_h = (
            2 * pad
            + title_h
            + len(measured) * row_h
            + row_gap * max(0, len(measured) - 1)
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            cv.rounded_rect(
                rect,
                radius=9.0,
                fill=_CARD_BG,
                shadow=Shadow(blur=6.0, dy=2.0) if style.shadow else None,
            )
            y = rect.top + pad
            for line, metrics in title_measured:
                cv.text(
                    (rect.left + pad, y + metrics.ascent),
                    line,
                    size=title_size,
                    color=_CARD_TEXT,
                    weight=700,
                )
                y += title_line_h
            if title_measured:
                y += row_gap
            for line, metrics in measured:
                cv.text(
                    (rect.left + pad, y + metrics.ascent),
                    line,
                    size=size,
                    color=_CARD_TEXT,
                    weight=weight,
                )
                y += row_h + row_gap

        return [Cell(card_w, card_h, _draw)]


class Legend(CornerStack):
    """A class-color key: swatch + name rows inside one card.

    Attributes:
        entries: Class names (colored from the palette) or explicit
            ``(name, color)`` pairs.
        title: Optional heading drawn above the rows.

    See `CornerStack` for
    ``corner``/``margin``/``gap``.

    Examples:
        Class names use the active palette; tuples pin an explicit color:

        >>> from luxonis_ml.vizlab import Legend
        >>> legend = Legend(entries=["car", ("road", "#555555")], title="classes")
        >>> len(legend.entries)
        2

    """

    entries: list[str | tuple[str, ColorLike]] = []
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
        title_size = size * 1.05
        pad, row_gap, swatch_gap = 10.0, 6.0, 8.0
        swatch = size
        # Wrap names to keep the card within the canvas; a wrapped name's swatch
        # is drawn on its first line, continuation lines sit under the text.
        name_avail = max(
            1.0,
            canvas.width - 2 * self.margin - 2 * pad - swatch - swatch_gap,
        )

        # (line, color-or-None, metrics); color set only on an entry's first line.
        rows: list[tuple[str, Color | None, TextMetrics]] = []
        for name, color in self._resolved_entries(ctx):
            wrapped = canvas.wrap_text(
                name, size, max_width=name_avail, weight=weight
            ) or [""]
            for i, line in enumerate(wrapped):
                rows.append(
                    (
                        line,
                        color if i == 0 else None,
                        canvas.measure_text(line, size, weight=weight),
                    )
                )
        title_lines = (
            canvas.wrap_text(
                self.title,
                title_size,
                max_width=max(1.0, canvas.width - 2 * self.margin - 2 * pad),
                weight=700,
            )
            if self.title is not None
            else []
        )
        title_measured = [
            (line, canvas.measure_text(line, title_size, weight=700))
            for line in title_lines
        ]
        if not rows and not title_measured:
            return []

        row_h = max((m.height for _, _, m in rows), default=size)
        content_w = max(
            (swatch + swatch_gap + m.width for _, _, m in rows),
            default=0.0,
        )
        title_line_h = max((m.height for _, m in title_measured), default=0.0)
        title_h = (
            len(title_measured) * title_line_h + row_gap
            if title_measured
            else 0.0
        )
        if title_measured:
            content_w = max(content_w, *(m.width for _, m in title_measured))

        card_w = content_w + 2 * pad
        card_h = (
            2 * pad
            + title_h
            + len(rows) * row_h
            + row_gap * max(0, len(rows) - 1)
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            cv.rounded_rect(
                rect,
                radius=9.0,
                fill=_CARD_BG,
                shadow=Shadow(blur=6.0, dy=2.0) if style.shadow else None,
            )
            y = rect.top + pad
            for line, metrics in title_measured:
                cv.text(
                    (rect.left + pad, y + metrics.ascent),
                    line,
                    size=title_size,
                    color=_CARD_TEXT,
                    weight=700,
                )
                y += title_line_h
            if title_measured:
                y += row_gap
            for line, color, metrics in rows:
                if color is not None:
                    sw_top = y + (row_h - swatch) / 2
                    cv.rounded_rect(
                        Rect(
                            rect.left + pad,
                            sw_top,
                            rect.left + pad + swatch,
                            sw_top + swatch,
                        ),
                        radius=3.0,
                        fill=color,
                    )
                cv.text(
                    (
                        rect.left + pad + swatch + swatch_gap,
                        y + metrics.ascent,
                    ),
                    line,
                    size=size,
                    color=_CARD_TEXT,
                    weight=weight,
                )
                y += row_h + row_gap

        return [Cell(card_w, card_h, _draw)]
