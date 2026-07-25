"""Image-level captions and a class-color legend.

`Caption` draws a short line of text as a card in a corner (optionally
larger, as a title). `Legend` draws a color key — a stack of swatch + name
rows inside a single card. Both are corner-stacked overlays (drawn on top of
everything, and reserved so box labels avoid them).
"""

from luxonis_ml.vizlab.canvas import Canvas, Shadow
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.style import Style

from .base import RenderContext
from .overlay import Cell, Corner, CornerStack, chip_cell

_CAPTION_BG = Color(22, 22, 26, 235)
_CARD_BG = Color(22, 22, 26, 214)
_CARD_TEXT = Color(238, 238, 240)


class Caption(CornerStack):
    """A short line of text drawn as a card in a corner.

    Attributes:
        text: The caption text (single line).
        background: Card fill color; the text uses a readable contrast of it.
        title: Draw larger and bolder, as a title.

    See `CornerStack` for
    ``corner``/``margin``/``gap``.

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
        return [
            chip_cell(
                ctx.canvas, self.text, Color.parse(self.background), cap_style
            )
        ]


class InfoCard(CornerStack):
    """A titled card of plain text rows, e.g. per-object metadata.

    Like `Legend` but without color swatches: a rounded card with an optional
    heading and one text row per line. Used to surface annotation metadata that
    has no bounding box to anchor a hover tooltip to.

    Attributes:
        rows: The text lines to show, top to bottom.
        title: Optional heading drawn above the rows.

    See `CornerStack` for ``corner``/``margin``/``gap``.

    """

    rows: list[str] = []
    title: str | None = None
    corner: Corner = Corner.TOP_LEFT

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        size, weight = style.font_size, style.font_weight
        pad, row_gap = 10.0, 6.0

        measured = [
            (text, canvas.measure_text(text, size, weight=weight))
            for text in self.rows
        ]
        title_metrics = (
            canvas.measure_text(self.title, size * 1.05, weight=700)
            if self.title is not None
            else None
        )
        if not measured and title_metrics is None:
            return []

        row_h = max((m.height for _, m in measured), default=size)
        content_w = max((m.width for _, m in measured), default=0.0)
        title_h = title_metrics.height + row_gap if title_metrics else 0.0
        if title_metrics is not None:
            content_w = max(content_w, title_metrics.width)

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
            if self.title is not None and title_metrics is not None:
                cv.text(
                    (rect.left + pad, y + title_metrics.ascent),
                    self.title,
                    size=size * 1.05,
                    color=_CARD_TEXT,
                    weight=700,
                )
                y += title_h
            for text, metrics in measured:
                cv.text(
                    (rect.left + pad, y + metrics.ascent),
                    text,
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
        pad, row_gap, swatch_gap = 10.0, 6.0, 8.0
        swatch = size

        items = self._resolved_entries(ctx)
        rows = [
            (name, color, canvas.measure_text(name, size, weight=weight))
            for name, color in items
        ]
        if not rows and self.title is None:
            return []

        row_h = max((m.height for _, _, m in rows), default=size)
        content_w = max(
            (swatch + swatch_gap + m.width for _, _, m in rows), default=0.0
        )
        title_metrics = (
            canvas.measure_text(self.title, size * 1.05, weight=700)
            if self.title is not None
            else None
        )
        title_h = title_metrics.height + row_gap if title_metrics else 0.0
        if title_metrics is not None:
            content_w = max(content_w, title_metrics.width)

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
            if self.title is not None and title_metrics is not None:
                cv.text(
                    (rect.left + pad, y + title_metrics.ascent),
                    self.title,
                    size=size * 1.05,
                    color=_CARD_TEXT,
                    weight=700,
                )
                y += title_h
            for name, color, metrics in rows:
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
                    name,
                    size=size,
                    color=_CARD_TEXT,
                    weight=weight,
                )
                y += row_h + row_gap

        return [Cell(card_w, card_h, _draw)]
