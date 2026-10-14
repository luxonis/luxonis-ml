"""Shared machinery for image-level, corner-stacked overlays.

Classification tags, legends, and captions are all *chrome*: image-level cards
stacked in a corner, reserved before spatial labels are placed and drawn on top of
everything (``OVERLAY = True``). `CornerStack` captures that shared behavior;
each subclass only supplies its `Cell` list.
"""

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.style import Style, Theme

from .base import Annotation, RenderContext
from .chip import chip_size, draw_chip

_WHITE = Color(255, 255, 255)

#: Shared brand card chrome, so every stacked card (legend, info card,
#: distribution panel) reads as one on-brand family. Sourced from
#: `luxonis_ml.utils.color.brand`. These are the *dark-mode* defaults; a render
#: on a light theme resolves the light variants via `resolve_chrome`.
CARD_BG = brand.CARD_BG
CARD_TEXT = brand.CARD_TEXT


def resolve_chrome(ctx: RenderContext) -> brand.Chrome:
    """Resolve the card/panel chrome for the context's theme background.

    Cards adapt to the active theme: navy fills with near-white text on a dark
    background, white fills with deep-purple text on a light one. Falls back to
    the dark chrome when no theme is set.

    Args:
        ctx: The current render context (its ``theme`` supplies the background).

    Returns:
        The `Chrome` whose colors suit the composite background.

    """
    background = (
        ctx.theme.background if ctx.theme is not None else brand.BACKGROUND
    )
    return brand.chrome_for(background)


def _on_dark(surface: Color) -> bool:
    """Whether ``surface`` is dark enough to want light text/outlines on it."""
    return surface.readable_text_color().r > 200


def shade_outline(fill: Color, surface: Color) -> Color:
    """Return a color-matched edge for a filled shape.

    A lighter (on a dark ``surface``) or darker (on a light one) shade of
    ``fill``, so wedges/bars/segments get a crisp rim without a foreign color.

    Args:
        fill: The fill color of the shape.
        surface: The color of the surface behind the shape.

    Returns:
        The edge color.

    """
    return fill.lighten(0.45) if _on_dark(surface) else fill.darken(0.4)


def swatch_outline(surface: Color) -> Color:
    """Return a uniform light-or-dark hairline for a color swatch.

    Unlike `shade_outline` this is not color-matched: every swatch on a card gets
    the same subtle light (on dark) or dark (on light) ring so the key reads as a
    set. The choice follows ``surface``.

    Args:
        surface: The color of the card behind the swatch.

    Returns:
        A translucent white on a dark surface, a translucent black otherwise.

    """
    return (
        Color(255, 255, 255, 150) if _on_dark(surface) else Color(0, 0, 0, 130)
    )


CellDraw = Callable[[Canvas, Rect], None]
"""Draws a single stacked cell's content into its placed rectangle."""


#: The canvas `CornerStack.content_size` measures on by default: large enough
#: that no cell wraps its text or flows into columns.
_MEASURE_CANVAS = (100_000, 100_000)


class Corner(Enum):
    """Which image corner an overlay stack is anchored to."""

    #: The upper-left corner.
    TOP_LEFT = "top-left"
    #: The upper-right corner.
    TOP_RIGHT = "top-right"
    #: The lower-left corner.
    BOTTOM_LEFT = "bottom-left"
    #: The lower-right corner.
    BOTTOM_RIGHT = "bottom-right"


@dataclass(frozen=True)
class Cell:
    """One item in a corner stack: its size and how to draw it.

    Attributes:
        width: Cell width in pixels.
        height: Cell height in pixels.
        draw: Callback that renders the cell into its placed rectangle.

    """

    width: float
    height: float
    draw: CellDraw


def _block_bounds(rects: list[Rect]) -> tuple[float, float, float, float]:
    """Return the ``(left, right, top, bottom)`` bounds enclosing ``rects``."""
    return (
        min(rect.left for rect in rects),
        max(rect.right for rect in rects),
        min(rect.top for rect in rects),
        max(rect.bottom for rect in rects),
    )


def _slide_clear(
    occupied: list[Rect],
    y: float,
    block_height: float,
    top: bool,
    gap: float,
) -> float:
    """Slide ``y`` along until the block clears every conflicting reserved rect."""
    while True:
        candidate_top = y if top else y - block_height
        candidate_bottom = candidate_top + block_height
        conflicts = [
            rect
            for rect in occupied
            if rect.bottom + gap > candidate_top
            and rect.top - gap < candidate_bottom
        ]
        if not conflicts:
            return y
        y = (
            max(rect.bottom for rect in conflicts) + gap
            if top
            else min(rect.top for rect in conflicts) - gap
        )


def _offset_positioned(
    positioned: list[tuple[Rect, CellDraw]], offset: float
) -> list[tuple[Rect, CellDraw]]:
    """Shift every placed rect vertically by ``offset``."""
    return [
        (
            Rect(
                rect.left, rect.top + offset, rect.right, rect.bottom + offset
            ),
            draw,
        )
        for rect, draw in positioned
    ]


def chip_cell(canvas: Canvas, text: str, color: Color, style: Style) -> Cell:
    """Build a stack cell that draws a single filled label chip.

    Args:
        canvas: The canvas (for measuring the chip).
        text: The chip text.
        color: The chip fill color.
        style: The resolved style.

    Returns:
        A `Cell` that draws the chip at its placed rectangle.

    """
    width, height, _ = chip_size(canvas, text, style)

    def _draw(cv: Canvas, rect: Rect) -> None:
        draw_chip(cv, (rect.left, rect.top), text, color, style)

    return Cell(width, height, _draw)


#: The gap between the pieces of an inline key, and between its rows, as a
#: share of the type size.
_INLINE_GAP = 0.6
_INLINE_ROW_GAP = 0.35


@dataclass(frozen=True)
class InlinePiece:
    """One part of an inline key, a run of text or a graphic, kept whole.

    Attributes:
        width: The piece width in pixels.
        height: The piece height in pixels.
        draw: Draws the piece with its left edge at ``x`` and its vertical
            middle at ``y``: ``draw(canvas, x, y)``.

    """

    width: float
    height: float
    draw: Callable[[Canvas, float, float], None]


def text_piece(
    canvas: Canvas,
    text: str,
    size: float,
    color: Color,
    *,
    weight: int = 400,
    mono: bool = False,
) -> InlinePiece:
    """Build an inline piece of markup text.

    Args:
        canvas: The canvas to measure on.
        text: The text, as inline markup.
        size: Type size in pixels.
        color: Text color.
        weight: Font weight.
        mono: Whether to use the monospace family.

    Returns:
        The piece, sized to the text.

    """
    metrics = canvas.measure_markup(text, size, weight=weight, mono=mono)

    def draw(cv: Canvas, x: float, middle: float) -> None:
        cv.markup(
            (x, middle + (metrics.ascent - metrics.descent) / 2),
            text,
            size=size,
            color=color,
            weight=weight,
            mono=mono,
        )

    return InlinePiece(metrics.width, metrics.height, draw)


def inline_cell(
    pieces: "Sequence[InlinePiece]",
    size: float,
    avail: float,
    *,
    hang: bool = False,
) -> Cell | None:
    """Flow ``pieces`` into rows no wider than ``avail``, as one cell.

    Args:
        pieces: The pieces, in reading order.
        size: Type size in pixels; the gaps follow it.
        avail: The widest a row may be, in pixels. A piece wider than that
            gets a row of its own.
        hang: Start each continuation row under the second piece, so a
            leading title stands out (a legend); otherwise rows start at
            the left edge.

    Returns:
        The cell, or ``None`` without pieces.

    """
    if not pieces:
        return None
    gap = size * _INLINE_GAP
    row_gap = size * _INLINE_ROW_GAP
    row_h = max(piece.height for piece in pieces)
    indent = pieces[0].width + gap if hang and len(pieces) > 1 else 0.0
    placed: list[tuple[InlinePiece, float, int]] = []
    x, row, width = 0.0, 0, 0.0
    for index, piece in enumerate(pieces):
        if index and x + piece.width > avail and x > indent:
            row += 1
            x = indent
        placed.append((piece, x, row))
        width = max(width, x + piece.width)
        x += piece.width + gap
    height = (row + 1) * row_h + row * row_gap

    def _draw(cv: Canvas, rect: Rect) -> None:
        for piece, left, line in placed:
            middle = rect.top + line * (row_h + row_gap) + row_h / 2
            piece.draw(cv, rect.left + left, middle)

    return Cell(width, height, _draw)


class CornerStack(Annotation):
    """Base for image-level overlays that stack cells in a corner.

    Corner stacks are drawn after spatial annotations. Their occupied rectangles
    are reserved before box and mask labels are placed, and separate overlays in
    the same corner are offset so every card remains visible.

    Attributes:
        corner: Which corner to anchor the stack to.
        margin: Distance from the image edges, in pixels.
        gap: Vertical gap between cells, in pixels.

    """

    corner: Corner = Corner.TOP_LEFT
    margin: float = 14.0
    gap: float = 8.0

    #: Image-level chrome. It reserves its space before everything else and
    #: draws on top of everything else.
    OVERLAY: ClassVar[bool] = True

    @abstractmethod
    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        """Return the cells to stack, top to bottom."""

    def resolve_color(self, ctx: RenderContext) -> Color:
        """Return white, because an overlay colors its own content.

        Args:
            ctx: The current render context.

        Returns:
            White.

        """
        return _WHITE

    def extent(self) -> Rect | None:
        """Return ``None``, because an image-level overlay has no local extent.

        Returns:
            Always ``None``.

        """
        return None

    def content_size(
        self,
        *,
        theme: Theme | None = None,
        style_scale: float = 1.0,
        canvas_size: tuple[int, int] = _MEASURE_CANVAS,
    ) -> tuple[float, float]:
        """Return the natural ``(width, height)`` of this stack's cells, in pixels.

        Measures without drawing, so a caller can size a standalone canvas to fit
        the overlay (e.g. render a legend or a distribution panel on its own).
        Width is the widest cell; height sums the cells plus the inter-cell gaps.

        Args:
            theme: Theme supplying default style/palette during measurement.
            style_scale: Style scale to measure at; keep at ``1.0`` for canvases
                below the style-scale reference size.
            canvas_size: The ``(width, height)`` of the canvas the stack would
                draw on. A cell fits its text to it: a `Legend` wraps its names
                to the width, and flows into columns when it is taller than the
                height. The default is large enough for neither.

        Returns:
            The content's ``(width, height)``; ``(0.0, 0.0)`` when it has no cells.

        """
        # An SVG canvas measures text like a raster one, without allocating
        # pixels for its size.
        ctx = RenderContext(
            canvas=Canvas.svg(*canvas_size),
            theme=theme,
            style_scale=style_scale,
        )
        cells = self._cells(ctx, self.resolve_style(ctx))
        if not cells:
            return (0.0, 0.0)
        width = max(cell.width for cell in cells)
        height = sum(cell.height for cell in cells) + self.gap * (
            len(cells) - 1
        )
        return (width, height)

    def _positioned(
        self, ctx: RenderContext, style: Style
    ) -> list[tuple[Rect, CellDraw]]:
        """Place each cell at the chosen corner and return ``(rect, draw)`` pairs."""
        cells = self._cells(ctx, style)
        if not cells:
            return []
        canvas = ctx.canvas
        top = self.corner in (Corner.TOP_LEFT, Corner.TOP_RIGHT)
        left = self.corner in (Corner.TOP_LEFT, Corner.BOTTOM_LEFT)
        total = sum(c.height for c in cells) + self.gap * (len(cells) - 1)
        # A stack taller than the image keeps its title in view and lets its
        # last rows run off the bottom edge.
        y = (
            self.margin
            if top
            else max(self.margin, canvas.height - self.margin - total)
        )
        placed: list[tuple[Rect, CellDraw]] = []
        for cell in cells:
            x = (
                self.margin
                if left
                else canvas.width - self.margin - cell.width
            )
            placed.append(
                (Rect(x, y, x + cell.width, y + cell.height), cell.draw)
            )
            y += cell.height + self.gap
        return placed

    def _avoid_reserved(
        self,
        positioned: list[tuple[Rect, CellDraw]],
        reserved: list[Rect],
        size: tuple[float, float],
    ) -> list[tuple[Rect, CellDraw]]:
        """Shift a corner stack past overlays already anchored there.

        The stack first slides along its edge, away from its corner. On a
        short image that can push it past the far edge, where it would not
        be seen; it then stands beside the overlays it meets instead, and
        where that does not fit either, it stays put and overlaps them.
        """
        if not positioned or not reserved:
            return positioned
        left, right, block_top, block_bottom = _block_bounds(
            [rect for rect, _ in positioned]
        )
        top = self.corner in (Corner.TOP_LEFT, Corner.TOP_RIGHT)
        occupied = [
            rect
            for rect in reserved
            if rect.right > left and rect.left < right
        ]
        start_y = block_top if top else block_bottom
        y = _slide_clear(
            occupied, start_y, block_bottom - block_top, top, self.gap
        )
        offset = (y - block_top) if top else (y - block_bottom)
        if offset == 0:
            return positioned
        width, height = size
        if block_top + offset >= 0 and block_bottom + offset <= height:
            return _offset_positioned(positioned, offset)
        met = [
            rect
            for rect in occupied
            if rect.bottom + self.gap > block_top
            and rect.top - self.gap < block_bottom
        ]
        if self.corner in (Corner.TOP_LEFT, Corner.BOTTOM_LEFT):
            shift = max(rect.right for rect in met) + self.gap - left
        else:
            shift = min(rect.left for rect in met) - self.gap - right
        if left + shift >= 0 and right + shift <= width:
            return [
                (
                    Rect(
                        rect.left + shift,
                        rect.top,
                        rect.right + shift,
                        rect.bottom,
                    ),
                    draw,
                )
                for rect, draw in positioned
            ]
        return positioned

    def reserve(self, ctx: RenderContext) -> None:
        """Reserve each cell's rect so spatial labels avoid the corner.

        Args:
            ctx: The current render context, whose layout records the
                reservations.

        """
        if ctx.layout is None:
            return
        positioned = self._avoid_reserved(
            self._positioned(ctx, self.resolve_style(ctx)),
            ctx.layout.placed,
            (ctx.canvas.width, ctx.canvas.height),
        )
        ctx.layout.overlay_positions[id(self)] = [
            rect for rect, _ in positioned
        ]
        for rect, _ in positioned:
            ctx.layout.reserve(rect)

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Draw each stacked cell into its placed rectangle.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: Unused (cells color their own content).

        """
        positioned = self._positioned(ctx, style)
        if ctx.layout is not None:
            reserved = ctx.layout.overlay_positions.get(id(self))
            if reserved is not None and len(reserved) == len(positioned):
                positioned = [
                    (rect, draw)
                    for rect, (_, draw) in zip(
                        reserved, positioned, strict=True
                    )
                ]
        for rect, draw_cell in positioned:
            draw_cell(ctx.canvas, rect)


class ColorKey(CornerStack):
    """A corner stack that says what the colors of a field or a mask mean.

    The base of `ColorBar`, `FlowWheel`, `NormalKey` and `Legend`. A key is
    drawn as a card in a corner of the image, or ``inline``: on one line,
    without a card, for a strip under the image (see
    `luxonis_ml.vizlab.layout.with_key_strip`), where a card would take
    three times the height.

    Attributes:
        title: Heading of the key; ``None`` draws none.
        inline: Draw the title and the scale side by side, without a card.
            The text has no card behind it, so an inline key is for a plain
            strip, not for drawing over an image.

    """

    title: str | None = None
    inline: bool = False

    #: Whether an inline key's continuation rows start under its second
    #: piece rather than at the left edge.
    INLINE_HANG: ClassVar[bool] = False

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        """Return the key's card, or its inline cell."""
        if not self.inline:
            return self._card_cells(ctx, style)
        cell = inline_cell(
            self._inline_pieces(ctx, style),
            style.font_size,
            ctx.canvas.width - 2 * self.margin,
            hang=self.INLINE_HANG,
        )
        return [] if cell is None else [cell]

    @abstractmethod
    def _card_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        """Return the cells of the key drawn as a card."""

    @abstractmethod
    def _inline_pieces(
        self, ctx: RenderContext, style: Style
    ) -> list[InlinePiece]:
        """Return the pieces of the key drawn inline, in reading order."""
