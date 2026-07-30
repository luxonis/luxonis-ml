"""Shared machinery for image-level, corner-stacked overlays.

Classification tags, legends, and captions are all *chrome*: image-level cards
stacked in a corner, reserved before spatial labels are placed and drawn on top of
everything (``OVERLAY = True``). `CornerStack` captures that shared behavior;
each subclass only supplies its `Cell` list.
"""

from abc import abstractmethod
from collections.abc import Callable
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
    """
    return fill.lighten(0.45) if _on_dark(surface) else fill.darken(0.4)


def swatch_outline(surface: Color) -> Color:
    """Return a uniform light-or-dark hairline for a color swatch.

    Unlike `shade_outline` this is not color-matched: every swatch on a card gets
    the same subtle light (on dark) or dark (on light) ring so the key reads as a
    set. The choice follows ``surface``.
    """
    return (
        Color(255, 255, 255, 150) if _on_dark(surface) else Color(0, 0, 0, 130)
    )


CellDraw = Callable[[Canvas, Rect], None]
"""Draws a single stacked cell's content into its placed rectangle."""


class Corner(Enum):
    """Which image corner an overlay stack is anchored to."""

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
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

    #: Image-level chrome: reserved before, and drawn on top of, everything else.
    OVERLAY: ClassVar[bool] = True

    @abstractmethod
    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        """Return the cells to stack, top to bottom."""

    def resolve_color(self, ctx: RenderContext) -> Color:
        """Overlays color their own content, so no single color is resolved."""
        return _WHITE

    def extent(self) -> Rect | None:
        """Image-level overlays have no local extent."""
        return None

    def content_size(
        self, *, theme: Theme | None = None, style_scale: float = 1.0
    ) -> tuple[float, float]:
        """Return the natural ``(width, height)`` of this stack's cells, in pixels.

        Measures without drawing, so a caller can size a standalone canvas to fit
        the overlay (e.g. render a legend or a distribution panel on its own).
        Width is the widest cell; height sums the cells plus the inter-cell gaps.

        Args:
            theme: Theme supplying default style/palette during measurement.
            style_scale: Style scale to measure at; keep at ``1.0`` for canvases
                below the style-scale reference size.

        Returns:
            The content's ``(width, height)``; ``(0.0, 0.0)`` when it has no cells.

        """
        ctx = RenderContext(
            canvas=Canvas.blank(1, 1), theme=theme, style_scale=style_scale
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
        y = self.margin if top else canvas.height - self.margin - total
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
    ) -> list[tuple[Rect, CellDraw]]:
        """Shift a corner stack past overlays already anchored there."""
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
        return _offset_positioned(positioned, offset)

    def reserve(self, ctx: RenderContext) -> None:
        """Reserve each cell's rect so spatial labels avoid the corner."""
        if ctx.layout is None:
            return
        positioned = self._avoid_reserved(
            self._positioned(ctx, self.resolve_style(ctx)),
            ctx.layout.placed,
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
