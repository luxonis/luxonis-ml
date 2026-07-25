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

from luxonis_ml.vizlab.canvas import Canvas
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.style import Style

from .base import Annotation, RenderContext
from .chip import chip_size, draw_chip

_WHITE = Color(255, 255, 255)

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

        rects = [rect for rect, _ in positioned]
        block_left = min(rect.left for rect in rects)
        block_right = max(rect.right for rect in rects)
        block_top = min(rect.top for rect in rects)
        block_bottom = max(rect.bottom for rect in rects)
        block_height = block_bottom - block_top
        top = self.corner in (Corner.TOP_LEFT, Corner.TOP_RIGHT)
        y = block_top if top else block_bottom

        def horizontally_overlaps(rect: Rect) -> bool:
            return rect.right > block_left and rect.left < block_right

        occupied = [rect for rect in reserved if horizontally_overlaps(rect)]
        while True:
            candidate_top = y if top else y - block_height
            candidate_bottom = candidate_top + block_height
            conflicts = [
                rect
                for rect in occupied
                if rect.bottom + self.gap > candidate_top
                and rect.top - self.gap < candidate_bottom
            ]
            if not conflicts:
                break
            y = (
                max(rect.bottom for rect in conflicts) + self.gap
                if top
                else min(rect.top for rect in conflicts) - self.gap
            )

        offset = (y - block_top) if top else (y - block_bottom)
        if offset == 0:
            return positioned
        return [
            (
                Rect(
                    rect.left,
                    rect.top + offset,
                    rect.right,
                    rect.bottom + offset,
                ),
                draw,
            )
            for rect, draw in positioned
        ]

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
