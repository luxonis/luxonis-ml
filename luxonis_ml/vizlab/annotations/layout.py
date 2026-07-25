"""Collision-aware placement of label chips within a single render pass.

One `LabelLayout` is shared across every annotation in a render (threaded
through `RenderContext`). Each label proposes a
few candidate positions around its box/mask; the layout picks the one that
overlaps previously placed chips the least and records it, so labels of different
instances avoid landing on top of one another — which matters most when boxes
overlap (e.g. a mixup of two scenes).
"""

from luxonis_ml.vizlab.geometry import XY, Rect
from luxonis_ml.vizlab.style import LabelPlacement


def _overlap_area(a: Rect, b: Rect) -> float:
    """Area of the intersection of two rectangles (``0`` if disjoint).

    Args:
        a: First rectangle.
        b: Second rectangle.

    Returns:
        The overlap area in square pixels.

    Examples:
        >>> _overlap_area(Rect(0.0, 0.0, 10.0, 10.0), Rect(5.0, 5.0, 15.0, 15.0))
        25.0
        >>> _overlap_area(Rect(0.0, 0.0, 10.0, 10.0), Rect(20.0, 20.0, 30.0, 30.0))
        0.0

    """
    dx = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    dy = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    return dx * dy


def label_candidates(
    region: Rect, width: float, height: float, placement: LabelPlacement
) -> list[XY]:
    """Propose chip top-left positions around ``region``, best first.

    Args:
        region: The annotation's box/extent the chip labels.
        width: Chip width in pixels.
        height: Chip height in pixels.
        placement: Preferred placement; ``INSIDE`` tries inside the box first.

    Returns:
        Candidate ``(x, y)`` top-left positions in preference order.

    """
    cx = region.center[0]
    above = [
        (region.left, region.top - height),
        (region.right - width, region.top - height),
        (cx - width / 2, region.top - height),
    ]
    inside = [
        (region.left, region.top),
        (region.right - width, region.top),
    ]
    below = [
        (region.left, region.bottom),
        (region.right - width, region.bottom),
    ]
    if placement is LabelPlacement.INSIDE:
        return inside + above + below
    return above + inside + below


class LabelLayout:
    """Tracks placed label chips and places new ones to minimize overlap."""

    def __init__(self, width: int, height: int) -> None:
        """Create a layout for a ``width`` x ``height`` canvas.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        """
        self.width = width
        self.height = height
        self.placed: list[Rect] = []

    def _clamp(self, x: float, y: float, width: float, height: float) -> Rect:
        """Clamp a chip of the given size fully inside the canvas."""
        cx = max(0.0, min(x, self.width - width))
        cy = max(0.0, min(y, self.height - height))
        return Rect(cx, cy, cx + width, cy + height)

    def place(self, width: float, height: float, candidates: list[XY]) -> Rect:
        """Choose and record the best position for a new chip.

        The first candidate with zero overlap wins; otherwise the one with the
        least total overlap against already-placed chips is used.

        Args:
            width: Chip width in pixels.
            height: Chip height in pixels.
            candidates: Top-left positions to try, in preference order.

        Returns:
            The chosen chip `Rect`, already recorded.

        """
        best: Rect | None = None
        best_cost = float("inf")
        for x, y in candidates:
            rect = self._clamp(x, y, width, height)
            cost = sum(_overlap_area(rect, other) for other in self.placed)
            if cost < best_cost:
                best, best_cost = rect, cost
                if cost == 0.0:
                    break
        assert best is not None  # candidates is always non-empty
        self.placed.append(best)
        return best

    def reserve(self, rect: Rect) -> None:
        """Record a fixed chip position (e.g. a corner tag) as occupied.

        Args:
            rect: The chip rectangle to mark as taken.

        """
        self.placed.append(rect)
