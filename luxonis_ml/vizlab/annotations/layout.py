"""Collision-aware placement of label chips within a single render pass.

One `LabelLayout` is shared across every annotation in a render (threaded
through `RenderContext`). Each label proposes a few candidate positions around
its box/mask; the layout picks the one that overlaps previously placed chips the
least and records it, so labels of different instances avoid landing on top of
one another — which matters most when boxes overlap (e.g. a mixup of two scenes).

When a scene is dense enough that every position adjacent to a box is already
taken, the layout falls back to a ring of positions pushed progressively away
from the box, trading a little distance for a lot less overlap. A chip that ends
up detached from its box reports a `Placement.leader` anchor so the caller can
draw a thin connector line back to it — the pushed-out label still reads as
belonging to its box instead of floating free.
"""

import math
from dataclasses import dataclass

from luxonis_ml.vizlab.geometry import XY, Rect
from luxonis_ml.vizlab.style import LabelPlacement

#: Cost (in chip-overlap-area equivalents) charged per pixel a chip is pushed
#: away from its box. Overlap is measured in px², displacement in px, so this
#: weight sets the exchange rate: a label leaves its box's side only when doing
#: so removes more chip-on-chip overlap than the push itself costs.
_DISPLACEMENT_WEIGHT = 6.0

#: A chip whose gap to its box exceeds this (px) is considered detached and gets
#: a leader line; at or below it the chip still visually touches its box.
_LEADER_MIN_GAP = 6.0


def _overlap_area(a: Rect, b: Rect) -> float:
    """Area of the intersection of two rectangles (``0`` if disjoint).

    Args:
        a: First rectangle.
        b: Second rectangle.

    Returns:
        The overlap area in square pixels.

    Examples:
        >>> _overlap_area(
        ...     Rect(0.0, 0.0, 10.0, 10.0), Rect(5.0, 5.0, 15.0, 15.0)
        ... )
        25.0
        >>> _overlap_area(
        ...     Rect(0.0, 0.0, 10.0, 10.0), Rect(20.0, 20.0, 30.0, 30.0)
        ... )
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


def _ring_candidates(region: Rect, width: float, height: float) -> list[XY]:
    """Propose chip positions pushed progressively away from ``region``.

    A fallback for dense scenes: when every adjacent slot is taken, these let a
    chip step out above, below, or to either side of its box in widening rings so
    it can escape the pile-up. Distances scale with the chip height so labels
    stack a line apart.

    Args:
        region: The box/extent the chip labels.
        width: Chip width in pixels.
        height: Chip height in pixels.

    Returns:
        Candidate ``(x, y)`` top-left positions, nearer rings first.

    """
    cx, cy = region.center
    left, right = region.left, region.right - width
    mid_x, mid_y = cx - width / 2, cy - height / 2
    out: list[XY] = []
    for gap in (height, 2.0 * height, 3.2 * height):
        above, below = region.top - height - gap, region.bottom + gap
        out += [
            (left, above),
            (mid_x, above),
            (right, above),
            (left, below),
            (mid_x, below),
            (right, below),
        ]
        far_right, far_left = region.right + gap, region.left - width - gap
        out += [
            (far_right, region.top),
            (far_right, mid_y),
            (far_left, region.top),
            (far_left, mid_y),
        ]
    return out


def _gap(a: Rect, b: Rect) -> float:
    """Edge-to-edge distance between two rectangles (``0`` when they touch/overlap)."""
    dx = max(0.0, a.left - b.right, b.left - a.right)
    dy = max(0.0, a.top - b.bottom, b.top - a.bottom)
    return math.hypot(dx, dy)


def _clamp_point(rect: Rect, x: float, y: float) -> XY:
    """Return the point on/inside ``rect`` closest to ``(x, y)``."""
    return (
        max(rect.left, min(x, rect.right)),
        max(rect.top, min(y, rect.bottom)),
    )


@dataclass(frozen=True)
class Placement:
    """A chosen chip position and, when detached, the box anchor to lead back to.

    Attributes:
        rect: The chip's placed rectangle in canvas pixels.
        leader: The point on the labeled box to draw a connector line from when
            the chip was pushed clear of it, or ``None`` when the chip still sits
            against its box (no leader needed).

    """

    rect: Rect
    leader: XY | None = None


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
        self.overlay_positions: dict[int, list[Rect]] = {}

    def _clamp(self, x: float, y: float, width: float, height: float) -> Rect:
        """Clamp a chip of the given size fully inside the canvas."""
        cx = max(0.0, min(x, self.width - width))
        cy = max(0.0, min(y, self.height - height))
        return Rect(cx, cy, cx + width, cy + height)

    def place(
        self,
        width: float,
        height: float,
        candidates: list[XY],
        *,
        region: Rect | None = None,
    ) -> Placement:
        """Choose and record the best position for a new chip.

        A candidate that sits against the box with zero overlap wins outright.
        Otherwise the cost is ``overlap_area + weight · displacement`` — so a chip
        stays by its box through a little overlap, but when the box's whole
        neighborhood is crowded it steps out along the `_ring_candidates` ring to
        somewhere clearer. A chip that lands clear of its box carries a
        `Placement.leader` back to it.

        Args:
            width: Chip width in pixels.
            height: Chip height in pixels.
            candidates: Adjacent top-left positions to try, in preference order.
            region: The labeled box, enabling the displaced-ring fallback and the
                leader line. ``None`` keeps the plain least-overlap behavior.

        Returns:
            The chosen `Placement` (its rectangle is already recorded).

        """
        options = candidates
        if region is not None:
            options = [*candidates, *_ring_candidates(region, width, height)]
        best: Rect | None = None
        best_cost = float("inf")
        best_gap = 0.0
        for x, y in options:
            rect = self._clamp(x, y, width, height)
            overlap = sum(_overlap_area(rect, other) for other in self.placed)
            gap = _gap(rect, region) if region is not None else 0.0
            cost = overlap + _DISPLACEMENT_WEIGHT * gap
            if cost < best_cost:
                best, best_cost, best_gap = rect, cost, gap
                if cost == 0.0:
                    break
        assert best is not None  # options is always non-empty
        self.placed.append(best)
        leader: XY | None = None
        if region is not None and best_gap > _LEADER_MIN_GAP:
            leader = _clamp_point(region, *best.center)
        return Placement(best, leader)

    def reserve(self, rect: Rect) -> None:
        """Record a fixed chip position (e.g. a corner tag) as occupied.

        Args:
            rect: The chip rectangle to mark as taken.

        """
        self.placed.append(rect)
