"""Small geometry helpers and type aliases shared across annotations.

Internally the library works in pixel coordinates with the origin at the
top-left. Boxes are stored as ``(left, top, right, bottom)`` (``ltrb``); the
public `BBox` accepts other conventions and
converts to this one.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

XY = tuple[float, float]
"""A point as an ``(x, y)`` pixel coordinate."""


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in pixel space (top-left origin).

    Attributes:
        left: Left edge x, in pixels.
        top: Top edge y, in pixels.
        right: Right edge x, in pixels.
        bottom: Bottom edge y, in pixels.

    Examples:
        >>> r = Rect(0.0, 0.0, 10.0, 20.0)
        >>> r.width, r.height, r.area
        (10.0, 20.0, 200.0)
        >>> r.center
        (5.0, 10.0)
        >>> Rect(-5.0, -5.0, 50.0, 50.0).clip_to(20.0, 20.0)
        Rect(left=0.0, top=0.0, right=20.0, bottom=20.0)
        >>> Rect(0.0, 0.0, 5.0, 5.0).union(Rect(3.0, 3.0, 10.0, 8.0))
        Rect(left=0.0, top=0.0, right=10.0, bottom=8.0)
        >>> Rect(0.0, 0.0, 4.0, 4.0).iou(Rect(2.0, 2.0, 6.0, 6.0))
        0.14285714285714285

    """

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        """Width in pixels (never negative)."""
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        """Height in pixels (never negative)."""
        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        """Area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> XY:
        """The rectangle center as an ``(x, y)`` point."""
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    def clip_to(self, width: float, height: float) -> "Rect":
        """Clip the rectangle to a ``[0, width] x [0, height]`` canvas.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        Returns:
            The clipped rectangle.

        """
        return Rect(
            max(0.0, min(self.left, width)),
            max(0.0, min(self.top, height)),
            max(0.0, min(self.right, width)),
            max(0.0, min(self.bottom, height)),
        )

    def union(self, other: "Rect") -> "Rect":
        """Return the smallest rectangle containing both rectangles.

        Args:
            other: The other rectangle.

        Returns:
            The bounding union.

        """
        return Rect(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )

    def intersection(self, other: "Rect") -> "Rect | None":
        """Return the overlapping rectangle, or ``None`` if they are disjoint.

        Args:
            other: The other rectangle.

        Returns:
            The intersection `Rect`, or ``None`` when the rectangles do not
            overlap (touching edges count as no overlap).

        Examples:
            >>> Rect(0.0, 0.0, 4.0, 4.0).intersection(Rect(2.0, 2.0, 6.0, 6.0))
            Rect(left=2.0, top=2.0, right=4.0, bottom=4.0)
            >>> Rect(0.0, 0.0, 1.0, 1.0).intersection(Rect(2.0, 2.0, 3.0, 3.0))

        """
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right, bottom)

    def iou(self, other: "Rect") -> float:
        """Return the intersection-over-union overlap with ``other`` in ``[0, 1]``.

        Args:
            other: The other rectangle.

        Returns:
            The IoU: intersection area divided by union area, or ``0.0`` when the
            rectangles are disjoint or both empty.

        Examples:
            >>> Rect(0.0, 0.0, 2.0, 2.0).iou(Rect(0.0, 0.0, 2.0, 2.0))
            1.0
            >>> Rect(0.0, 0.0, 2.0, 2.0).iou(Rect(5.0, 5.0, 7.0, 7.0))
            0.0

        """
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        inter_area = inter.area
        union = self.area + other.area - inter_area
        return inter_area / union if union > 0.0 else 0.0


def bounding_rect(points: Sequence[XY]) -> Rect:
    """Compute the axis-aligned bounding rectangle of a set of points.

    Args:
        points: A non-empty sequence of ``(x, y)`` points.

    Returns:
        The bounding `Rect`.

    Raises:
        ValueError: If ``points`` is empty.

    Examples:
        >>> bounding_rect([(1.0, 2.0), (5.0, 0.0), (3.0, 9.0)])
        Rect(left=1.0, top=0.0, right=5.0, bottom=9.0)

    """
    if not points:
        raise ValueError("bounding_rect requires at least one point")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return Rect(min(xs), min(ys), max(xs), max(ys))


def arrow_head(
    tip: XY, direction: XY, length: float, width: float
) -> tuple[XY, XY, XY]:
    """Return the three corners of a triangular arrow head.

    The head points along ``direction`` with its apex at ``tip``, so the caller
    draws it as a filled triangle. Shared by every mark that says "this way":
    a `Polyline`'s direction chevrons and an `Arrow`'s heads.

    Args:
        tip: The apex ``(x, y)``, in pixels.
        direction: The direction the head points in; any non-zero length.
        length: Distance from the apex back to the base, in pixels.
        width: Full width of the base, in pixels.

    Returns:
        The apex followed by the two base corners. A zero ``direction`` has no
        orientation to draw, so it collapses to three copies of ``tip`` — a
        degenerate triangle that paints nothing.

    Examples:
        >>> arrow_head((10.0, 0.0), (1.0, 0.0), 4.0, 4.0)
        ((10.0, 0.0), (6.0, 2.0), (6.0, -2.0))
        >>> arrow_head((0.0, 0.0), (0.0, 0.0), 4.0, 4.0)
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    """
    dx, dy = direction
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        return (tip, tip, tip)
    ux, uy = dx / norm, dy / norm
    base = (tip[0] - ux * length, tip[1] - uy * length)
    half = width / 2.0
    return (
        tip,
        (base[0] - uy * half, base[1] + ux * half),
        (base[0] + uy * half, base[1] - ux * half),
    )


def oriented_corners(
    cx: float, cy: float, width: float, height: float, angle_rad: float
) -> tuple[XY, XY, XY, XY]:
    """Return the four corners of an oriented (rotated) rectangle.

    Corners are ordered clockwise from the (pre-rotation) top-left, rotated about
    the center. The rotation is clockwise on a top-left-origin, y-down canvas.

    Args:
        cx: Center x, in pixels.
        cy: Center y, in pixels.
        width: Box width, in pixels.
        height: Box height, in pixels.
        angle_rad: Rotation about the center, in radians.

    Returns:
        The four ``(x, y)`` corners.

    Examples:
        >>> oriented_corners(0.0, 0.0, 2.0, 4.0, 0.0)
        ((-1.0, -2.0), (1.0, -2.0), (1.0, 2.0), (-1.0, 2.0))

    """
    hw, hh = width / 2.0, height / 2.0
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    def _rot(dx: float, dy: float) -> XY:
        return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)

    return (_rot(-hw, -hh), _rot(hw, -hh), _rot(hw, hh), _rot(-hw, hh))
