"""A `HitMap`: which `Tooltip` sits under a given pixel.

Rendering an image (or a composite of images) can emit a `HitMap` — the list of
``(rect, tooltip)`` pairs for every annotation that carries hover content, in the
final pixel coordinates of the rendered frame. An interactive viewer queries it
with `hit` on mouse-move; `offset`/`scaled` transform a map when its frame is
placed into a larger composite or resized for display.
"""

from dataclasses import dataclass, field

from .geometry import Rect
from .tooltip import Tooltip


@dataclass
class HitMap:
    """Hover hit-test entries in final frame pixels.

    Each entry is a ``(rect, tooltip)`` pair; `hit` returns the tooltip of the
    smallest rectangle containing a point, so a small box nested inside a large
    one still wins the hover.

    Attributes:
        items: The ``(rect, tooltip)`` entries, in draw order.

    """

    items: list[tuple[Rect, Tooltip]] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "HitMap":
        """Return a hit map with no entries."""
        return cls([])

    def hit(self, x: float, y: float) -> Tooltip | None:
        """Return the tooltip of the smallest box containing ``(x, y)``.

        Args:
            x: Point x in frame pixels.
            y: Point y in frame pixels.

        Returns:
            The matching `Tooltip`, or ``None`` when no box contains the point.
            On ties the first (earliest-drawn) box wins.

        """
        best: Tooltip | None = None
        best_area: float | None = None
        for rect, tooltip in self.items:
            if rect.left <= x <= rect.right and rect.top <= y <= rect.bottom:
                area = rect.area
                if best_area is None or area < best_area:
                    best, best_area = tooltip, area
        return best

    def offset(self, dx: float, dy: float) -> "HitMap":
        """Return a copy with every rectangle shifted by ``(dx, dy)`` pixels."""
        return HitMap(
            [
                (Rect(r.left + dx, r.top + dy, r.right + dx, r.bottom + dy), t)
                for r, t in self.items
            ]
        )

    def scaled(self, factor: float) -> "HitMap":
        """Return a copy with every rectangle scaled about the origin."""
        return HitMap(
            [
                (
                    Rect(
                        r.left * factor,
                        r.top * factor,
                        r.right * factor,
                        r.bottom * factor,
                    ),
                    t,
                )
                for r, t in self.items
            ]
        )

    def merge(self, other: "HitMap") -> "HitMap":
        """Return a new map with this map's entries followed by ``other``'s."""
        return HitMap([*self.items, *other.items])

    def __or__(self, other: "HitMap") -> "HitMap":
        """``self | other`` — shorthand for `merge`."""
        return self.merge(other)
