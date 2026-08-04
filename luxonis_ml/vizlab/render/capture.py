"""Interaction maps: what sits under a given pixel.

Rendering an image (or a composite of images) can emit three parallel maps, each
a list of ``(rect, payload)`` pairs in the final pixel coordinates of the
rendered frame:

- a `HitMap` of hover `Tooltip` content, for the annotations that carry any;
- a `ClickMap` of viewer *actions* (panel controls, legend swatches);
- a `PickMap` of the source data an annotation was built from, which an
  interactive viewer prints or copies when the annotation is clicked.

All three share `RegionMap`: a viewer queries one with `RegionMap.hit` on mouse
input, and `RegionMap.offset`/`RegionMap.scaled` transform a map when its frame
is placed into a larger composite or resized for display.
"""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from typing_extensions import Self

from luxonis_ml.typing import ParamValue
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.tooltip import Tooltip

#: What a map's regions resolve to — a tooltip, an action, or source data.
PayloadT = TypeVar("PayloadT")


@dataclass
class RegionMap(Generic[PayloadT]):
    """Hit-test entries in final frame pixels, each carrying a payload.

    `hit` returns the payload of the *smallest* rectangle containing a point, so
    a small box nested inside a large one still wins.

    Attributes:
        items: The ``(rect, payload)`` entries, in draw order.

    """

    items: list[tuple[Rect, PayloadT]] = field(default_factory=list)

    @classmethod
    def empty(cls) -> Self:
        """Return a map with no entries."""
        return cls([])

    def hit(self, x: float, y: float) -> PayloadT | None:
        """Return the payload of the smallest box containing ``(x, y)``.

        Args:
            x: Point x in frame pixels.
            y: Point y in frame pixels.

        Returns:
            The matching payload, or ``None`` when no box contains the point.
            On ties the first (earliest-drawn) box wins.

        """
        best: PayloadT | None = None
        best_area: float | None = None
        for rect, payload in self.items:
            if rect.left <= x <= rect.right and rect.top <= y <= rect.bottom:
                area = rect.area
                if best_area is None or area < best_area:
                    best, best_area = payload, area
        return best

    def offset(self, dx: float, dy: float) -> Self:
        """Return a copy with every rectangle shifted by ``(dx, dy)`` pixels."""
        return type(self)(
            [
                (Rect(r.left + dx, r.top + dy, r.right + dx, r.bottom + dy), p)
                for r, p in self.items
            ]
        )

    def scaled(self, factor_x: float, factor_y: float | None = None) -> Self:
        """Return a copy with rectangles scaled about the origin on each axis."""
        if factor_y is None:
            factor_y = factor_x
        return type(self)(
            [
                (
                    Rect(
                        r.left * factor_x,
                        r.top * factor_y,
                        r.right * factor_x,
                        r.bottom * factor_y,
                    ),
                    p,
                )
                for r, p in self.items
            ]
        )

    def merge(self, other: Self) -> Self:
        """Return a new map with this map's entries followed by ``other``'s."""
        return type(self)([*self.items, *other.items])

    def __or__(self, other: Self) -> Self:
        """``self | other`` — shorthand for `merge`."""
        return self.merge(other)


@dataclass
class HitMap(RegionMap[Tooltip]):
    """Hover regions, each carrying the `Tooltip` to show over it."""


@dataclass
class ClickMap(RegionMap[str]):
    """Click regions, each carrying an action string a viewer dispatches.

    An action is opaque to the map — e.g. ``"key:m"`` for a control or
    ``"class:car"`` for a legend swatch (see `luxonis_ml.vizlab.viewer.Viewer`).
    """


@dataclass
class PickMap(RegionMap["ParamValue"]):
    """Click regions, each carrying the source data of the annotation drawn there.

    The payload is JSON-like — for a dataset detection, the LDF annotation it was
    rendered from — so a viewer can print or copy it when the annotation is
    clicked. A `None` payload is never stored, which is what lets `RegionMap.hit`
    report a miss as ``None``.
    """


@dataclass
class InteractionCapture:
    """Mutable render-time collector for hover, click, and pick regions.

    A capture carries an affine transform from the scene currently being drawn
    to the final output pixels. Nested composites derive child captures instead
    of manually offsetting maps after rendering, so interaction regions follow
    the same placement and scaling path as their pixels.
    """

    hover: list[tuple[Rect, Tooltip]] = field(default_factory=list)
    clicks: list[tuple[Rect, str]] = field(default_factory=list)
    picks: list[tuple[Rect, "ParamValue"]] = field(default_factory=list)
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def transformed(
        self,
        x: float,
        y: float,
        scale_x: float = 1.0,
        scale_y: float | None = None,
    ) -> "InteractionCapture":
        """Return a view that maps child-local coordinates into this capture."""
        if scale_y is None:
            scale_y = scale_x
        return InteractionCapture(
            hover=self.hover,
            clicks=self.clicks,
            picks=self.picks,
            scale_x=self.scale_x * scale_x,
            scale_y=self.scale_y * scale_y,
            offset_x=self.offset_x + self.scale_x * x,
            offset_y=self.offset_y + self.scale_y * y,
        )

    def _rect(self, rect: Rect) -> Rect:
        return Rect(
            rect.left * self.scale_x + self.offset_x,
            rect.top * self.scale_y + self.offset_y,
            rect.right * self.scale_x + self.offset_x,
            rect.bottom * self.scale_y + self.offset_y,
        )

    def add_hover(self, rect: Rect, tooltip: Tooltip) -> None:
        """Add a hover region expressed in the current scene's coordinates."""
        self.hover.append((self._rect(rect), tooltip))

    def add_click(self, rect: Rect, action: str) -> None:
        """Add a click region expressed in the current scene's coordinates."""
        self.clicks.append((self._rect(rect), action))

    def add_pick(self, rect: Rect, source: "ParamValue") -> None:
        """Add a pickable region expressed in the current scene's coordinates."""
        self.picks.append((self._rect(rect), source))

    def add_hitmap(self, hitmap: HitMap) -> None:
        """Add every entry from ``hitmap`` using the current transform."""
        for rect, tooltip in hitmap.items:
            self.add_hover(rect, tooltip)

    def add_clickmap(self, clickmap: ClickMap) -> None:
        """Add every entry from ``clickmap`` using the current transform."""
        for rect, action in clickmap.items:
            self.add_click(rect, action)

    def add_pickmap(self, pickmap: PickMap) -> None:
        """Add every entry from ``pickmap`` using the current transform."""
        for rect, source in pickmap.items:
            self.add_pick(rect, source)
