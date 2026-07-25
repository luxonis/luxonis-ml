"""Bounding boxes — axis-aligned or oriented — with an auto-styled label chip.

`BBox` subclasses the Luxonis Data Format `BBoxAnnotation`,
so it reuses its normalized top-left ``x, y, w, h`` coordinates (and their
validation) and adds only rendering: a translucent fill, a solid stroke, and an
optional rounded label chip (class name, confidence, and/or payload). An
``angle`` field rotates the box about its center for oriented detectors — the one
thing LDF boxes don't carry.

An axis-aligned box (angle 0) is drawn as a rounded rectangle with a soft shadow;
a rotated one is drawn as an anti-aliased quad (square corners, no shadow).
"""

import math

from luxonis_ml.ldf import BBoxAnnotation
from luxonis_ml.vizlab.canvas import Shadow
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import (
    XY,
    Rect,
    bounding_rect,
    oriented_corners,
)
from luxonis_ml.vizlab.style import Palette, Style

from .base import Annotation, RenderContext
from .chip import place_label


class BBox(BBoxAnnotation, Annotation):
    """A single bounding box, axis-aligned or oriented.

    Reuses the normalized ``x, y, w, h`` fields (top-left origin) of
    `BBoxAnnotation`; ``angle`` adds optional rotation.

    Attributes:
        angle: Rotation about the box center (0 for an axis-aligned box).
        angle_unit: ``"deg"`` (default) or ``"rad"`` for ``angle``.

    See `Annotation` for the shared
    ``label``, ``score``, ``payload``, ``color``, ``style``, and ``palette`` fields.

    Examples:
        >>> BBox(x=0.1, y=0.2, w=0.3, h=0.4)._axis_aligned()
        True
        >>> BBox(x=0.1, y=0.2, w=0.3, h=0.4, angle=15)._axis_aligned()
        False

    """

    angle: float = 0.0
    angle_unit: str = "deg"

    @classmethod
    def from_ldf(
        cls,
        annotation: BBoxAnnotation,
        *,
        label: str | None = None,
        score: float | None = None,
        palette: Palette | None = None,
    ) -> "BBox":
        """Build a renderable box from an LDF `BBoxAnnotation`.

        Reuses the annotation's coordinates directly; only rendering state is added.

        Args:
            annotation: The LDF bounding-box annotation.
            label: Class label for the chip and palette color.
            score: Optional confidence shown on the chip.
            palette: Palette used to color the box from ``label``.

        Returns:
            The equivalent `BBox`.

        """
        return cls(
            **annotation.model_dump(),
            label=label,
            score=score,
            palette=palette,
        )

    def _angle_rad(self) -> float:
        """Convert ``angle`` in the box's ``angle_unit`` to radians."""
        return (
            math.radians(self.angle)
            if self.angle_unit == "deg"
            else float(self.angle)
        )

    def _axis_aligned(self) -> bool:
        """Whether the box has no rotation (drawn as a rounded rect)."""
        return float(self.angle) == 0.0

    def _rect(self, width: int, height: int) -> Rect:
        """Resolve the normalized box to a pixel rectangle, clipped to the canvas."""
        rect = Rect(
            self.x * width,
            self.y * height,
            (self.x + self.w) * width,
            (self.y + self.h) * height,
        )
        return rect.clip_to(width, height)

    def _corners(self, width: int, height: int) -> list[XY]:
        """Resolve the box to four pixel corner points (rotated by ``angle``)."""
        rect = Rect(
            self.x * width,
            self.y * height,
            (self.x + self.w) * width,
            (self.y + self.h) * height,
        )
        cx, cy = rect.center
        return list(
            oriented_corners(
                cx, cy, rect.width, rect.height, self._angle_rad()
            )
        )

    def extent(self) -> Rect | None:
        """Return ``None``: normalized boxes have no pixel extent until render.

        Returns:
            Always ``None``.

        """
        return None

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Draw the box and its label chip onto the canvas.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved box color.

        """
        canvas = ctx.canvas
        fill = (
            color.with_alpha(style.fill_alpha)
            if style.fill_alpha > 0
            else None
        )

        if self._axis_aligned():
            region = self._rect(canvas.width, canvas.height)
            canvas.rounded_rect(
                region,
                radius=style.corner_radius,
                fill=fill,
                stroke=color,
                stroke_width=style.stroke_width,
                dash=style.dash,
                shadow=Shadow() if style.shadow else None,
            )
        else:
            corners = self._corners(canvas.width, canvas.height)
            canvas.polygon(
                corners,
                fill=fill,
                stroke=color,
                stroke_width=style.stroke_width,
                dash=style.dash,
            )
            region = bounding_rect(corners)

        place_label(
            ctx, region, self.label, self.score, self.payload, color, style
        )
