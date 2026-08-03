"""Arrows that state a relation between two things in a scene.

`Arrow` draws "this, because of that": a detection and the track it continues, a
plate and the car it belongs to, a prediction and the ground truth it was matched
against. What makes it more than a line is that an endpoint may be *another
annotation* rather than a pair of coordinates. Those references resolve while the
scene is drawn, against the canvas the scene is drawn on — so an arrow anchored
to a box lands on that box's edge at every render size, and follows the box if it
moves. Nothing has to be recomputed by the caller, and no pixel coordinate is
ever baked into the annotation.

An arrow between two shapes stops at their bounds rather than at their centers,
leaving a small gap, so it reads as touching both without covering either. Give
it a ``curvature`` when several relations share a neighbourhood: bowing them by
different amounts (or in opposite directions) keeps a bundle of arrows legible
where a bundle of straight lines would collapse into one smudge.
"""

import math
from typing import ClassVar, Literal, TypeAlias

from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import XY, Rect, arrow_head, bounding_rect
from luxonis_ml.vizlab.style import Style

from .base import Annotation, RenderContext
from .chip import place_label

Endpoint: TypeAlias = tuple[float, float] | Annotation
"""Where an arrow ends: a normalized ``(x, y)`` point, or another annotation."""

Heads = Literal["end", "start", "both", "none"]
"""Which ends of an arrow carry a head."""

#: Length of an arrow head, and the width of its base, in stroke widths.
_HEAD_LENGTH = 4.5
_HEAD_WIDTH = 3.4

#: Points sampled along a curved arrow. Enough that the flattening is invisible
#: at any sane render size, few enough that an SVG path stays small.
_CURVE_STEPS = 24


class Arrow(Annotation):
    """An arrow from one point or annotation to another.

    Attributes:
        start: Where the arrow comes from: a normalized ``(x, y)`` point, or an
            annotation whose bounds the arrow leaves from.
        end: Where the arrow goes to, in the same two forms.
        curvature: How far the arrow bows away from the straight chord, as a
            fraction of its length. ``0`` (the default) is a straight line;
            ``0.2`` is a gentle arc and the sign picks the side, so two arrows
            between the same pair of shapes can be given opposite signs to keep
            them apart.
        heads: Which ends carry an arrow head — ``"end"`` (the default),
            ``"start"``, ``"both"``, or ``"none"`` for a plain connector.
        gap: Distance in pixels left between an annotation's bounds and the
            arrow's tip. Ignored for a raw point endpoint, which is taken
            literally.

    See `Annotation` for the shared ``label``, ``score``, ``payload``,
    ``color``, ``style``, ``tooltip``, and ``palette`` fields; a label is drawn
    as a chip at the middle of the arrow, placed by the same collision-aware
    layout as every other chip.

    Note:
        Reference the annotations an arrow points at as *siblings* in the scene
        (add them, and the arrow, to the same `Image`), not as ancestors of it.
        An arrow nested inside an annotation it also points at would make the
        scene graph cyclic, which nothing that walks it — rendering, the render
        cache, ``repr`` — is prepared for.

    Examples:
        Endpoints resolve at render time, against the canvas being drawn on:

        >>> from luxonis_ml.vizlab import Arrow, BBox
        >>> car = BBox(x=0.0, y=0.4, w=0.2, h=0.2, label="car")
        >>> plate = BBox(x=0.7, y=0.4, w=0.2, h=0.2, label="plate")
        >>> arrow = Arrow(start=car, end=plate, label="reads")
        >>> arrow._endpoints(100, 100)
        ((24.0, 50.0), (66.0, 50.0))

        The boxes' facing edges, plus the 4px gap — and moving a box moves the
        arrow, because nothing was baked in:

        >>> plate.x = 0.5
        >>> arrow._endpoints(100, 100)
        ((24.0, 50.0), (46.0, 50.0))

    """

    LAYER: ClassVar[str] = "relation"

    start: Endpoint
    end: Endpoint
    curvature: float = 0.0
    heads: Heads = "end"
    gap: float = 4.0

    def extent(self) -> Rect | None:
        """Return ``None``: an arrow has no pixel extent until render.

        Returns:
            Always ``None``.

        """
        return None

    def _anchor(
        self, target: Endpoint, width: int, height: int
    ) -> Rect | None:
        """Resolve one endpoint to the rectangle the arrow attaches to.

        A raw point becomes an empty rectangle at that point, so the same edge
        arithmetic covers both kinds of endpoint: leaving an empty rectangle
        lands exactly on the point given.
        """
        if isinstance(target, Annotation):
            return target.region_at(width, height)
        x, y = target
        return Rect(x * width, y * height, x * width, y * height)

    def _exit(self, rect: Rect, toward: XY) -> XY:
        """Return where the arrow leaves ``rect`` heading for ``toward``.

        The point where the ray from the rectangle's center to ``toward``
        crosses its boundary, pushed a further `gap` pixels out. An empty
        rectangle (a raw point endpoint) has no boundary to cross and no gap to
        leave, so it returns its own position.
        """
        cx, cy = rect.center
        dx, dy = toward[0] - cx, toward[1] - cy
        norm = math.hypot(dx, dy)
        if norm == 0.0:
            return (cx, cy)
        ux, uy = dx / norm, dy / norm
        half_w, half_h = rect.width / 2.0, rect.height / 2.0
        if half_w == 0.0 and half_h == 0.0:
            return (cx, cy)
        # The shorter of the two axis crossings is the one on the boundary.
        span = min(
            half_w / abs(ux) if ux else math.inf,
            half_h / abs(uy) if uy else math.inf,
        )
        reach = span + self.gap
        return (cx + ux * reach, cy + uy * reach)

    def _control(self, start: XY, end: XY) -> XY:
        """Return the quadratic control point that bows the arrow.

        At zero curvature this is the chord's midpoint, which makes the curve a
        straight line — so the curved and straight cases share one code path.
        """
        mx, my = (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0
        if self.curvature == 0.0:
            return (mx, my)
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0.0:
            return (mx, my)
        # Perpendicular to the chord; the sign of ``curvature`` picks the side.
        offset = self.curvature * length
        return (mx - dy / length * offset, my + dx / length * offset)

    def _endpoints(self, width: int, height: int) -> tuple[XY, XY] | None:
        """Resolve both endpoints to the pixels the arrow runs between.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        Returns:
            The ``(start, end)`` points, or ``None`` when an endpoint cannot be
            resolved (an annotation with no bounds, or two endpoints on top of
            one another, which no arrow can point along).

        """
        first = self._anchor(self.start, width, height)
        second = self._anchor(self.end, width, height)
        if first is None or second is None:
            return None
        if first.center == second.center:
            return None
        # Both ends leave toward the control point, so the arrow departs along
        # its own tangent rather than along the chord it bows away from.
        control = self._control(first.center, second.center)
        return self._exit(first, control), self._exit(second, control)

    def _path(self, width: int, height: int) -> list[XY] | None:
        """Sample the drawn run of the arrow, flattening any curvature."""
        ends = self._endpoints(width, height)
        if ends is None:
            return None
        start, end = ends
        if self.curvature == 0.0:
            return [start, end]
        control = self._control(start, end)
        return [
            _quadratic(start, control, end, step / _CURVE_STEPS)
            for step in range(_CURVE_STEPS + 1)
        ]

    def region_at(self, width: int, height: int) -> Rect | None:
        """Return the bounds of the drawn arrow, in canvas pixels.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        Returns:
            The bounding `Rect`, or ``None`` when the arrow does not resolve.

        """
        path = self._path(width, height)
        return bounding_rect(path) if path else None

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Draw the connector and its head(s).

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved arrow color.

        """
        canvas = ctx.canvas
        path = self._path(canvas.width, canvas.height)
        if path is None:
            return
        stroke = self.outline_color(ctx, color)
        canvas.polygon(
            path,
            stroke=stroke,
            stroke_width=style.stroke_width,
            dash=style.dash,
            closed=False,
        )
        length = _HEAD_LENGTH * style.stroke_width
        head_width = _HEAD_WIDTH * style.stroke_width
        for tip, tail in self._head_tips(path):
            canvas.polygon(
                list(
                    arrow_head(
                        tip,
                        (tip[0] - tail[0], tip[1] - tail[1]),
                        length,
                        head_width,
                    )
                ),
                fill=stroke,
            )

    def _head_tips(self, path: list[XY]) -> list[tuple[XY, XY]]:
        """Return the ``(tip, preceding point)`` pairs that need a head."""
        tips: list[tuple[XY, XY]] = []
        if self.heads in ("end", "both"):
            tips.append((path[-1], path[-2]))
        if self.heads in ("start", "both"):
            tips.append((path[0], path[1]))
        return tips

    def draw_label(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Place the arrow's mid-label chip and emit its hover region.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved arrow color.

        """
        canvas = ctx.canvas
        path = self._path(canvas.width, canvas.height)
        if path is None:
            return
        region = bounding_rect(path)
        pad = style.stroke_width
        ctx.emit_hit(
            Rect(
                region.left - pad,
                region.top - pad,
                region.right + pad,
                region.bottom + pad,
            ),
            self.tooltip,
        )
        middle = _midpoint(path)
        place_label(
            ctx,
            Rect(middle[0], middle[1], middle[0], middle[1]),
            self.label,
            self.score,
            self.payload,
            color,
            style,
            id(self),
        )


def _midpoint(path: list[XY]) -> XY:
    """Return the point halfway along ``path``.

    A sampled curve has an odd number of points, so its middle sample *is* the
    halfway point. A straight arrow is two points and has no middle sample, so
    the two are averaged — taking the middle index there would put the label on
    the arrow's tip.

    Args:
        path: The drawn points, at least two of them.

    Returns:
        The halfway ``(x, y)``.

    Examples:
        >>> _midpoint([(0.0, 0.0), (10.0, 4.0)])
        (5.0, 2.0)
        >>> _midpoint([(0.0, 0.0), (3.0, 9.0), (10.0, 4.0)])
        (3.0, 9.0)

    """
    half = len(path) // 2
    if len(path) % 2 == 1:
        return path[half]
    before, after = path[half - 1], path[half]
    return ((before[0] + after[0]) / 2.0, (before[1] + after[1]) / 2.0)


def _quadratic(start: XY, control: XY, end: XY, t: float) -> XY:
    """Return the point at ``t`` on the quadratic Bézier through ``control``."""
    u = 1.0 - t
    a, b, c = u * u, 2.0 * u * t, t * t
    return (
        a * start[0] + b * control[0] + c * end[0],
        a * start[1] + b * control[1] + c * end[1],
    )
