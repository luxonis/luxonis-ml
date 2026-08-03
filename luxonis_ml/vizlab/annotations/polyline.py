"""Open and closed runs of points: lane lines, boundaries, paths, wireframes.

`Polyline` is the label type for structure that is a *line* rather than a
region. A lane marking, a road or field boundary, a tracked object's trajectory,
a wireframe edge loop: a mask models all of these badly, because rasterizing a
line throws away the ordering that makes it one and leaves a fill a few pixels
wide whose contour, traced back, is a hairpin around it. Keeping the vertices
means the run can be drawn thin and crisp at any resolution, tapered along its
length, arrowed to show which way it goes, and colored by a value that varies
from one end to the other.

There is no LDF polyline to adapt from. LDF accepts ``points`` as a
*construction* format for `SegmentationAnnotation`, but its validator rasterizes
them to an RLE mask straight away, so by the time an annotation object exists
the vertices are gone — a `Polyline` is always built by the caller (or by a
model's output), never recovered from a stored LDF label.
"""

import bisect
import math
from typing import ClassVar

from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import XY, Rect, arrow_head, bounding_rect
from luxonis_ml.vizlab.gradient import (
    DEFAULT_GRADIENT,
    Gradient,
    resolve_gradient,
)
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.style import Style

from .base import Annotation, RenderContext
from .chip import place_label

#: Length of a direction chevron, in stroke widths, and the width of its base.
#: A chevron has to read as an arrow at a glance without swallowing the line it
#: sits on, which puts it a few stroke widths across whatever the stroke is.
_CHEVRON_LENGTH = 4.0
_CHEVRON_WIDTH = 3.2


class Polyline(Annotation):
    """An ordered run of points, drawn open or closed.

    Coordinates are normalized to the source image, like every other spatial
    annotation, so a polyline drawn on a thumbnail and on the full frame is the
    same annotation.

    Attributes:
        points: The vertices as normalized ``(x, y)`` pairs, in order. Fewer
            than two points draws nothing.
        closed: Whether the last point connects back to the first (a ring).
        fill: Whether a ``closed`` ring is filled translucently at
            ``style.fill_alpha``, as a box is. Off by default: a polyline is a
            line, and a wireframe or a lane loop reads better unfilled.
        widths: Per-vertex stroke multipliers over ``style.stroke_width``,
            index-aligned to ``points``; a short list falls back to ``1.0`` for
            the rest. Use it to taper a trajectory so its older end is thinner.
        values: Per-vertex scalars, index-aligned to ``points``, colored through
            ``gradient`` so a quantity varying along the run (speed, curvature,
            per-point confidence) is visible in the line itself. Each segment
            fades between its endpoints' colors, and the class/palette color is
            not used while this is set.
        gradient: Colormap for ``values``: a `Gradient` or the name of a preset
            (see `luxonis_ml.vizlab.gradient.GRADIENTS`). ``None`` inherits the
            render options' gradient, then the library default.
        vmin: Value mapped to the low end of the gradient; ``None`` uses the
            smallest of ``values``. Pin both ends to put two polylines on one
            scale (an unpinned line normalizes over its own range alone).
        vmax: Value mapped to the high end; ``None`` uses the largest.
        arrows: How many direction chevrons to space evenly along the run, the
            last one sitting at the final vertex. ``1`` is a single arrowhead at
            the end; ``0`` (the default) draws none.

    See `Annotation` for the shared ``label``, ``score``, ``payload``,
    ``color``, ``style``, ``tooltip``, and ``palette`` fields.

    Examples:
        Normalized vertices resolve against the canvas at render time:

        >>> from luxonis_ml.vizlab import Polyline
        >>> lane = Polyline(points=[(0.1, 0.5), (0.9, 0.5)], label="lane")
        >>> lane._pixels(200, 100)
        [(20.0, 50.0), (180.0, 50.0)]
        >>> lane.region_at(200, 100)
        Rect(left=20.0, top=50.0, right=180.0, bottom=50.0)

        A closed run returns to its first vertex:

        >>> ring = Polyline(
        ...     points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], closed=True
        ... )
        >>> ring._run(ring._pixels(10, 10))[-1]
        (0.0, 0.0)

    """

    LAYER: ClassVar[str] = "polyline"

    points: list[tuple[float, float]] = []
    closed: bool = False
    fill: bool = False
    widths: list[float] | None = None
    values: list[float] | None = None
    gradient: Gradient | str | None = None
    vmin: float | None = None
    vmax: float | None = None
    arrows: int = 0

    def _pixels(self, width: int, height: int) -> list[XY]:
        """Resolve the normalized vertices to canvas pixels."""
        return [(x * width, y * height) for x, y in self.points]

    def _run(self, pixels: list[XY]) -> list[XY]:
        """Return the drawn run: the vertices, plus the first again if closed."""
        if self.closed and len(pixels) > 2:
            return [*pixels, pixels[0]]
        return pixels

    def extent(self) -> Rect | None:
        """Return ``None``: normalized points have no pixel extent until render.

        Returns:
            Always ``None``.

        """
        return None

    def region_at(self, width: int, height: int) -> Rect | None:
        """Return the bounding rectangle of the vertices, in canvas pixels.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        Returns:
            The bounding `Rect`, or ``None`` when there are no points.

        """
        pixels = self._pixels(width, height)
        return bounding_rect(pixels) if pixels else None

    def _anchor_region(self, width: int, height: int, style: Style) -> Rect:
        """Return the rectangle the label chip is placed against.

        A ring is labeled like a box, against its bounds. An open run is not:
        its bounds are a rectangle the line only touches at two corners, so a
        chip placed against them can end up nowhere near the line. It is labeled
        at its first vertex instead — the end a lane line or a trajectory is
        read from — with the chip layout free to push the chip away and draw a
        leader back if that spot is crowded.
        """
        pixels = self._pixels(width, height)
        if self.closed:
            return bounding_rect(pixels)
        pad = style.stroke_width
        x, y = pixels[0]
        return Rect(x - pad, y - pad, x + pad, y + pad)

    def _hit_region(
        self, width: int, height: int, style: Style
    ) -> Rect | None:
        """Return the hover region: the vertex bounds padded by the stroke."""
        region = self.region_at(width, height)
        if region is None:
            return None
        pad = style.stroke_width
        return Rect(
            region.left - pad,
            region.top - pad,
            region.right + pad,
            region.bottom + pad,
        )

    def _per_vertex(
        self, values: list[float] | None, count: int, default: float
    ) -> list[float]:
        """Spread an index-aligned list over ``count`` vertices of the run.

        A closed run has one more drawn point than it has vertices, and a short
        list is filled out with ``default``, so both stay in step with the
        points the caller actually gave.
        """
        source = values or []
        out = [
            float(source[i]) if i < len(source) else default
            for i in range(min(count, len(self.points)))
        ]
        # The closing point is the first vertex again, so it repeats its value.
        while len(out) < count:
            out.append(out[0] if out else default)
        return out

    def _vertex_colors(
        self, ctx: RenderContext, count: int
    ) -> list[Color] | None:
        """Color each vertex from ``values``, or ``None`` when there are none."""
        if not self.values:
            return None
        scalars = self._per_vertex(self.values, count, 0.0)
        low = float(self.vmin) if self.vmin is not None else min(scalars)
        high = float(self.vmax) if self.vmax is not None else max(scalars)
        gradient = resolve_gradient(
            self.gradient or ctx.gradient or DEFAULT_GRADIENT
        )
        span = high - low
        return [
            gradient.color_at((value - low) / span if span > 0 else 0.0)
            for value in scalars
        ]

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Draw the run — fill, stroke, then direction chevrons.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved polyline color.

        """
        canvas = ctx.canvas
        run = self._run(self._pixels(canvas.width, canvas.height))
        if len(run) < 2:
            return
        # A nested sub-label fills in its own class color but strokes in the
        # parent's, so it stays visibly tied to it (see Annotation.outline_color).
        stroke = self.outline_color(ctx, color)
        if self.closed and self.fill and style.fill_alpha > 0:
            canvas.polygon(run, fill=color.with_alpha(style.fill_alpha))
        colors = self._vertex_colors(ctx, len(run))
        widths = (
            None
            if self.widths is None
            else [
                style.stroke_width * m
                for m in self._per_vertex(self.widths, len(run), 1.0)
            ]
        )
        if colors is None and widths is None:
            # The uniform case is one path rather than N segments: a single
            # stroke in the raster backend, and a single <path> in an SVG.
            canvas.polygon(
                run,
                stroke=stroke,
                stroke_width=style.stroke_width,
                dash=style.dash,
                closed=False,
            )
        else:
            self._draw_segments(canvas, run, stroke, colors, widths, style)
        self._draw_arrows(canvas, run, colors[-1] if colors else stroke, style)

    def _draw_segments(
        self,
        canvas: Canvas,
        run: list[XY],
        stroke: Color,
        colors: list[Color] | None,
        widths: list[float] | None,
        style: Style,
    ) -> None:
        """Stroke the run one segment at a time, varying color and/or width.

        A segment spanning two colors is a true gradient stroke rather than a
        run of solid pieces; one whose ends share a color is an ordinary dashable
        line. Width varies per segment (the mean of its two endpoints), which is
        what makes a taper look continuous under round caps.
        """
        for i in range(len(run) - 1):
            width = (
                style.stroke_width
                if widths is None
                else (widths[i] + widths[i + 1]) / 2.0
            )
            first = stroke if colors is None else colors[i]
            second = stroke if colors is None else colors[i + 1]
            if first == second:
                canvas.polygon(
                    [run[i], run[i + 1]],
                    stroke=first,
                    stroke_width=width,
                    dash=style.dash,
                    closed=False,
                )
            else:
                canvas.gradient_line(
                    run[i], run[i + 1], first, second, width=width
                )

    def _draw_arrows(
        self, canvas: Canvas, run: list[XY], color: Color, style: Style
    ) -> None:
        """Draw ``arrows`` direction chevrons spaced evenly along the run."""
        if self.arrows <= 0:
            return
        lengths = [0.0]
        for i in range(len(run) - 1):
            step = math.dist(run[i], run[i + 1])
            lengths.append(lengths[-1] + step)
        total = lengths[-1]
        if total <= 0.0:
            return
        for k in range(1, self.arrows + 1):
            tip, direction = _at_distance(
                run, lengths, total * k / self.arrows
            )
            canvas.polygon(
                list(
                    arrow_head(
                        tip,
                        direction,
                        _CHEVRON_LENGTH * style.stroke_width,
                        _CHEVRON_WIDTH * style.stroke_width,
                    )
                ),
                fill=color,
            )

    def draw_label(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Place the polyline's label chip and emit its hover region.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved polyline color.

        """
        if not self.points:
            return
        canvas = ctx.canvas
        hit = self._hit_region(canvas.width, canvas.height, style)
        if hit is not None:
            ctx.emit_hit(hit, self.tooltip)
        place_label(
            ctx,
            self._anchor_region(canvas.width, canvas.height, style),
            self.label,
            self.score,
            self.payload,
            color,
            style,
            id(self),
        )


def _at_distance(
    run: list[XY], lengths: list[float], distance: float
) -> tuple[XY, XY]:
    """Return the point ``distance`` along ``run`` and the direction there.

    Args:
        run: The drawn points.
        lengths: Cumulative arc length at each point (``lengths[0] == 0``).
        distance: How far along the run to walk, in pixels.

    Returns:
        The ``(point, direction)`` pair; the direction is the containing
        segment's, unnormalized.

    """
    # The first vertex at or past ``distance`` ends the segment containing it.
    index = min(max(bisect.bisect_left(lengths, distance), 1), len(run) - 1)
    start, end = run[index - 1], run[index]
    span = lengths[index] - lengths[index - 1]
    t = 1.0 if span <= 0.0 else (distance - lengths[index - 1]) / span
    point = (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
    )
    return point, (end[0] - start[0], end[1] - start[1])
