"""Scale bars and rulers: what a distance in the picture is worth.

A pixel means nothing on its own. `ScaleBar` pins a corner of the frame with a
bar of a known length — "this much is 50 mm" — and `Ruler` measures between two
points inside the scene. Both take the calibration from the caller as a plain
``pixels_per_unit`` number: LDF carries no camera or calibration model, and
inventing a schema for one here would be guessing at a format that belongs in
the dataset layer, not in the renderer.

A bar has to be a length that can be read off it, so `ScaleBar` never draws the
raw fraction of the frame it was allowed: it picks the largest *round* length
(1, 2, or 5 times a power of ten) that fits, which is why a map's scale bar
always says 50 m and never 47.3 m.

Both take the calibration in the pixels of the image the caller measured — set
``reference_width`` when that differs from the width the scene is rendered at,
or the bar will keep its display length while the picture around it shrinks, and
quietly lie.
"""

import math
from typing import ClassVar

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import XY, Rect, bounding_rect
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.style import Style

from .base import Annotation, RenderContext
from .card import draw_card_background
from .chip import place_label
from .overlay import Cell, Corner, CornerStack, resolve_chrome

#: Padding inside the scale bar's card, and the gap between bar and caption.
_PAD = 9.0
_GAP = 5.0
#: Height of the bar's end ticks, and of a ruler's, in stroke widths.
_TICK = 3.0


def nice_length(limit: float) -> float:
    """Return the largest round length that fits within ``limit``.

    Round means 1, 2, or 5 times a power of ten — the lengths a reader can
    divide by eye. Used to choose what a scale bar should measure, given how
    much room it has.

    Args:
        limit: The largest acceptable length, in the same units as the result.

    Returns:
        The chosen length, always ``<= limit``.

    Raises:
        ValueError: If ``limit`` is not a positive, finite number.

    Examples:
        >>> nice_length(37.0)
        20.0
        >>> nice_length(0.9)
        0.5
        >>> nice_length(1000.0)
        1000.0

    """
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"nice_length needs a positive limit, got {limit}")
    base = 10.0 ** math.floor(math.log10(limit))
    for step in (5.0, 2.0, 1.0):
        if step * base <= limit:
            return step * base
    return base  # pragma: no cover - one of the steps above always fits


def format_units(value: float) -> str:
    """Format a measured value for a scale bar or ruler caption.

    Whole numbers lose their decimal point, and everything else is cut to three
    significant figures — a measurement drawn on a picture is read, not
    computed with, so trailing digits are noise.

    Args:
        value: The measured value.

    Returns:
        The formatted number, without a unit.

    Examples:
        >>> format_units(50.0), format_units(12.44), format_units(0.05)
        ('50', '12.4', '0.05')
        >>> format_units(1234.6)
        '1235'

    """
    if value == int(value) and abs(value) < 1e9:
        return str(int(value))
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    return f"{value:.3g}"


def _display_scale(canvas_width: int, reference_width: int | None) -> float:
    """Return display pixels per calibrated pixel."""
    if reference_width is None or reference_width <= 0:
        return 1.0
    return canvas_width / reference_width


class ScaleBar(CornerStack):
    """A corner-pinned bar of a known length, captioned with what it measures.

    Attributes:
        pixels_per_unit: How many image pixels one unit spans. The default of
            ``1`` with the default ``unit`` makes it a plain pixel ruler, which
            is what an uncalibrated image can honestly show.
        unit: The unit's name, appended to the caption. Caller-authored, so it
            may carry inline markup (``"<i>µ</i>m"``).
        reference_width: Width, in pixels, of the image ``pixels_per_unit`` was
            measured on. ``None`` (the default) reads the calibration as
            relative to the canvas being drawn, which is right whenever the
            scene renders at its source resolution.
        max_fraction: The largest share of the canvas width the bar may take,
            before rounding down to a round length.

    See `CornerStack` for ``corner``/``margin``/``gap``. A scale bar is chrome
    rather than a label: it takes no class color and emits no hover region.

    Examples:
        >>> from luxonis_ml.vizlab import ScaleBar
        >>> bar = ScaleBar(pixels_per_unit=8.0, unit="m")
        >>> bar._choose(400)  # a quarter of 400px is 12.5m, so 10m is drawn
        (10.0, 80.0)

        Rendering the same scene at half size halves the bar with it, as long
        as the calibration says which image it was measured on:

        >>> ScaleBar(
        ...     pixels_per_unit=8.0, unit="m", reference_width=400
        ... )._choose(200)
        (10.0, 40.0)

    """

    corner: Corner = Corner.BOTTOM_RIGHT

    pixels_per_unit: float = 1.0
    unit: str = "px"
    reference_width: int | None = None
    max_fraction: float = 0.25

    def _choose(self, canvas_width: int) -> "tuple[float, float] | None":
        """Pick the bar's length in units and in display pixels.

        Args:
            canvas_width: Width of the canvas being drawn on, in pixels.

        Returns:
            The ``(units, pixels)`` pair, or ``None`` when the calibration or
            the room available leaves nothing sensible to draw.

        """
        per_unit = self.pixels_per_unit * _display_scale(
            canvas_width, self.reference_width
        )
        room = canvas_width * self.max_fraction
        if not math.isfinite(per_unit) or per_unit <= 0.0 or room <= 0.0:
            return None
        units = nice_length(room / per_unit)
        return units, units * per_unit

    def _caption(self, units: float) -> str:
        """Compose the bar's caption from a measured length."""
        return (
            f"{format_units(units)} {self.unit}"
            if self.unit
            else (format_units(units))
        )

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        chosen = self._choose(canvas.width)
        if chosen is None:
            return []
        units, bar_width = chosen
        text = self._caption(units)
        metrics = canvas.measure_markup(
            text, style.font_size, weight=style.font_weight
        )
        chrome = resolve_chrome(ctx)
        tick = _TICK * style.stroke_width
        width = max(bar_width, metrics.width) + 2 * _PAD
        height = tick + _GAP + metrics.height + 2 * _PAD

        def draw(cv: Canvas, rect: Rect) -> None:
            _draw_bar(
                cv, rect, style, chrome, bar_width, tick, text, metrics.ascent
            )

        return [Cell(width, height, draw)]


def _draw_bar(
    canvas: Canvas,
    rect: Rect,
    style: Style,
    chrome: brand.Chrome,
    bar_width: float,
    tick: float,
    text: str,
    ascent: float,
) -> None:
    """Paint the card, the ticked bar, and the caption centered beneath it."""
    draw_card_background(canvas, rect, style, chrome)
    ink = chrome.card_text
    center_x = rect.center[0]
    left = center_x - bar_width / 2.0
    right = left + bar_width
    baseline = rect.top + _PAD + tick
    canvas.line((left, baseline), (right, baseline), ink, style.stroke_width)
    for x in (left, right):
        canvas.line(
            (x, baseline - tick), (x, baseline), ink, style.stroke_width
        )
    text_width = canvas.measure_markup(
        text, style.font_size, weight=style.font_weight
    ).width
    canvas.markup(
        (center_x - text_width / 2.0, baseline + _GAP + ascent),
        text,
        size=style.font_size,
        color=ink,
        weight=style.font_weight,
    )


class Ruler(Annotation):
    """A measured line between two points in the scene.

    The distance is measured on the image, converted through
    ``pixels_per_unit``, and shown on a label chip at the middle of the line —
    placed by the same collision-aware layout as every other chip, so a ruler
    across a busy frame does not bury a detection's label.

    Attributes:
        start: One end, as a normalized ``(x, y)`` point.
        end: The other end, normalized.
        pixels_per_unit: How many image pixels one unit spans; the default of
            ``1`` measures in pixels.
        unit: The unit's name, appended to the measurement. Caller-authored, so
            it may carry inline markup.
        reference_width: Width, in pixels, of the image ``pixels_per_unit`` was
            measured on (see `ScaleBar`).
        ticks: Whether to cap both ends with a perpendicular tick.
        measurement: Whether the measured span is shown on the chip. Turning it
            off leaves a bare measured line, and is how an interactive viewer's
            "labels off" reaches text that is generated rather than given (see
            `luxonis_ml.vizlab.viewer.LayerState.apply_layers`).

    See `Annotation` for the shared ``label``, ``color``, ``style``,
    ``tooltip``, and ``palette`` fields. The measurement rides in as the chip's
    payload, so a ``label`` of your own is shown alongside it rather than
    instead of it; setting ``payload`` explicitly replaces the measurement.

    Examples:
        >>> from luxonis_ml.vizlab import Ruler
        >>> span = Ruler(
        ...     start=(0.1, 0.5), end=(0.6, 0.5), pixels_per_unit=8.0, unit="m"
        ... )
        >>> span.measure(160, 100)  # 80px across, at 8px per metre
        '10 m'

        A diagonal is measured in the image plane, not per axis:

        >>> Ruler(start=(0.0, 0.0), end=(0.3, 0.4)).measure(100, 100)
        '50 px'

    """

    LAYER: ClassVar[str] = "overlay"

    start: tuple[float, float]
    end: tuple[float, float]
    pixels_per_unit: float = 1.0
    unit: str = "px"
    reference_width: int | None = None
    ticks: bool = True
    measurement: bool = True

    def _points(self, width: int, height: int) -> tuple[XY, XY]:
        """Resolve both ends to canvas pixels."""
        return (
            (self.start[0] * width, self.start[1] * height),
            (self.end[0] * width, self.end[1] * height),
        )

    def measure(self, width: int, height: int) -> str:
        """Return the span between the two ends, formatted with its unit.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        Returns:
            The measurement, e.g. ``"10 m"``.

        """
        (x0, y0), (x1, y1) = self._points(width, height)
        span = math.dist((x0, y0), (x1, y1)) / _display_scale(
            width, self.reference_width
        )
        per_unit = self.pixels_per_unit
        value = span / per_unit if per_unit > 0 else span
        text = format_units(value)
        return f"{text} {self.unit}" if self.unit else text

    def extent(self) -> Rect | None:
        """Return ``None``: normalized ends have no pixel extent until render.

        Returns:
            Always ``None``.

        """
        return None

    def region_at(self, width: int, height: int) -> Rect | None:
        """Return the bounds of the measured span, in canvas pixels.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        Returns:
            The bounding `Rect` of the two ends.

        """
        return bounding_rect(list(self._points(width, height)))

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Draw the span and its end ticks.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved ruler color.

        """
        canvas = ctx.canvas
        (x0, y0), (x1, y1) = self._points(canvas.width, canvas.height)
        ink = self.outline_color(ctx, color)
        canvas.line((x0, y0), (x1, y1), ink, style.stroke_width)
        length = math.hypot(x1 - x0, y1 - y0)
        if not self.ticks or length == 0.0:
            return
        # Perpendicular to the span, so both caps read as one instrument
        # whatever angle the measurement was taken at.
        reach = _TICK * style.stroke_width
        nx, ny = -(y1 - y0) / length * reach, (x1 - x0) / length * reach
        for x, y in ((x0, y0), (x1, y1)):
            canvas.line(
                (x - nx, y - ny), (x + nx, y + ny), ink, style.stroke_width
            )

    def draw_label(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Place the measurement chip and emit the ruler's hover region.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved ruler color.

        """
        canvas = ctx.canvas
        (x0, y0), (x1, y1) = self._points(canvas.width, canvas.height)
        pad = _TICK * style.stroke_width
        region = self.region_at(canvas.width, canvas.height)
        assert region is not None
        ctx.emit_hit(
            Rect(
                region.left - pad,
                region.top - pad,
                region.right + pad,
                region.bottom + pad,
            ),
            self.tooltip,
            self.source,
        )
        middle = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        payload = self.payload
        if payload is None and self.measurement:
            payload = self.measure(canvas.width, canvas.height)
        place_label(
            ctx,
            Rect(middle[0], middle[1], middle[0], middle[1]),
            self.label,
            self.score,
            payload,
            color,
            style,
            id(self),
        )
