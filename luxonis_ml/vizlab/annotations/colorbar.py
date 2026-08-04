"""A colour key: the gradient strip that makes a `Heatmap`'s colours readable.

A `Heatmap` maps values to colours, but nothing on the image says *which* value
a colour stands for. `ColorBar` is that missing half — the continuous
counterpart to `Legend`, which keys the discrete class palette. It draws the
same gradient as a strip, labelled with the values at its ends, on the same
theme-aware card as every other overlay.

Build one straight from the field it describes with `ColorBar.for_heatmap`, so
the key and the overlay can never disagree about either the gradient or the
range.

.. image:: TODO-HOST/color_key.png
   :alt: A colour key with bare ends, with a unit and interior ticks, and
         stacked above a class legend in the same corner.
"""

from typing import ClassVar

import numpy as np

from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.gradient import (
    DEFAULT_DIVERGING_GRADIENT,
    DEFAULT_GRADIENT,
    Gradient,
    resolve_gradient,
)
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.style import Style

from .base import RenderContext
from .card import draw_card_background
from .distribution.metrics import _PAD, _ROW_GAP
from .fields import FlowField, ScalarField, _hsv_to_rgb
from .heatmap import Heatmap
from .overlay import Cell, Corner, CornerStack, resolve_chrome, swatch_outline

#: Samples across the gradient; enough that the strip reads as continuous.
_RAMP = 256

#: Minimum strip width, as a multiple of the type size.
_MIN_STRIP = 9.0


def _tick_label(
    value: float, unit: str | None, span: float = 0.0, digits: int = 4
) -> str:
    """Format a tick value compactly, matching the panel's scalar style.

    A tick is read against the strip's own range, so ``span`` decides what
    counts as noise: a value many orders of magnitude below the range is the
    tail of a float rather than a quantity, and rendering it as ``1.313e-10``
    tells a reader nothing except that the axis is unconsidered. Anything under
    a millionth of the span is labelled ``0``. A span of ``0`` — an unknown or
    degenerate range — formats the value as given.

    Args:
        value: The value the tick stands for.
        unit: Appended to the number, or ``None``.
        span: The strip's full range, used to recognize float noise.
        digits: Significant figures to keep.

    Returns:
        The tick's text.

    Examples:
        >>> _tick_label(1.3130042e-10, None, span=1.0)
        '0'
        >>> _tick_label(1.3130042e-10, None, span=4e-10)
        '1.313e-10'
        >>> _tick_label(0.5, "px", span=1.0)
        '0.5 px'

    """
    if span > 0.0 and abs(value) < span * 1e-6:
        value = 0.0
    text = f"{value:.{digits}g}"
    return f"{text} {unit}" if unit else text


class ColorBar(CornerStack):
    """A gradient strip labelled with the values its ends stand for.

    Sits in an image corner like `Legend`, and stacks with it rather than over
    it. The strip is built from the same `Gradient` call `Heatmap` colours its
    field with, so the key is never an approximation of the overlay.

    Attributes:
        gradient: A `Gradient`, or the name of a preset. ``None`` (the default)
            inherits the render options' gradient, exactly as `Heatmap` does, so
            a key and a field left unset always agree.
        vmin: The value at the strip's left end.
        vmax: The value at the strip's right end.
        title: Heading above the strip; ``None`` draws none.
        unit: Appended to every tick label, e.g. ``"px"``.
        ticks: How many labels to draw, ``2`` being the two ends. Three or more
            adds evenly spaced interior ticks.

    See `Annotation` for the shared fields (``color``/``palette`` do not apply —
    the colours come from the gradient).

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import ColorBar, Heatmap, Image
        >>> field = np.linspace(0.0, 40.0, 6 * 8).reshape(6, 8)
        >>> heat = Heatmap(values=field, gradient="viridis")
        >>> key = ColorBar.for_heatmap(heat, title="depth", unit="m")
        >>> (key.vmin, key.vmax)
        (0.0, 40.0)
        >>> Image(np.zeros((60, 90, 3), np.uint8)).add(heat).add(
        ...     key
        ... ).render().shape
        (60, 90, 4)

    """

    gradient: Gradient | str | None = None
    vmin: float = 0.0
    vmax: float = 1.0
    title: str | None = None
    unit: str | None = None
    ticks: int = 2
    corner: Corner = Corner.BOTTOM_RIGHT

    #: Drawn over the image like every other corner overlay.
    OVERLAY: ClassVar[bool] = True

    @classmethod
    def for_heatmap(
        cls,
        heatmap: "Heatmap | ScalarField",
        *,
        title: str | None = None,
        **fields: object,
    ) -> "ColorBar":
        """Build the key that reads a given field.

        Takes the field's gradient *as-is* — when both are ``None`` the two
        inherit the same context gradient, so they cannot drift apart — and its
        resolved value range, which already excludes nodata pixels.

        A centered field gets an odd number of ticks by default, so one of them
        lands exactly on the center and the neutral color is labelled rather
        than left to be inferred.

        Args:
            heatmap: The overlay this key describes — a `Heatmap` or a
                `ScalarField`, which share the two members this needs.
            title: Heading for the strip; ``None`` draws none.
            **fields: Any other `ColorBar` field, such as ``corner`` or ``unit``.

        Returns:
            A `ColorBar` matching ``heatmap``.

        """
        low, high = heatmap.value_range()
        gradient = heatmap.gradient
        center = getattr(heatmap, "center", None)
        if gradient is None and center is not None:
            # Match ScalarField's own diverging fallback, or the key would show
            # a different colormap than the field it describes.
            gradient = DEFAULT_DIVERGING_GRADIENT
        if center is not None:
            fields.setdefault("ticks", 3)
        return cls(
            gradient=gradient,
            vmin=low,
            vmax=high,
            title=title,
            **fields,  # pyright: ignore[reportArgumentType]
        )

    def _tick_values(self) -> "list[float]":
        """Space the labelled values evenly across the strip, left to right."""
        count = max(2, self.ticks)
        span = self.vmax - self.vmin
        return [
            self.vmin + span * (index / (count - 1)) for index in range(count)
        ]

    def _ramp(self, ctx: RenderContext) -> np.ndarray:
        """Build the strip's pixels from the gradient the field is coloured by."""
        gradient = self.gradient or ctx.gradient or DEFAULT_GRADIENT
        rgb = resolve_gradient(gradient).colorize(
            np.linspace(0.0, 1.0, _RAMP)[None, :]
        )
        return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        """Measure and draw the one card holding the strip and its labels."""
        canvas = ctx.canvas
        chrome = resolve_chrome(ctx)
        size = style.font_size
        strip_h = size * 0.9
        ramp = self._ramp(ctx)

        title_m = (
            canvas.measure_markup(self.title, size * 1.05, weight=700)
            if self.title
            else None
        )
        values = self._tick_values()
        span = max(values) - min(values) if values else 0.0
        labels = [_tick_label(v, self.unit, span) for v in values]
        label_ms = [
            canvas.measure_markup(text, size, mono=True) for text in labels
        ]
        # Ticks sit under the strip, so they must all fit side by side with a
        # gap between them. Sizing by the *widest* label rather than by their
        # sum is what keeps an interior tick clear of its neighbours: interior
        # labels are centred on their fraction, so a wide end label (a minus
        # sign is enough) would otherwise run into the one beside it.
        widest = max(m.width for m in label_ms)
        strip_w = max(
            size * _MIN_STRIP,
            widest * len(label_ms) + _ROW_GAP * (len(label_ms) - 1),
            title_m.width if title_m else 0.0,
        )
        label_h = max(m.height for m in label_ms)
        card_w = strip_w + 2 * _PAD
        card_h = (
            2 * _PAD
            + (title_m.height + _ROW_GAP if title_m else 0.0)
            + strip_h
            + _ROW_GAP
            + label_h
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            draw_card_background(cv, rect, style, chrome)
            y = rect.top + _PAD
            if title_m is not None and self.title:
                cv.markup(
                    (rect.left + _PAD, y + title_m.ascent),
                    self.title,
                    size=size * 1.05,
                    color=chrome.card_title,
                    weight=700,
                )
                y += title_m.height + _ROW_GAP
            left = rect.left + _PAD
            cv.blit_scaled(ramp, left, y, strip_w, strip_h, radius=strip_h / 2)
            cv.rounded_rect(
                Rect(left, y, left + strip_w, y + strip_h),
                radius=strip_h / 2,
                stroke=swatch_outline(chrome.card_bg),
                stroke_width=1.0,
            )
            y += strip_h + _ROW_GAP
            # First label flush left, last flush right, the rest centred on
            # their fraction, so the strip reads as a real axis.
            last = len(labels) - 1
            for index, (text, metrics) in enumerate(
                zip(labels, label_ms, strict=True)
            ):
                fraction = index / last if last else 0.0
                x = left + fraction * strip_w - fraction * metrics.width
                cv.markup(
                    (x, y + metrics.ascent),
                    text,
                    size=size,
                    color=chrome.card_text,
                    mono=True,
                )

        return [Cell(card_w, card_h, _draw)]


#: Pixels across the rendered wheel. Drawn once per key and scaled to fit, so it
#: only has to be fine enough that the hue sweep reads as continuous.
_WHEEL = 128

#: Wheel diameter, as a multiple of the type size.
_WHEEL_SIZE = 5.0


def _flow_wheel(size: int) -> np.ndarray:
    """Render the direction disc: hue around, saturation outwards."""
    axis = np.linspace(-1.0, 1.0, size)
    dx, dy = np.meshgrid(axis, axis)
    radius = np.hypot(dx, dy)
    # Matches FlowField: screen y runs downwards, so negate it to make the hue
    # follow the direction a viewer sees motion travel.
    hue = (np.arctan2(-dy, dx) / (2.0 * np.pi)) % 1.0
    rgb = _hsv_to_rgb(hue, np.clip(radius, 0.0, 1.0), np.ones_like(radius))
    # Fade the last pixel-width of the rim rather than cutting it, so the disc
    # does not alias into a polygon at small sizes.
    edge = np.clip((1.0 - radius) * size * 0.5, 0.0, 1.0)
    return np.dstack([rgb, np.round(edge * 255).astype(np.uint8)])


class FlowWheel(CornerStack):
    """The direction key for a `FlowField`: a disc of hue by heading.

    `ColorBar`'s counterpart for a two-channel field. A strip cannot key one —
    flow's color encodes an angle, which wraps, so the scale is a circle rather
    than a line. The radius carries magnitude, and the label under the disc says
    what displacement the rim stands for.

    Attributes:
        max_magnitude: Displacement the rim stands for, in pixels.
        title: Heading above the disc; ``None`` draws none.
        unit: Appended to the magnitude label.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import FlowField, FlowWheel, Image
        >>> flow = np.zeros((2, 4, 5), np.float32)
        >>> flow[0] = 3.0
        >>> field = FlowField(values=flow)
        >>> key = FlowWheel.for_field(field, title="motion")
        >>> key.max_magnitude
        3.0
        >>> Image(np.zeros((60, 90, 3), np.uint8)).add(field).add(
        ...     key
        ... ).render().shape
        (60, 90, 4)

    """

    max_magnitude: float = 1.0
    title: str | None = None
    unit: str | None = "px"
    corner: Corner = Corner.BOTTOM_RIGHT

    #: Drawn over the image like every other corner overlay.
    OVERLAY: ClassVar[bool] = True

    @classmethod
    def for_field(
        cls, field: FlowField, *, title: str | None = None, **fields: object
    ) -> "FlowWheel":
        """Build the key that reads a given flow field.

        Args:
            field: The overlay this key describes.
            title: Heading for the disc; ``None`` draws none.
            **fields: Any other `FlowWheel` field, such as ``corner``.

        Returns:
            A `FlowWheel` matching ``field``.

        """
        return cls(
            max_magnitude=field.peak_magnitude(),
            title=title,
            **fields,  # pyright: ignore[reportArgumentType]
        )

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        """Measure and draw the one card holding the disc and its label."""
        canvas = ctx.canvas
        chrome = resolve_chrome(ctx)
        size = style.font_size
        disc = size * _WHEEL_SIZE
        wheel = _flow_wheel(_WHEEL)

        title_m = (
            canvas.measure_markup(self.title, size * 1.05, weight=700)
            if self.title
            else None
        )
        label = _tick_label(
            self.max_magnitude, self.unit, self.max_magnitude, digits=2
        )
        label_m = canvas.measure_markup(label, size, mono=True)
        inner = max(disc, label_m.width, title_m.width if title_m else 0.0)
        card_w = inner + 2 * _PAD
        card_h = (
            2 * _PAD
            + (title_m.height + _ROW_GAP if title_m else 0.0)
            + disc
            + _ROW_GAP
            + label_m.height
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            draw_card_background(cv, rect, style, chrome)
            y = rect.top + _PAD
            if title_m is not None and self.title:
                cv.markup(
                    (rect.left + _PAD, y + title_m.ascent),
                    self.title,
                    size=size * 1.05,
                    color=chrome.card_title,
                    weight=700,
                )
                y += title_m.height + _ROW_GAP
            # Centre the disc and its label on the card, since neither fills the
            # width once a longer title has set it.
            cv.blit_scaled(
                wheel,
                rect.left + _PAD + (inner - disc) / 2,
                y,
                disc,
                disc,
            )
            y += disc + _ROW_GAP
            cv.markup(
                (
                    rect.left + _PAD + (inner - label_m.width) / 2,
                    y + label_m.ascent,
                ),
                label,
                size=size,
                color=chrome.card_text,
                mono=True,
            )

        return [Cell(card_w, card_h, _draw)]
