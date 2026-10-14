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

from typing import ClassVar, Literal

import numpy as np

from luxonis_ml.vizlab.color import Color
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
from .fields import (
    FlowField,
    NormalMap,
    ScalarField,
    _hsv_to_rgb,
    normal_rgb,
)
from .heatmap import Heatmap
from .overlay import (
    Cell,
    ColorKey,
    Corner,
    InlinePiece,
    resolve_chrome,
    swatch_outline,
    text_piece,
)

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


class ColorBar(ColorKey):
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
        log: Whether the field is read on a log scale. The interior ticks are
            then spaced by ratio; `for_heatmap` also says ``· log`` in the
            title.
        clipped_low: Whether the field has values below ``vmin``; the low end
            label then reads ``≤``.
        clipped_high: Whether the field has values above ``vmax``; the high
            end label then reads ``≥``.

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
    log: bool = False
    clipped_low: bool = False
    clipped_high: bool = False
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
        resolved value range, which already excludes nodata pixels. An end
        that values lie beyond is marked, and a log-scale field gets a log key.

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
        clipped_low, clipped_high = heatmap.clipped_ends()
        fields.setdefault("clipped_low", clipped_low)
        fields.setdefault("clipped_high", clipped_high)
        log = getattr(heatmap, "log", False)
        fields.setdefault("log", log)
        if log and title:
            title = f"{title} \u00b7 log"
        return cls(
            gradient=gradient,
            vmin=low,
            vmax=high,
            title=title,
            **fields,  # pyright: ignore[reportArgumentType]
        )

    def _tick_values(self) -> "list[float]":
        """Space the labelled values evenly across the strip, left to right.

        On a log scale they are evenly spaced by ratio, as the colors are.
        """
        count = max(2, self.ticks)
        fractions = [index / (count - 1) for index in range(count)]
        if self.log and self.vmin > 0 and self.vmax > 0:
            ratio = self.vmax / self.vmin
            return [self.vmin * ratio**fraction for fraction in fractions]
        span = self.vmax - self.vmin
        return [self.vmin + span * fraction for fraction in fractions]

    def _ramp(self, ctx: RenderContext) -> np.ndarray:
        """Build the strip's pixels from the gradient the field is coloured by."""
        gradient = self.gradient or ctx.gradient or DEFAULT_GRADIENT
        rgb = resolve_gradient(gradient).colorize(
            np.linspace(0.0, 1.0, _RAMP)[None, :]
        )
        return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])

    def _tick_labels(self) -> "list[str]":
        """Label the ticks left to right, marking an end that clips."""
        values = self._tick_values()
        span = max(values) - min(values) if values else 0.0
        labels = [_tick_label(v, self.unit, span) for v in values]
        if self.clipped_low:
            labels[0] = f"\u2264{labels[0]}"
        if self.clipped_high:
            labels[-1] = f"\u2265{labels[-1]}"
        return labels

    def _inline_pieces(
        self, ctx: RenderContext, style: Style
    ) -> list[InlinePiece]:
        """Lay the key out as an axis: each tick label between two bars.

        The bar is split at the interior ticks, so every label sits on the
        line at its own place on the scale.
        """
        canvas = ctx.canvas
        chrome = resolve_chrome(ctx)
        size = style.font_size
        ramp = self._ramp(ctx)
        labels = self._tick_labels()
        segments = len(labels) - 1
        width = size * _INLINE_BAR / segments
        pieces = []
        if self.title:
            pieces.append(
                text_piece(
                    canvas, self.title, size, chrome.card_title, weight=700
                )
            )
        for index, label in enumerate(labels):
            pieces.append(
                text_piece(canvas, label, size, chrome.card_text, mono=True)
            )
            if index < segments:
                columns = ramp[
                    :,
                    round(index / segments * _RAMP) : round(
                        (index + 1) / segments * _RAMP
                    ),
                ]
                pieces.append(
                    _bar_piece(columns, width, size * 0.6, chrome.card_bg)
                )
        return pieces

    def _card_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        """Measure and draw the one card holding the strip and its labels."""
        canvas = ctx.canvas
        chrome = resolve_chrome(ctx)
        size = style.font_size
        strip_h = size * 0.9
        ramp = self._ramp(ctx)

        title = self.title
        title_m = (
            canvas.measure_markup(title, size * 1.05, weight=700)
            if title
            else None
        )
        labels = self._tick_labels()
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
            if title_m is not None and title:
                cv.markup(
                    (rect.left + _PAD, y + title_m.ascent),
                    title,
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


class FlowWheel(ColorKey):
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

    def _label(self) -> str:
        """Say what displacement the rim stands for."""
        return _tick_label(
            self.max_magnitude, self.unit, self.max_magnitude, digits=2
        )

    def _card_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        """Measure and draw the one card holding the disc and its label."""
        wheel = _flow_wheel(_WHEEL)
        return [_disc_card(ctx, style, self.title, wheel, self._label())]

    def _inline_pieces(
        self, ctx: RenderContext, style: Style
    ) -> list[InlinePiece]:
        """Lay the key out as the title, a small disc, and its label."""
        return _disc_pieces(
            ctx, style, self.title, _flow_wheel(_WHEEL), self._label()
        )


def _disc_card(
    ctx: RenderContext,
    style: Style,
    title: str | None,
    disc_rgba: np.ndarray,
    label: str,
) -> Cell:
    """Build the card of a disc key: a title, the disc, and a label under it."""
    canvas = ctx.canvas
    chrome = resolve_chrome(ctx)
    size = style.font_size
    disc = size * _WHEEL_SIZE
    title_m = (
        canvas.measure_markup(title, size * 1.05, weight=700)
        if title
        else None
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
        if title_m is not None and title:
            cv.markup(
                (rect.left + _PAD, y + title_m.ascent),
                title,
                size=size * 1.05,
                color=chrome.card_title,
                weight=700,
            )
            y += title_m.height + _ROW_GAP
        # Centre the disc and its label on the card, since neither fills the
        # width once a longer title has set it.
        cv.blit_scaled(
            disc_rgba, rect.left + _PAD + (inner - disc) / 2, y, disc, disc
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

    return Cell(card_w, card_h, _draw)


def _normal_sphere(size: int, facing: int) -> np.ndarray:
    """Render a sphere whose every pixel is colored by its own normal.

    Args:
        size: The width and height in pixels.
        facing: The sign of z on the hemisphere that faces the camera.

    Returns:
        An ``(size, size, 4)`` RGBA image, transparent outside the sphere.

    """
    axis = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    x, y = np.meshgrid(axis, axis)
    z = facing * np.sqrt(np.clip(1.0 - x**2 - y**2, 0.0, 1.0))
    rgb = normal_rgb(np.stack([x, y, z]))
    # Fade the last pixel-width of the rim, as the flow wheel does.
    edge = np.clip((1.0 - np.hypot(x, y)) * size * 0.5, 0.0, 1.0)
    return np.dstack([rgb, np.round(edge * 255).astype(np.uint8)])


class NormalKey(ColorKey):
    """The key of a `NormalMap`: a sphere colored by its own surface normals.

    A normal map colors a direction, so its key is the set of all directions
    a surface can face: a sphere seen from the camera, in the colors the map
    gives them. The label names the screen axes.

    Attributes:
        title: Heading above the sphere; ``None`` draws none.
        facing: The sign of z on surfaces that face the camera: ``-1`` when
            the camera looks along +z (OpenCV), ``1`` when z points at the
            viewer (OpenGL).

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import Image, NormalKey, NormalMap
        >>> normals = np.zeros((3, 4, 5), np.float32)
        >>> normals[2] = 1.0  # all facing the camera, OpenGL-style
        >>> key = NormalKey.for_field(NormalMap(values=normals))
        >>> key.facing
        1
        >>> Image(np.zeros((90, 90, 3), np.uint8)).add(key).render().shape
        (90, 90, 4)

    """

    title: str | None = None
    facing: Literal[-1, 1] = -1
    corner: Corner = Corner.BOTTOM_RIGHT

    #: Drawn over the image like every other corner overlay.
    OVERLAY: ClassVar[bool] = True

    @classmethod
    def for_field(
        cls, field: NormalMap, *, title: str | None = None, **fields: object
    ) -> "NormalKey":
        """Build the key that reads a given normal map.

        The sphere shows the hemisphere the field's surfaces face, read from
        the median z of the field.

        Args:
            field: The overlay this key describes.
            title: Heading above the sphere; ``None`` draws none.
            **fields: Any other `NormalKey` field, such as ``corner``.

        Returns:
            A `NormalKey` matching ``field``.

        """
        data = field.field()
        z = None if data is None or data.shape[0] < 3 else data[2]
        finite = np.zeros(0) if z is None else z[np.isfinite(z)]
        facing = 1 if finite.size and float(np.median(finite)) > 0 else -1
        return cls(
            title=title,
            facing=facing,
            **fields,  # pyright: ignore[reportArgumentType]
        )

    def _card_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        """Measure and draw the one card holding the sphere and its label."""
        sphere = _normal_sphere(_WHEEL, self.facing)
        return [_disc_card(ctx, style, self.title, sphere, _NORMAL_AXES)]

    def _inline_pieces(
        self, ctx: RenderContext, style: Style
    ) -> list[InlinePiece]:
        """Lay the key out as the title, a small sphere, and the axes."""
        sphere = _normal_sphere(_WHEEL, self.facing)
        return _disc_pieces(ctx, style, self.title, sphere, _NORMAL_AXES)


#: The length of an inline color bar, as a multiple of the type size.
_INLINE_BAR = 9.0
#: The diameter of an inline disc key, as a multiple of the type size.
_INLINE_DISC = 1.6
#: The label of a normal-map key: which way x and y point on screen.
_NORMAL_AXES = "x\u2192 right  y\u2193 down"


def _bar_piece(
    columns: np.ndarray, width: float, height: float, card_bg: Color
) -> InlinePiece:
    """Build an inline piece of a color bar from some columns of its ramp."""

    def draw(cv: Canvas, x: float, middle: float) -> None:
        top = middle - height / 2
        cv.blit_scaled(columns, x, top, width, height, radius=height / 2)
        cv.rounded_rect(
            Rect(x, top, x + width, top + height),
            radius=height / 2,
            stroke=swatch_outline(card_bg),
            stroke_width=1.0,
        )

    return InlinePiece(width, height, draw)


def _disc_pieces(
    ctx: RenderContext,
    style: Style,
    title: str | None,
    disc_rgba: np.ndarray,
    label: str,
) -> list[InlinePiece]:
    """Build the inline pieces of a disc key: the title, the disc, a label."""
    canvas = ctx.canvas
    chrome = resolve_chrome(ctx)
    size = style.font_size
    diameter = size * _INLINE_DISC

    def draw(cv: Canvas, x: float, middle: float) -> None:
        cv.blit_scaled(disc_rgba, x, middle - diameter / 2, diameter, diameter)

    pieces = []
    if title:
        pieces.append(
            text_piece(canvas, title, size, chrome.card_title, weight=700)
        )
    pieces.append(InlinePiece(diameter, diameter, draw))
    pieces.append(text_piece(canvas, label, size, chrome.card_text, mono=True))
    return pieces
