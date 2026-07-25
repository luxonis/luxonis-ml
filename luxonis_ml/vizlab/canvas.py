"""A thin, opinionated wrapper over a Skia raster surface.

`Canvas` exposes the handful of anti-aliased primitives the annotations
need — rounded rectangles with translucent fill and drop shadow, strokes,
polygons, circles, text, and mask overlays — and normalizes everything to and
from an ``(H, W, 4)`` RGBA ``uint8`` array so the rest of the library never
touches Skia types directly.
"""

from dataclasses import dataclass

import numpy as np
import skia

from .color import Color
from .fonts import DEFAULT_FONTS, FontManager
from .geometry import XY, Rect

_RGBA = skia.ColorType.kRGBA_8888_ColorType
_UNPREMUL = skia.AlphaType.kUnpremul_AlphaType


def _color4f(color: Color) -> skia.Color4f:
    """Convert a `Color` to a Skia ``Color4f``.

    Args:
        color: The color to convert.

    Returns:
        The equivalent ``skia.Color4f``.

    """
    return skia.Color4f(
        color.r / 255.0, color.g / 255.0, color.b / 255.0, color.a / 255.0
    )


_DEFAULT_SHADOW_COLOR = Color(0, 0, 0, 90)


@dataclass(frozen=True)
class Shadow:
    """A soft drop shadow for a filled shape.

    Attributes:
        color: Shadow color (usually semi-transparent black).
        blur: Gaussian blur sigma in pixels.
        dx: Horizontal offset in pixels.
        dy: Vertical offset in pixels.

    """

    color: Color = _DEFAULT_SHADOW_COLOR
    blur: float = 6.0
    dx: float = 0.0
    dy: float = 2.0


@dataclass(frozen=True)
class TextMetrics:
    """Measured extent of a run of text.

    Attributes:
        width: Advance width in pixels.
        ascent: Distance from baseline to the top (positive, pixels).
        descent: Distance from baseline to the bottom (positive, pixels).

    """

    width: float
    ascent: float
    descent: float

    @property
    def height(self) -> float:
        """Total line height in pixels, i.e. ascent plus descent."""
        return self.ascent + self.descent


class Canvas:
    """A drawable RGBA raster backed by a Skia surface."""

    def __init__(
        self, surface: skia.Surface, fonts: FontManager = DEFAULT_FONTS
    ) -> None:
        """Wrap an existing Skia surface.

        Prefer `blank` or `from_rgba` over calling this directly.

        Args:
            surface: The Skia surface to draw on.
            fonts: Font manager used for text primitives.

        """
        self._surface = surface
        self._canvas = surface.getCanvas()
        self._fonts = fonts

    @classmethod
    def blank(
        cls, width: int, height: int, fonts: FontManager = DEFAULT_FONTS
    ) -> "Canvas":
        """Create a transparent canvas of the given size.

        Args:
            width: Width in pixels.
            height: Height in pixels.
            fonts: Font manager used for text primitives.

        Returns:
            A new, fully transparent `Canvas`.

        """
        info = skia.ImageInfo.Make(int(width), int(height), _RGBA, _UNPREMUL)
        surface = skia.Surface.MakeRaster(info)
        canvas = cls(surface, fonts)
        canvas._canvas.clear(skia.Color4f(0, 0, 0, 0))
        return canvas

    @classmethod
    def from_rgba(
        cls, rgba: np.ndarray, fonts: FontManager = DEFAULT_FONTS
    ) -> "Canvas":
        """Create a canvas initialized with an RGBA image.

        Args:
            rgba: An ``(H, W, 4)`` ``uint8`` array in RGBA order.
            fonts: Font manager used for text primitives.

        Returns:
            A new `Canvas` with ``rgba`` drawn as its background.

        """
        height, width = rgba.shape[:2]
        canvas = cls.blank(width, height, fonts)
        image = skia.Image.fromarray(
            np.ascontiguousarray(rgba), colorType=_RGBA
        )
        canvas._canvas.drawImage(image, 0, 0)
        return canvas

    @property
    def width(self) -> int:
        """Canvas width in pixels."""
        return self._surface.width()

    @property
    def height(self) -> int:
        """Canvas height in pixels."""
        return self._surface.height()

    def to_rgba(self) -> np.ndarray:
        """Snapshot the current canvas as an RGBA array.

        Returns:
            An ``(H, W, 4)`` ``uint8`` array in RGBA order.

        """
        image = self._surface.makeImageSnapshot()
        return image.toarray(colorType=_RGBA)

    def scaled(self, width: int, height: int) -> "Canvas":
        """Return a new canvas with the current content scaled to ``(width, height)``.

        Used to render pixel-heavy fills at the source resolution and then scale
        the whole raster once to the display size, before crisp vector content
        (strokes, labels) is drawn on top at that size.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            A new `Canvas`; the original is left unchanged. Returns a snapshot
            copy at the same size when the dimensions already match.

        """
        image = self._surface.makeImageSnapshot()
        out = Canvas.blank(int(width), int(height), self._fonts)
        out._canvas.drawImageRect(
            image,
            skia.Rect.MakeWH(self.width, self.height),
            skia.Rect.MakeWH(float(width), float(height)),
            skia.SamplingOptions(skia.FilterMode.kLinear),
            skia.Paint(AntiAlias=True),
        )
        return out

    def blit(self, rgba: np.ndarray, x: float, y: float) -> None:
        """Draw an RGBA image onto the canvas with its top-left at ``(x, y)``.

        Args:
            rgba: An ``(H, W, 4)`` ``uint8`` RGBA array.
            x: Destination left in pixels.
            y: Destination top in pixels.

        """
        image = skia.Image.fromarray(
            np.ascontiguousarray(rgba), colorType=_RGBA
        )
        self._canvas.drawImage(image, float(x), float(y))

    # -- primitives ---------------------------------------------------------

    def _fill_paint(self, color: Color) -> skia.Paint:
        """Build an anti-aliased fill paint.

        Args:
            color: Fill color.

        Returns:
            A configured fill ``skia.Paint``.

        """
        return skia.Paint(
            Color=_color4f(color), AntiAlias=True, Style=skia.Paint.kFill_Style
        )

    def _stroke_paint(
        self,
        color: Color,
        width: float,
        dash: tuple[float, float] | None = None,
    ) -> skia.Paint:
        """Build an anti-aliased stroke paint, optionally dashed.

        Args:
            color: Stroke color.
            width: Stroke width in pixels.
            dash: ``(on, off)`` dash lengths in pixels, or ``None`` for a solid line.

        Returns:
            A configured stroke ``skia.Paint``.

        """
        paint = skia.Paint(
            Color=_color4f(color),
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=float(width),
            # Butt caps keep dash gaps crisp; round caps would bridge them.
            StrokeCap=skia.Paint.kButt_Cap if dash else skia.Paint.kRound_Cap,
            StrokeJoin=skia.Paint.kRound_Join,
        )
        if dash is not None:
            paint.setPathEffect(
                skia.DashPathEffect.Make([float(dash[0]), float(dash[1])], 0.0)
            )
        return paint

    def rounded_rect(
        self,
        rect: Rect,
        radius: float = 0.0,
        *,
        fill: Color | None = None,
        stroke: Color | None = None,
        stroke_width: float = 2.0,
        dash: tuple[float, float] | None = None,
        shadow: Shadow | None = None,
    ) -> None:
        """Draw a rounded rectangle with optional fill, stroke, and shadow.

        Args:
            rect: The rectangle to draw.
            radius: Corner radius in pixels.
            fill: Fill color, or ``None`` for no fill.
            stroke: Stroke color, or ``None`` for no stroke.
            stroke_width: Stroke width in pixels.
            dash: ``(on, off)`` dash lengths for the stroke, or ``None`` for solid.
            shadow: Optional drop shadow drawn behind the shape.

        """
        rrect = skia.RRect.MakeRectXY(
            skia.Rect.MakeLTRB(rect.left, rect.top, rect.right, rect.bottom),
            radius,
            radius,
        )
        if shadow is not None:
            paint = self._fill_paint(shadow.color)
            paint.setMaskFilter(
                skia.MaskFilter.MakeBlur(
                    skia.BlurStyle.kNormal_BlurStyle, shadow.blur
                )
            )
            self._canvas.save()
            self._canvas.translate(shadow.dx, shadow.dy)
            self._canvas.drawRRect(rrect, paint)
            self._canvas.restore()
        if fill is not None:
            self._canvas.drawRRect(rrect, self._fill_paint(fill))
        if stroke is not None:
            self._canvas.drawRRect(
                rrect, self._stroke_paint(stroke, stroke_width, dash)
            )

    def polygon(
        self,
        points: list[XY],
        *,
        fill: Color | None = None,
        stroke: Color | None = None,
        stroke_width: float = 2.0,
        dash: tuple[float, float] | None = None,
        closed: bool = True,
    ) -> None:
        """Draw a polygon or polyline.

        Args:
            points: Vertices as ``(x, y)`` points.
            fill: Fill color (ignored when ``closed`` is ``False``).
            stroke: Stroke color, or ``None`` for no stroke.
            stroke_width: Stroke width in pixels.
            dash: ``(on, off)`` dash lengths for the stroke, or ``None`` for solid.
            closed: Whether to close the path back to the first point.

        """
        if len(points) < 2:
            return
        path = skia.Path()
        path.moveTo(*points[0])
        for pt in points[1:]:
            path.lineTo(*pt)
        if closed:
            path.close()
        if fill is not None and closed:
            self._canvas.drawPath(path, self._fill_paint(fill))
        if stroke is not None:
            self._canvas.drawPath(
                path, self._stroke_paint(stroke, stroke_width, dash)
            )

    def line(self, p1: XY, p2: XY, color: Color, width: float = 2.0) -> None:
        """Draw a straight line segment.

        Args:
            p1: Start point.
            p2: End point.
            color: Line color.
            width: Line width in pixels.

        """
        self._canvas.drawLine(
            p1[0], p1[1], p2[0], p2[1], self._stroke_paint(color, width)
        )

    def circle(
        self,
        center: XY,
        radius: float,
        *,
        fill: Color | None = None,
        stroke: Color | None = None,
        stroke_width: float = 2.0,
    ) -> None:
        """Draw a circle with optional fill and stroke.

        Args:
            center: Center point.
            radius: Radius in pixels.
            fill: Fill color, or ``None`` for no fill.
            stroke: Stroke color, or ``None`` for no stroke.
            stroke_width: Stroke width in pixels.

        """
        if fill is not None:
            self._canvas.drawCircle(
                center[0], center[1], radius, self._fill_paint(fill)
            )
        if stroke is not None:
            self._canvas.drawCircle(
                center[0],
                center[1],
                radius,
                self._stroke_paint(stroke, stroke_width),
            )

    def measure_text(
        self, text: str, size: float, *, weight: int = 400
    ) -> TextMetrics:
        """Measure a run of text without drawing it.

        Args:
            text: The string to measure.
            size: Text size in pixels.
            weight: OpenType weight (100-900).

        Returns:
            The `TextMetrics` for the run.

        """
        font = self._fonts.font(size, weight=weight)
        width = font.measureText(text)
        metrics = font.getMetrics()
        return TextMetrics(
            width=width, ascent=-metrics.fAscent, descent=metrics.fDescent
        )

    def text(
        self,
        origin: XY,
        text: str,
        *,
        size: float,
        color: Color,
        weight: int = 400,
    ) -> None:
        """Draw text with its baseline-left at ``origin``.

        Args:
            origin: The ``(x, y)`` baseline-left position.
            text: The string to draw.
            size: Text size in pixels.
            color: Text color.
            weight: OpenType weight (100-900).

        """
        font = self._fonts.font(size, weight=weight)
        paint = skia.Paint(Color=_color4f(color), AntiAlias=True)
        self._canvas.drawString(text, origin[0], origin[1], font, paint)

    def overlay_mask(
        self,
        mask: np.ndarray,
        color: Color,
        *,
        alpha: float = 0.45,
    ) -> None:
        """Blend a translucent color wherever a mask is set.

        Args:
            mask: An ``(H, W)`` array; non-zero (or ``> 0.5`` for floats) pixels are
                filled. Must match the canvas size.
            color: Fill color for the mask region.
            alpha: Opacity of the fill in ``[0, 1]``.

        """
        m = mask > 0.5 if mask.dtype.kind == "f" else mask.astype(bool)
        rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        rgba[m] = (
            color.r,
            color.g,
            color.b,
            round(max(0.0, min(1.0, alpha)) * 255),
        )
        image = skia.Image.fromarray(
            np.ascontiguousarray(rgba), colorType=_RGBA
        )
        self._canvas.drawImage(image, 0, 0)
