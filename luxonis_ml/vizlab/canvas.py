"""A thin, opinionated wrapper over a Skia raster surface.

`Canvas` exposes the handful of anti-aliased primitives the annotations
need — rounded rectangles with translucent fill and drop shadow, strokes,
polygons, circles, text, and mask overlays — and normalizes everything to and
from an ``(H, W, 4)`` RGBA ``uint8`` array so the rest of the library never
touches Skia types directly.
"""

import contextlib
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import skia

from .color import Color
from .fonts import DEFAULT_FONTS, FontManager
from .geometry import XY, Rect
from .markup import Span

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


def _skia_color(color: Color) -> int:
    """Convert a `Color` to a packed Skia ``SkColor`` integer (for shaders)."""
    return skia.Color(color.r, color.g, color.b, color.a)


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


def _break_word(
    word: str, width_of: Callable[[str], float], limit: float
) -> list[str]:
    """Hard-break a word into chunks that each fit within ``limit`` pixels."""
    if len(word) <= 1 or width_of(word) <= limit:
        return [word]
    pieces: list[str] = []
    start = 0
    while start < len(word):
        end = start + 1
        while end < len(word) and width_of(word[start : end + 1]) <= limit:
            end += 1
        pieces.append(word[start:end])
        start = end
    return pieces


def _wrap_words(
    words: list[str], width_of: Callable[[str], float], limit: float
) -> list[str]:
    """Greedily pack ``words`` into lines no wider than ``limit`` pixels."""
    lines: list[str] = []
    current = ""
    for word in words:
        for piece in _break_word(word, width_of, limit):
            candidate = piece if not current else f"{current} {piece}"
            if not current or width_of(candidate) <= limit:
                current = candidate
            else:
                lines.append(current)
                current = piece
    lines.append(current)
    return lines


@dataclass
class _WrapState:
    """Mutable accumulator for `Canvas.wrap_spans`: built lines and the open one."""

    lines: "list[list[Span]]" = field(default_factory=list)
    current: "list[Span]" = field(default_factory=list)
    width: float = 0.0
    gap: bool = False  # a pending space before the next word

    def newline(self) -> None:
        """Flush the open line and start an empty one."""
        self.lines.append(self.current)
        self.current, self.width, self.gap = [], 0.0, False


def _span_atoms(
    spans: Sequence[Span],
) -> "list[tuple[Span | None, bool]]":
    """Break spans into wrap atoms: ``(word, _)`` or ``(None, is_newline)``.

    A ``word`` atom is a whitespace-free run carrying its span's style; a
    ``None`` atom is a separator — a forced line break when its flag is set,
    otherwise a collapsible space.
    """
    atoms: list[tuple[Span | None, bool]] = []
    for span in spans:
        for part in re.split(r"(\s+)", span.text):
            if not part:
                continue
            if part.isspace():
                atoms.append((None, "\n" in part))
            else:
                atoms.append(
                    (Span(part, span.weight, span.italic, span.mono), False)
                )
    return atoms


class Canvas:
    """A drawable Skia canvas backed by a raster buffer or an SVG document.

    The drawing primitives are backend-agnostic — the same calls paint pixels on
    a raster surface or record vector elements into an SVG document. Which one a
    canvas is depends on how it was built: `blank` / `from_rgba` make a raster
    canvas (snapshot it with `to_rgba`), `svg` makes an SVG one (finalize it with
    `finish_svg`).
    """

    def __init__(
        self,
        canvas: skia.Canvas,
        size: tuple[int, int],
        fonts: FontManager = DEFAULT_FONTS,
        *,
        antialias: bool = True,
        surface: "skia.Surface | None" = None,
        svg_stream: "skia.DynamicMemoryWStream | None" = None,
    ) -> None:
        """Wrap a Skia drawing canvas together with its backing output.

        Prefer the `blank`, `from_rgba`, and `svg` constructors over calling this
        directly. Exactly one backing is given: ``surface`` for a raster canvas or
        ``svg_stream`` for an SVG one; it decides which of `to_rgba` / `finish_svg`
        the canvas supports.

        Args:
            canvas: The Skia canvas to issue draw calls to.
            size: The ``(width, height)`` of the canvas in pixels.
            fonts: Font manager used for text primitives.
            antialias: Whether shape fills and strokes are anti-aliased. ``False``
                gives jagged edges but noticeably faster rasterization of dense
                scenes; text stays anti-aliased regardless, so labels stay legible.
            surface: The backing raster surface, for a raster canvas.
            svg_stream: The backing SVG stream, for an SVG canvas.

        """
        self._canvas: skia.Canvas = canvas
        self._size = (int(size[0]), int(size[1]))
        self._fonts = fonts
        self._antialias = antialias
        self._surface = surface
        self._svg_stream = svg_stream

    @classmethod
    def blank(
        cls,
        width: int,
        height: int,
        fonts: FontManager = DEFAULT_FONTS,
        *,
        antialias: bool = True,
    ) -> "Canvas":
        """Create a transparent canvas of the given size.

        Args:
            width: Width in pixels.
            height: Height in pixels.
            fonts: Font manager used for text primitives.
            antialias: Whether shape fills and strokes are anti-aliased.

        Returns:
            A new, fully transparent `Canvas`.

        """
        info = skia.ImageInfo.Make(int(width), int(height), _RGBA, _UNPREMUL)
        surface = skia.Surface.MakeRaster(info)
        canvas = cls(
            surface.getCanvas(),
            (int(width), int(height)),
            fonts,
            antialias=antialias,
            surface=surface,
        )
        canvas._canvas.clear(skia.Color4f(0, 0, 0, 0))
        return canvas

    @classmethod
    def svg(
        cls,
        width: int,
        height: int,
        fonts: FontManager = DEFAULT_FONTS,
        *,
        antialias: bool = True,
        text_as_paths: bool = True,
    ) -> "Canvas":
        """Create an SVG-recording canvas of the given size.

        Draw primitives (rectangles, polygons, text, lines) record as real SVG
        vector elements; blitted rasters (a base photo, a mask-fill patch) embed
        as base64 ``<image>`` elements. Finalize with `finish_svg` to get the SVG
        document bytes. Unlike a raster canvas this has no pixel buffer, so
        `to_rgba`, `scaled`, and snapshots are unavailable.

        Args:
            width: Viewport width in pixels.
            height: Viewport height in pixels.
            fonts: Font manager used for text primitives.
            antialias: Whether shape fills/strokes set the anti-alias paint flag
                (SVG renderers anti-alias regardless; kept for API parity).
            text_as_paths: Emit glyphs as ``<path>`` outlines rather than
                ``<text>``, so the file renders identically anywhere without the
                fonts installed. On by default; turn off for smaller,
                text-selectable output that depends on the viewer's fonts.

        Returns:
            A new SVG-backed `Canvas`.

        """
        stream = skia.DynamicMemoryWStream()
        # 0x01 = SkSVGCanvas::kConvertTextToPaths_Flag (skia-python exposes no
        # named constant), so glyphs become self-contained path outlines.
        flags = 0x01 if text_as_paths else 0
        svg_canvas = skia.SVGCanvas.Make(
            skia.Rect.MakeWH(int(width), int(height)), stream, flags
        )
        return cls(
            svg_canvas,
            (int(width), int(height)),
            fonts,
            antialias=antialias,
            svg_stream=stream,
        )

    @classmethod
    def from_rgba(
        cls,
        rgba: np.ndarray,
        fonts: FontManager = DEFAULT_FONTS,
        *,
        antialias: bool = True,
    ) -> "Canvas":
        """Create a canvas initialized with an RGBA image.

        Args:
            rgba: An ``(H, W, 4)`` ``uint8`` array in RGBA order.
            fonts: Font manager used for text primitives.
            antialias: Whether shape fills and strokes are anti-aliased.

        Returns:
            A new `Canvas` with ``rgba`` drawn as its background.

        """
        height, width = rgba.shape[:2]
        canvas = cls.blank(width, height, fonts, antialias=antialias)
        image = skia.Image.fromarray(
            np.ascontiguousarray(rgba), colorType=_RGBA
        )
        canvas._canvas.drawImage(image, 0, 0)
        return canvas

    @property
    def width(self) -> int:
        """Canvas width in pixels."""
        return self._size[0]

    @property
    def height(self) -> int:
        """Canvas height in pixels."""
        return self._size[1]

    def to_rgba(self) -> np.ndarray:
        """Snapshot the current canvas as an RGBA array.

        Returns:
            An ``(H, W, 4)`` ``uint8`` array in RGBA order.

        """
        if self._surface is None:
            raise ValueError("to_rgba is unavailable on an SVG canvas")
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
        if self._surface is None:
            raise ValueError("scaled is unavailable on an SVG canvas")
        image = self._surface.makeImageSnapshot()
        out = Canvas.blank(
            int(width), int(height), self._fonts, antialias=self._antialias
        )
        out._canvas.drawImageRect(
            image,
            skia.Rect.MakeWH(self.width, self.height),
            skia.Rect.MakeWH(float(width), float(height)),
            skia.SamplingOptions(skia.FilterMode.kLinear),
            skia.Paint(AntiAlias=True),
        )
        return out

    def blit(
        self, rgba: np.ndarray, x: float, y: float, *, radius: float = 0.0
    ) -> None:
        """Draw an RGBA image onto the canvas with its top-left at ``(x, y)``.

        Args:
            rgba: An ``(H, W, 4)`` ``uint8`` RGBA array.
            x: Destination left in pixels.
            y: Destination top in pixels.
            radius: Corner radius in pixels; ``> 0`` clips the image to a rounded
                rectangle (anti-aliased) so it reads as a rounded surface.

        """
        image = skia.Image.fromarray(
            np.ascontiguousarray(rgba), colorType=_RGBA
        )
        if radius > 0.0:
            height, width = rgba.shape[:2]
            rrect = skia.RRect.MakeRectXY(
                skia.Rect.MakeXYWH(float(x), float(y), width, height),
                radius,
                radius,
            )
            self._canvas.save()
            self._canvas.clipRRect(rrect, doAntiAlias=True)
            self._canvas.drawImage(image, float(x), float(y))
            self._canvas.restore()
        else:
            self._canvas.drawImage(image, float(x), float(y))

    def blit_scaled(
        self,
        rgba: np.ndarray,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        radius: float = 0.0,
    ) -> None:
        """Draw an RGBA raster into the rect ``(x, y, width, height)``, scaled.

        The backend-agnostic way to lay a sub-image's raster down: a raster canvas
        resamples it into the rect, an SVG canvas embeds it as one ``<image>`` with
        that placement. Used to embed a photo-plus-fills base under a crisp vector
        layer, at native size or scaled to a display box.

        Args:
            rgba: An ``(H, W, 4)`` ``uint8`` RGBA array (any size).
            x: Destination left in pixels.
            y: Destination top in pixels.
            width: Destination width in pixels.
            height: Destination height in pixels.
            radius: Corner radius in pixels; ``> 0`` clips to a rounded rectangle.

        """
        image = skia.Image.fromarray(
            np.ascontiguousarray(rgba), colorType=_RGBA
        )
        dst = skia.Rect.MakeXYWH(
            float(x), float(y), float(width), float(height)
        )
        clipped = radius > 0.0
        if clipped:
            self._canvas.save()
            self._canvas.clipRRect(
                skia.RRect.MakeRectXY(dst, radius, radius), doAntiAlias=True
            )
        self._canvas.drawImageRect(
            image,
            skia.Rect.MakeWH(rgba.shape[1], rgba.shape[0]),
            dst,
            skia.SamplingOptions(skia.FilterMode.kLinear),
            skia.Paint(AntiAlias=True),
        )
        if clipped:
            self._canvas.restore()

    def draw_base(self, rgba: np.ndarray) -> None:
        """Draw an RGBA raster scaled to fill the whole canvas (the base layer)."""
        self.blit_scaled(rgba, 0.0, 0.0, self.width, self.height)

    @contextlib.contextmanager
    def clip_rounded(self, rect: Rect, radius: float) -> "Iterator[None]":
        """Clip drawing to a rounded rectangle for the duration of the block.

        Used to draw a sub-scene (e.g. a composited image) with rounded corners:
        everything drawn inside the ``with`` is masked to the rounded ``rect``.

        Args:
            rect: The rectangle to clip to.
            radius: Corner radius in pixels.

        """
        self._canvas.save()
        self._canvas.clipRRect(
            skia.RRect.MakeRectXY(
                skia.Rect.MakeLTRB(
                    rect.left, rect.top, rect.right, rect.bottom
                ),
                radius,
                radius,
            ),
            doAntiAlias=True,
        )
        try:
            yield
        finally:
            self._canvas.restore()

    @contextlib.contextmanager
    def viewport(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        logical: tuple[int, int] | None = None,
    ) -> "Iterator[Canvas]":
        """Draw into a sub-rectangle, yielding a canvas local to that region.

        Clips to ``(x, y, width, height)`` and translates the origin there, so the
        yielded canvas' draw calls land inside the rect. Pass ``logical`` to map a
        different logical coordinate space onto the rect (its content is scaled to
        fit) — e.g. drawing a natural-size composite into a smaller display box.
        The yielded canvas shares this one's backing and reports ``logical`` (or the
        rect) as its size; it is only valid inside the ``with`` block.

        Args:
            x: Sub-rect left in the current coordinate space.
            y: Sub-rect top.
            width: Sub-rect width.
            height: Sub-rect height.
            logical: The ``(width, height)`` the yielded canvas reports and draws
                in; scaled onto the rect. Defaults to the rect's own size.

        Yields:
            A `Canvas` drawing into the region (sharing this canvas' backing).

        """
        lw, lh = (
            logical if logical is not None else (round(width), round(height))
        )
        self._canvas.save()
        self._canvas.translate(float(x), float(y))
        self._canvas.clipRect(
            skia.Rect.MakeWH(float(width), float(height)), doAntiAlias=True
        )
        if (lw, lh) != (width, height):
            self._canvas.scale(width / lw, height / lh)
        try:
            yield Canvas(
                self._canvas,
                (int(lw), int(lh)),
                self._fonts,
                antialias=self._antialias,
            )
        finally:
            self._canvas.restore()

    def finish_svg(self) -> bytes:
        """Finalize an SVG canvas (from `svg`) and return the SVG document bytes.

        Dropping the recording canvas flushes its buffered draw commands to the
        backing stream, which is then detached. The canvas is spent afterwards.

        Returns:
            The SVG document as UTF-8 bytes.

        Raises:
            ValueError: If this is a raster canvas rather than one from `svg`.

        """
        if self._svg_stream is None:
            raise ValueError(
                "finish_svg is only valid for an SVG canvas (see Canvas.svg)"
            )
        # The SVG backend buffers; releasing the canvas flushes it to the stream.
        self._canvas = None  # type: ignore[assignment]
        self._svg_stream.flush()
        return bytes(self._svg_stream.detachAsData())

    # -- primitives ---------------------------------------------------------

    def _fill_paint(self, color: Color) -> skia.Paint:
        """Build an anti-aliased fill paint.

        Args:
            color: Fill color.

        Returns:
            A configured fill ``skia.Paint``.

        """
        return skia.Paint(
            Color=_color4f(color),
            AntiAlias=self._antialias,
            Style=skia.Paint.kFill_Style,
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
            AntiAlias=self._antialias,
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

    def gradient_line(
        self, p1: XY, p2: XY, c1: Color, c2: Color, width: float = 2.0
    ) -> None:
        """Draw a straight line whose color fades from ``c1`` to ``c2``.

        A single Skia linear-gradient stroke — a true continuous gradient rather
        than a run of solid sub-segments.

        Args:
            p1: Start point (takes color ``c1``).
            p2: End point (takes color ``c2``).
            c1: Color at ``p1``.
            c2: Color at ``p2``.
            width: Line width in pixels.

        """
        paint = self._stroke_paint(c1, width)
        paint.setShader(
            skia.GradientShader.MakeLinear(
                points=[skia.Point(p1[0], p1[1]), skia.Point(p2[0], p2[1])],
                colors=[_skia_color(c1), _skia_color(c2)],
            )
        )
        self._canvas.drawLine(p1[0], p1[1], p2[0], p2[1], paint)

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
        self,
        text: str,
        size: float,
        *,
        weight: int = 400,
        italic: bool = False,
        mono: bool = False,
    ) -> TextMetrics:
        """Measure a run of text without drawing it.

        Args:
            text: The string to measure.
            size: Text size in pixels.
            weight: OpenType weight (100-900).
            italic: Whether to measure the italic variant.
            mono: Whether to measure the monospace family.

        Returns:
            The `TextMetrics` for the run.

        """
        font = self._fonts.font(size, weight=weight, italic=italic, mono=mono)
        width = font.measureText(text)
        metrics = font.getMetrics()
        # Size to the font's ink box (fTop/fBottom), not the typographic
        # ascent/descent (fAscent/fDescent): tall glyphs such as JetBrains Mono's
        # full-height brackets/braces reach the ink box, so a card sized to the
        # typographic metrics clips them top and bottom. fTop/fBottom are always
        # at least as tall, so this only ever adds room, never removes it.
        return TextMetrics(
            width=width, ascent=-metrics.fTop, descent=metrics.fBottom
        )

    def wrap_text(
        self,
        text: str,
        size: float,
        *,
        max_width: float,
        weight: int = 400,
        italic: bool = False,
        mono: bool = False,
    ) -> list[str]:
        """Greedily word-wrap ``text`` to fit ``max_width`` pixels.

        Splits on spaces and existing newlines; a single word wider than
        ``max_width`` is hard-broken so no returned line exceeds the width. Used
        to keep long labels and titles inside the canvas.

        Args:
            text: The string to wrap.
            size: Text size in pixels.
            max_width: Maximum line width in pixels.
            weight: OpenType weight (100-900).
            italic: Whether the text is italic (affects measured width).
            mono: Whether the text uses the monospace family.

        Returns:
            The wrapped lines, top to bottom (empty when ``text`` is empty).

        """
        if not text:
            return []
        limit = max(1.0, max_width)

        def width(run: str) -> float:
            return self.measure_text(
                run, size, weight=weight, italic=italic, mono=mono
            ).width

        lines: list[str] = []
        for paragraph in text.split("\n"):
            lines.extend(_wrap_words(paragraph.split(" "), width, limit))
        return lines

    def text(
        self,
        origin: XY,
        text: str,
        *,
        size: float,
        color: Color,
        weight: int = 400,
        italic: bool = False,
        mono: bool = False,
    ) -> None:
        """Draw text with its baseline-left at ``origin``.

        Args:
            origin: The ``(x, y)`` baseline-left position.
            text: The string to draw.
            size: Text size in pixels.
            color: Text color.
            weight: OpenType weight (100-900).
            italic: Whether to draw the italic variant.
            mono: Whether to draw with the monospace family.

        """
        font = self._fonts.font(size, weight=weight, italic=italic, mono=mono)
        paint = skia.Paint(Color=_color4f(color), AntiAlias=True)
        self._canvas.drawString(text, origin[0], origin[1], font, paint)

    # -- rich text (styled spans) --------------------------------------------

    def _measure_span(self, span: "Span", size: float) -> TextMetrics:
        """Measure a single styled `Span`."""
        return self.measure_text(
            span.text,
            size,
            weight=span.weight,
            italic=span.italic,
            mono=span.mono,
        )

    def measure_spans(
        self, spans: "Sequence[Span]", size: float
    ) -> TextMetrics:
        """Measure a styled line: summed widths, tallest ascent/descent.

        Args:
            spans: The styled runs making up one line.
            size: Text size in pixels.

        Returns:
            The combined `TextMetrics` for the line.

        """
        if not spans:
            return self.measure_text("", size)
        width = ascent = descent = 0.0
        for span in spans:
            metrics = self._measure_span(span, size)
            width += metrics.width
            ascent = max(ascent, metrics.ascent)
            descent = max(descent, metrics.descent)
        return TextMetrics(width=width, ascent=ascent, descent=descent)

    def draw_spans(
        self,
        origin: XY,
        spans: "Sequence[Span]",
        *,
        size: float,
        color: Color,
    ) -> None:
        """Draw styled runs left-to-right from a shared baseline.

        Args:
            origin: The ``(x, y)`` baseline-left position of the line.
            spans: The styled runs to draw in order.
            size: Text size in pixels.
            color: Text color (shared by every run).

        """
        x, y = origin
        for span in spans:
            if not span.text:
                continue
            self.text(
                (x, y),
                span.text,
                size=size,
                color=color,
                weight=span.weight,
                italic=span.italic,
                mono=span.mono,
            )
            x += self._measure_span(span, size).width

    def wrap_spans(
        self, spans: "Sequence[Span]", size: float, *, max_width: float
    ) -> "list[list[Span]]":
        """Word-wrap styled runs to ``max_width``, preserving each run's style.

        Splits on whitespace (newlines force a break); a word wider than
        ``max_width`` is hard-broken so no line exceeds the width.

        Args:
            spans: The styled runs making up the paragraph.
            size: Text size in pixels.
            max_width: Maximum line width in pixels.

        Returns:
            One list of `Span` per output line (at least one line).

        """
        limit = max(1.0, max_width)
        space_w = self.measure_text(" ", size).width
        state = _WrapState()
        for word, newline in _span_atoms(spans):
            if word is None:
                if newline:
                    state.newline()
                elif state.current:
                    state.gap = True
                continue
            for piece in self._break_span(word, size, limit):
                self._wrap_word(state, piece, size, limit, space_w)
        state.lines.append(state.current)
        return state.lines

    def _break_span(
        self, span: Span, size: float, limit: float
    ) -> "list[Span]":
        """Hard-break an over-wide span into pieces that each fit ``limit``."""
        if not span.text or self._measure_span(span, size).width <= limit:
            return [span]

        def width_of(run: str) -> float:
            return self.measure_text(
                run,
                size,
                weight=span.weight,
                italic=span.italic,
                mono=span.mono,
            ).width

        return [
            Span(piece, span.weight, span.italic, span.mono)
            for piece in _break_word(span.text, width_of, limit)
        ]

    def _wrap_word(
        self,
        state: "_WrapState",
        word: Span,
        size: float,
        limit: float,
        space_w: float,
    ) -> None:
        """Place one word into ``state``, breaking to a new line if it overflows."""
        w = self._measure_span(word, size).width
        lead = space_w if state.gap else 0.0
        if state.current and state.width + lead + w > limit:
            state.newline()
        if state.gap and state.current:
            state.current.append(
                Span(" ", word.weight, word.italic, word.mono)
            )
            state.width += space_w
        state.current.append(word)
        state.width += w
        state.gap = False

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
        # Paint only the mask's bounding box: an instance mask covers a small
        # patch, so allocating and compositing a full-canvas RGBA buffer per mask
        # is almost all wasted work. Crop to the set pixels and blit that patch.
        rows = np.any(m, axis=1)
        if not rows.any():
            return
        cols = np.any(m, axis=0)
        y0 = int(np.argmax(rows))
        y1 = int(len(rows) - np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols))
        x1 = int(len(cols) - np.argmax(cols[::-1]))
        patch = m[y0:y1, x0:x1]
        rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        rgba[patch] = (
            color.r,
            color.g,
            color.b,
            round(max(0.0, min(1.0, alpha)) * 255),
        )
        image = skia.Image.fromarray(rgba, colorType=_RGBA)
        self._canvas.drawImage(image, float(x0), float(y0))


def gaussian_blur(rgba: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur an RGBA image by ``sigma`` pixels, via Skia.

    Used for the frosted-glass backdrop behind translucent UI cards. Edges clamp
    (no dark halo pulled in from beyond the image), and the same Skia raster path
    as the rest of `Canvas` is used, so no extra dependency is needed.

    Args:
        rgba: An ``(H, W, 4)`` ``uint8`` array in RGBA order.
        sigma: Blur radius (standard deviation) in pixels; ``<= 0`` returns a
            copy unchanged.

    Returns:
        A new ``(H, W, 4)`` ``uint8`` RGBA array, blurred.

    """
    if sigma <= 0.0:
        return rgba.copy()

    height, width = rgba.shape[:2]
    info = skia.ImageInfo.Make(int(width), int(height), _RGBA, _UNPREMUL)
    surface = skia.Surface.MakeRaster(info)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color4f(0, 0, 0, 0))
    image = skia.Image.fromarray(np.ascontiguousarray(rgba), colorType=_RGBA)
    paint = skia.Paint(
        ImageFilter=skia.ImageFilters.Blur(
            float(sigma), float(sigma), skia.TileMode.kClamp
        )
    )
    canvas.drawImage(image, 0.0, 0.0, skia.SamplingOptions(), paint)
    return surface.makeImageSnapshot().toarray(colorType=_RGBA)
