"""A heatmap overlay: a dense scalar field colored through a gradient.

`Heatmap` is a rendering-only construct (it has no single LDF annotation
counterpart, like `SemanticMask`): it takes an ``(H, W)`` array of magnitudes —
an attention/saliency map, a density estimate, a per-pixel score — normalizes it,
colors it through a `Gradient` (colormap), and blends the result over the image.

Unlike the spatial annotations it does not use the class palette; its color comes
entirely from its gradient, so a set of gradient *themes* (plus a custom-gradient
escape hatch) controls its look independently of the other labels.
"""

from typing import ClassVar

import numpy as np

from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.gradient import (
    DEFAULT_GRADIENT,
    Gradient,
    resolve_gradient,
)
from luxonis_ml.vizlab.style import Style

from .base import Annotation, RenderContext

_WHITE = Color(255, 255, 255)


def _resize_field(field: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinearly resample a 2-D float field to ``(height, width)``.

    Heatmaps are often lower-resolution than the image they annotate (e.g. a
    small attention map). Bilinear resampling keeps the overlay smooth rather
    than blocky, and needs no OpenCV.

    Args:
        field: An ``(H, W)`` array.
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        The field resampled to ``(height, width)`` as ``float64`` (the original,
        as ``float64``, when it already matches).

    """
    src_h, src_w = field.shape[:2]
    f = np.asarray(field, dtype=np.float64)
    if (src_h, src_w) == (height, width):
        return f
    ys = np.linspace(0.0, src_h - 1, height)
    xs = np.linspace(0.0, src_w - 1, width)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = f[y0][:, x0] * (1.0 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1.0 - wx) + f[y1][:, x1] * wx
    return top * (1.0 - wy) + bot * wy


def field_valid(
    values: np.ndarray, ignore_value: float | None = None
) -> np.ndarray:
    """Mask the pixels of a field that carry data.

    Args:
        values: The raw field.
        ignore_value: A "no data" sentinel, compared exactly.

    Returns:
        A boolean array, ``True`` where the value is finite and not the sentinel.

    """
    v = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(v)
    if ignore_value is not None:
        valid &= v != ignore_value
    return valid


def field_range(
    values: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    normalize: bool = True,
    center: float | None = None,
    ignore_value: float | None = None,
) -> "tuple[float, float]":
    """Resolve the value range a field's gradient spans.

    ``center`` applies only when *neither* bound is pinned: it makes the
    automatic range symmetric around that value, so the gradient's midpoint lands
    exactly on it. Pinning both bounds is a more specific instruction and wins.

    Args:
        values: The raw field.
        vmin: Pin the low end.
        vmax: Pin the high end.
        normalize: When ``False`` and neither bound is pinned, the field is
            treated as already normalized and the range is ``(0, 1)``.
        center: Value to center an automatic range on.
        ignore_value: A "no data" sentinel, left out of an automatic range.

    Returns:
        The low and high ends, or ``(0.0, 0.0)`` when nothing is valid.

    """
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    if vmin is None and vmax is None and not normalize:
        return 0.0, 1.0
    v = np.asarray(values, dtype=np.float64)
    valid = field_valid(v, ignore_value)
    if not valid.any():
        return 0.0, 0.0
    data = v[valid]
    low = float(vmin) if vmin is not None else float(data.min())
    high = float(vmax) if vmax is not None else float(data.max())
    if center is not None and vmin is None and vmax is None:
        radius = max(high - center, center - low, 0.0)
        return center - radius, center + radius
    return low, high


def normalize_field(
    values: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    normalize: bool = True,
    center: float | None = None,
    ignore_value: float | None = None,
) -> np.ndarray:
    """Scale a field into ``[0, 1]`` over its valid pixels.

    Invalid pixels land at ``0`` rather than propagating: without this a single
    ``NaN`` makes the whole normalized field ``NaN``, which the gradient then
    casts to arbitrary bytes. Callers make them transparent, so they are never
    read as a real low value.
    """
    v = np.asarray(values, dtype=np.float64)
    low, high = field_range(
        v,
        vmin=vmin,
        vmax=vmax,
        normalize=normalize,
        center=center,
        ignore_value=ignore_value,
    )
    if high <= low:
        return np.zeros_like(v)
    out = np.zeros_like(v)
    np.divide(v - low, high - low, out=out, where=field_valid(v, ignore_value))
    return np.clip(out, 0.0, 1.0)


def field_rgba(
    field: np.ndarray,
    valid: np.ndarray,
    *,
    gradient: Gradient | str,
    alpha: float,
    weight_by_value: bool,
    width: int,
    height: int,
) -> np.ndarray:
    """Resample a normalized field to the canvas and color it.

    Args:
        field: The source-resolution field, already scaled to ``[0, 1]``.
        valid: The source-resolution mask of pixels carrying data.
        gradient: Colormap to run the values through.
        alpha: Peak opacity in ``[0, 1]``.
        weight_by_value: Whether opacity scales with the value.
        width: Target canvas width.
        height: Target canvas height.

    Returns:
        An ``(height, width, 4)`` ``uint8`` RGBA overlay.

    """
    scaled = np.clip(_resize_field(field, width, height), 0.0, 1.0)
    rgb = resolve_gradient(gradient).colorize(scaled)
    peak = round(max(0.0, min(1.0, alpha)) * 255)
    weights = scaled if weight_by_value else np.ones_like(scaled)
    if not valid.all():
        # Resampling the mask as a float and multiplying — rather than
        # thresholding it — anti-aliases the nodata boundary for free.
        # Resampling the *values* still bleeds across that boundary, but invalid
        # pixels normalize to 0, so the bleed is bounded to the low end of the
        # range and is already faded out by this alpha.
        weights = weights * np.clip(
            _resize_field(valid.astype(np.float64), width, height), 0.0, 1.0
        )
    return np.dstack([rgb, (weights * peak).astype(np.uint8)])


class Heatmap(Annotation):
    """A translucent gradient-colored overlay of a dense scalar field.

    The field is normalized to ``[0, 1]`` (see ``normalize``/``vmin``/``vmax``),
    colored through ``gradient``, and blended over the image. By default low
    values fade to transparent so cold regions leave the underlying pixels
    visible; set ``weight_by_value=False`` for a flat opacity everywhere.

    The overlay is painted at the source resolution in the first (raster) render
    pass and scaled with the image, so it stays smooth when the image is resized
    for display.

    .. image:: TODO-HOST/heatmaps.png
       :alt: One density field colored through several gradient themes.

    Attributes:
        values: The ``(H, W)`` scalar field. May be any resolution; it is
            resampled to the image. ``None`` draws nothing.
        gradient: A `Gradient`, or the name of a preset
            (see `luxonis_ml.vizlab.gradient.GRADIENTS`). ``None`` (the default)
            inherits the render options' gradient, then the library default.
        alpha: Peak overlay opacity in ``[0, 1]`` (the opacity at the maximum
            value; lower values are more transparent when ``weight_by_value``).
        weight_by_value: When ``True`` (default) each pixel's opacity scales with
            its normalized value, so the heatmap fades out over cold regions.
            When ``False`` the whole field is drawn at ``alpha``.
        normalize: When ``True`` (default) the field is min-max normalized to
            ``[0, 1]``. Ignored when ``vmin``/``vmax`` are given.
        vmin: Lower bound of the value range; values at or below it map to ``0``.
            ``None`` uses the field minimum (or ``0`` when ``normalize`` is off).
        vmax: Upper bound of the value range; values at or above it map to ``1``.
            ``None`` uses the field maximum (or ``1`` when ``normalize`` is off).
        center: Value the gradient's midpoint sits on, making the automatic range
            symmetric around it. Set it to ``0`` with a diverging gradient (see
            `luxonis_ml.vizlab.gradient.GRADIENTS`) to read a signed field — an
            error or residual map — where the sign matters as much as the
            magnitude; otherwise min-max stretching puts zero at an arbitrary
            color. Ignored when both ``vmin`` and ``vmax`` are pinned, since that
            is the more specific instruction.
        ignore_value: A "no data" sentinel — pixels exactly equal to it are left
            out of an automatic range and drawn fully transparent. Use it for
            fields that mark gaps with a value rather than with ``NaN``, such as
            a stereo disparity map whose unmatched pixels are ``0``. Compared
            exactly, since a sentinel is a stored constant rather than a
            measurement. Non-finite values are always treated this way, with or
            without a sentinel.

    See `Annotation` for the shared fields
    (``label``/``color``/``palette`` do not apply — color comes from the gradient).

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import Heatmap, Image
        >>> field = np.linspace(0, 1, 6 * 8).reshape(6, 8)
        >>> heat = Heatmap(values=field, gradient="viridis")
        >>> Image(np.zeros((6, 8, 3), np.uint8)).add(heat).render().shape
        (6, 8, 4)

    """

    LAYER: ClassVar[str] = "field"

    values: np.ndarray | None = None
    gradient: Gradient | str | None = None
    alpha: float = 0.8
    weight_by_value: bool = True
    normalize: bool = True
    vmin: float | None = None
    vmax: float | None = None
    center: float | None = None
    ignore_value: float | None = None

    def _valid(self, values: np.ndarray) -> np.ndarray:
        """Mask the pixels that carry data.

        Args:
            values: The raw field.

        Returns:
            A boolean array, ``True`` where the value is finite and not the
            ``ignore_value`` sentinel.

        """
        return field_valid(values, self.ignore_value)

    def _range(self, values: np.ndarray) -> "tuple[float, float]":
        """Resolve the mapped range of ``values`` over its valid pixels."""
        return field_range(
            values,
            vmin=self.vmin,
            vmax=self.vmax,
            normalize=self.normalize,
            center=self.center,
            ignore_value=self.ignore_value,
        )

    def value_range(self) -> "tuple[float, float]":
        """Report the values that map to the gradient's two ends.

        Resolves ``vmin``/``vmax``/``normalize`` against the *valid* pixels, so
        a nodata sentinel or a stray ``NaN`` never widens the range. `ColorBar`
        labels its ends with this.

        Returns:
            The low and high ends of the mapped range, or ``(0.0, 0.0)`` when
            the field is empty or entirely invalid.

        Examples:
            >>> import numpy as np
            >>> from luxonis_ml.vizlab import Heatmap
            >>> field = np.array([[0.0, 5.0, 10.0]])
            >>> Heatmap(values=field).value_range()
            (0.0, 10.0)
            >>> Heatmap(values=field, ignore_value=0.0).value_range()
            (5.0, 10.0)

        """
        values = np.zeros(0) if self.values is None else self.values
        return self._range(np.asarray(values))

    def _normalized(self, values: np.ndarray) -> np.ndarray:
        """Scale the field into ``[0, 1]`` per ``vmin``/``vmax``/``normalize``.

        Invalid pixels land at ``0`` rather than propagating: without this a
        single ``NaN`` makes the whole normalized field ``NaN``, which the
        gradient then casts to arbitrary bytes. `draw_fill` makes them
        transparent, so they are never read as a real low value.
        """
        return normalize_field(
            values,
            vmin=self.vmin,
            vmax=self.vmax,
            normalize=self.normalize,
            center=self.center,
            ignore_value=self.ignore_value,
        )

    def resolve_color(self, ctx: RenderContext) -> Color:
        """Heatmaps color per value through the gradient, so no single color applies.

        Args:
            ctx: The current render context (unused).

        Returns:
            An unused placeholder color; the real colors come from the gradient.

        """
        return _WHITE

    def extent(self) -> Rect | None:
        """Heatmaps cover the whole image and have no local extent.

        Returns:
            Always ``None``.

        """
        return None

    def draw_fill(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Colorize the field and blend it over the image (first, raster pass).

        Args:
            ctx: The current render context (native resolution).
            style: The resolved style (unused).
            color: Unused (colors come from the gradient).

        """
        if self.values is None:
            return
        raw = np.asarray(self.values)
        if raw.size == 0:
            return
        canvas = ctx.canvas
        rgba = field_rgba(
            self._normalized(raw),
            self._valid(raw),
            # Own gradient wins; else inherit the render options' default; else
            # the library default. (Names and Gradient objects are all truthy.)
            gradient=self.gradient or ctx.gradient or DEFAULT_GRADIENT,
            alpha=self.alpha,
            weight_by_value=self.weight_by_value,
            width=canvas.width,
            height=canvas.height,
        )
        canvas.blit(rgba, 0, 0)

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Heatmaps have no sharp vector layer; everything is the raster fill.

        Args:
            ctx: The current render context (unused).
            style: The resolved style (unused).
            color: Unused.

        """
