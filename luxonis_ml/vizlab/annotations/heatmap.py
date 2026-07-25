"""A heatmap overlay: a dense scalar field colored through a gradient.

`Heatmap` is a rendering-only construct (it has no single LDF annotation
counterpart, like `SemanticMask`): it takes an ``(H, W)`` array of magnitudes —
an attention/saliency map, a density estimate, a per-pixel score — normalizes it,
colors it through a `Gradient` (colormap), and blends the result over the image.

Unlike the spatial annotations it does not use the class palette; its color comes
entirely from its gradient, so a set of gradient *themes* (plus a custom-gradient
escape hatch) controls its look independently of the other labels.
"""

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


class Heatmap(Annotation):
    """A translucent gradient-colored overlay of a dense scalar field.

    The field is normalized to ``[0, 1]`` (see ``normalize``/``vmin``/``vmax``),
    colored through ``gradient``, and blended over the image. By default low
    values fade to transparent so cold regions leave the underlying pixels
    visible; set ``weight_by_value=False`` for a flat opacity everywhere.

    The overlay is painted at the source resolution in the first (raster) render
    pass and scaled with the image, so it stays smooth when the image is resized
    for display.

    Attributes:
        values: The ``(H, W)`` scalar field. May be any resolution; it is
            resampled to the image. ``None`` draws nothing.
        gradient: A `Gradient`, or the name of a preset
            (see `luxonis_ml.vizlab.gradient.GRADIENTS`).
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

    values: np.ndarray | None = None
    gradient: Gradient | str = DEFAULT_GRADIENT
    alpha: float = 0.8
    weight_by_value: bool = True
    normalize: bool = True
    vmin: float | None = None
    vmax: float | None = None

    def _normalized(self, values: np.ndarray) -> np.ndarray:
        """Scale the field into ``[0, 1]`` per ``vmin``/``vmax``/``normalize``."""
        v = np.asarray(values, dtype=np.float64)
        if self.vmin is not None or self.vmax is not None:
            lo = self.vmin if self.vmin is not None else float(v.min())
            hi = self.vmax if self.vmax is not None else float(v.max())
        elif self.normalize:
            lo, hi = float(v.min()), float(v.max())
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            return np.zeros_like(v)
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0)

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
        canvas = ctx.canvas
        field = self._normalized(np.asarray(self.values))
        field = _resize_field(field, canvas.width, canvas.height)
        field = np.clip(field, 0.0, 1.0)
        rgb = resolve_gradient(self.gradient).colorize(field)
        peak = round(max(0.0, min(1.0, self.alpha)) * 255)
        if self.weight_by_value:
            alpha = (field * peak).astype(np.uint8)
        else:
            alpha = np.full(field.shape, peak, dtype=np.uint8)
        rgba = np.dstack([rgb, alpha])
        canvas.blit(rgba, 0, 0)

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Heatmaps have no sharp vector layer; everything is the raster fill.

        Args:
            ctx: The current render context (unused).
            style: The resolved style (unused).
            color: Unused.

        """
