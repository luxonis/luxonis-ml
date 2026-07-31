"""Dense array labels as drawable annotations, one class per kind of array.

An LDF array label is a raw tensor with no declared meaning, so nothing about it
says how it should be drawn. The same ``(3, H, W)`` block is equally plausibly
surface normals, an RGB picture, or three-class scores, and the three want
completely different pictures. Rather than guess once and hide the guess, each
reading is its own annotation, and
`luxonis_ml.vizlab.adapters.arrays.resolve_array_kind` decides which one a given
label gets.

They all share `ArrayField`, which owns the part that does not vary: where the
numbers come from (a ``.npy`` path, exactly as LDF stores it, or an in-memory
array, which is all the loader ever hands over), how they are unwrapped, and the
blit that puts the result on the canvas. A subclass supplies only ``_rgba`` —
the function turning values into pixels — which is the entire difference between
a depth map and an optical flow field.

.. image:: TODO-HOST/array_kinds.png
   :alt: The same scene drawn as a scalar field, a signed error map, optical
         flow, surface normals, a plain image, class scores, the certainty
         behind them, and the two combined.

Note:
    `ArrayField` widens `ArrayAnnotation.path` to optional and adds ``values``.
    That is deliberate but temporary: `LuxonisLoader` delivers arrays already
    loaded into memory, with no path to point at, so an annotation that insisted
    on a path could not be built at all on the path that matters most. When LDF's
    own `ArrayAnnotation` learns to hold an array, ``values`` moves up to it and
    these classes keep working unchanged.

"""

from typing import ClassVar

import numpy as np
from pydantic import (
    ConfigDict,
    FilePath,
    field_serializer,
    field_validator,
    model_validator,
)

from luxonis_ml.ldf import ArrayAnnotation
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.gradient import (
    DEFAULT_DIVERGING_GRADIENT,
    DEFAULT_GRADIENT,
    Gradient,
)
from luxonis_ml.vizlab.style import Style

from .base import Annotation, RenderContext
from .heatmap import (
    _resize_field,
    field_range,
    field_rgba,
    field_valid,
    normalize_field,
)
from .mask import SemanticMask, resize_mask

_WHITE = Color(255, 255, 255)

#: How far a per-pixel channel sum may drift from 1 and still read as an
#: already-normalized distribution rather than raw logits.
_DISTRIBUTION_TOLERANCE = 1e-3

#: Peak opacity for a field drawn over a photo, so the picture underneath stays
#: readable. A field in its own tile has nothing to show through and is opaque.
OVERLAY_ALPHA = 0.6


def _hsv_to_rgb(
    hue: np.ndarray, saturation: np.ndarray, value: np.ndarray
) -> np.ndarray:
    """Convert HSV planes in ``[0, 1]`` to an ``(H, W, 3)`` ``uint8`` image.

    ``colorsys`` is scalar-only and a flow field is a million pixels, so this is
    the standard sextant formulation done in one vectorized pass.
    """
    h6 = (hue % 1.0) * 6.0
    sector = np.floor(h6).astype(np.intp) % 6
    frac = h6 - np.floor(h6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * frac)
    t = value * (1.0 - saturation * (1.0 - frac))
    reds = np.choose(sector, [value, q, p, p, t, value])
    greens = np.choose(sector, [t, value, value, q, p, p])
    blues = np.choose(sector, [p, p, t, value, value, q])
    stacked = np.dstack([reds, greens, blues])
    return np.round(np.clip(stacked, 0.0, 1.0) * 255).astype(np.uint8)


def _opaque(rgb: np.ndarray, alpha: float, valid: np.ndarray) -> np.ndarray:
    """Attach a flat alpha channel to an RGB image, holing out invalid pixels."""
    peak = round(max(0.0, min(1.0, alpha)) * 255)
    band = np.where(valid, peak, 0).astype(np.uint8)
    return np.dstack([rgb, band])


class ArrayField(ArrayAnnotation, Annotation):
    """Base for every way of drawing a dense array label.

    Holds the values — from a ``.npy`` path or straight from memory — and blits
    whatever ``_rgba`` makes of them. Subclasses differ only in that one method.

    Attributes:
        path: The ``.npy`` file, as LDF stores it. Optional here, unlike on
            `ArrayAnnotation`, because loader output arrives already in memory.
        values: The array itself. Give exactly one of this or ``path``.
        alpha: Peak opacity in ``[0, 1]``.

    See `Annotation` for the shared fields (``color``/``palette`` do not apply —
    the colors come from the values).

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: FilePath | None = None
    values: np.ndarray | None = None
    alpha: float = 1.0

    #: Fields cover the whole image, beneath every other spatial annotation, so
    #: boxes and keypoints are never buried under one.
    BACKGROUND: ClassVar[bool] = True
    LAYER: ClassVar[str] = "field"

    @field_validator("path")
    @classmethod
    def _validate_path(cls, path: "FilePath | None") -> "FilePath | None":
        """Accept a missing path; otherwise apply LDF's own file checks."""
        if path is None:
            return None
        return ArrayAnnotation._validate_path(path)

    @field_serializer("path", when_used="json")
    def _serialize_path(self, value: "FilePath | None") -> str | None:
        return None if value is None else str(value)

    @model_validator(mode="after")
    def _require_one_source(self) -> "ArrayField":
        if (self.path is None) == (self.values is None):
            raise ValueError(
                "An array field needs exactly one of 'path' or 'values'."
            )
        return self

    def to_numpy(self) -> np.ndarray:
        """Return the array, preferring the in-memory one over a reload."""
        if self.values is not None:
            return np.asarray(self.values)
        return super().to_numpy()

    def field(self) -> "np.ndarray | None":
        """Return the payload this annotation draws, or ``None`` if unusable.

        Unwrapping happens here rather than at construction so an annotation
        built straight from loader output — still carrying the
        ``(N, n_classes, ...)`` prefix — draws the same as one built from a bare
        ``.npy``.
        """
        from luxonis_ml.vizlab.adapters.arrays import array_payload

        payload = array_payload(self.to_numpy())
        return None if payload is None else payload.data

    def _rgba(
        self, data: np.ndarray, ctx: RenderContext
    ) -> "np.ndarray | None":
        """Turn the payload into an RGBA image, at whatever resolution suits.

        Args:
            data: The unwrapped payload, ``(H, W)`` or ``(C, H, W)``.
            ctx: The current render context.

        Returns:
            An ``(h, w, 4)`` ``uint8`` image, or ``None`` to draw nothing.

        """
        raise NotImplementedError

    def resolve_color(self, ctx: RenderContext) -> Color:
        """Array fields color per value, so no single color applies."""
        return _WHITE

    def extent(self) -> Rect | None:
        """Array fields cover the whole image and have no local extent."""
        return None

    def draw_fill(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Blit the field over the image (first, raster pass)."""
        data = self.field()
        if data is None:
            return
        rgba = self._rgba(data, ctx)
        if rgba is None or rgba.size == 0:
            return
        canvas = ctx.canvas
        if rgba.shape[:2] == (canvas.height, canvas.width):
            canvas.blit(rgba, 0, 0)
        else:
            canvas.blit_scaled(rgba, 0, 0, canvas.width, canvas.height)

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Array fields have no sharp vector layer; everything is the fill."""


class ScalarField(ArrayField):
    """One measurement per pixel, read through a gradient.

    The depth/disparity/anomaly view, and — with ``center`` set — the signed
    error view. Magnitude deliberately does not drive opacity the way `Heatmap`
    defaults to: a disparity of 0.56 px is a real measurement, not a weak signal,
    and fading it out would misreport the data. Transparency is ``ignore_value``'s
    job alone.

    Attributes:
        gradient: Colormap; ``None`` inherits the render options'.
        vmin: Pin the low end of the value range.
        vmax: Pin the high end.
        center: Value the gradient's midpoint sits on. See `Heatmap.center`.
        ignore_value: A "no data" sentinel, drawn transparent.
        weight_by_value: Whether opacity scales with the value.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import Image, ScalarField
        >>> field = ScalarField(values=np.linspace(0, 9, 12).reshape(3, 4))
        >>> Image(np.zeros((6, 8, 3), np.uint8)).add(field).render().shape
        (6, 8, 4)

    """

    gradient: Gradient | str | None = None
    vmin: float | None = None
    vmax: float | None = None
    center: float | None = None
    ignore_value: float | None = None
    weight_by_value: bool = False

    def value_range(self) -> "tuple[float, float]":
        """Report the values the gradient's two ends stand for."""
        data = self.field()
        return field_range(
            np.zeros(0) if data is None else data,
            vmin=self.vmin,
            vmax=self.vmax,
            center=self.center,
            ignore_value=self.ignore_value,
        )

    def resolve_gradient_name(self, ctx: RenderContext) -> "Gradient | str":
        """Pick the colormap, defaulting a centered field to a diverging one.

        A signed field read through a sequential colormap hides its sign: the
        midpoint lands on whatever color happens to sit halfway. Centering only
        pays off against a gradient with a neutral middle.
        """
        if self.gradient is not None:
            return self.gradient
        if self.center is not None:
            return DEFAULT_DIVERGING_GRADIENT
        return ctx.gradient or DEFAULT_GRADIENT

    def _rgba(self, data: np.ndarray, ctx: RenderContext) -> np.ndarray:
        if data.ndim > 2:
            data = np.nanmax(data, axis=0)
        canvas = ctx.canvas
        return field_rgba(
            normalize_field(
                data,
                vmin=self.vmin,
                vmax=self.vmax,
                center=self.center,
                ignore_value=self.ignore_value,
            ),
            field_valid(data, self.ignore_value),
            gradient=self.resolve_gradient_name(ctx),
            alpha=self.alpha,
            weight_by_value=self.weight_by_value,
            width=canvas.width,
            height=canvas.height,
        )


class FlowField(ArrayField):
    """A two-channel displacement field, drawn as a direction color wheel.

    The convention every optical-flow paper since Middlebury uses: hue is the
    direction the pixel moved, saturation is how far. Magnitude alone would be a
    scalar field and would throw away the direction, which is most of the signal.

    Attributes:
        max_magnitude: Displacement mapped to full saturation. ``None`` uses the
            field's own maximum, which makes two clips look alike even when one
            moved twice as fast — pin it to compare them.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import FlowField, Image
        >>> flow = np.zeros((2, 4, 5), np.float32)
        >>> flow[0] = 1.0  # everything moving right
        >>> Image(np.zeros((8, 10, 3), np.uint8)).add(
        ...     FlowField(values=flow)
        ... ).render().shape
        (8, 10, 4)

    """

    max_magnitude: float | None = None

    def peak_magnitude(self) -> float:
        """Return the displacement mapped to full saturation."""
        if self.max_magnitude is not None:
            return float(self.max_magnitude)
        data = self.field()
        if data is None or data.ndim < 3 or data.shape[0] < 2:
            return 0.0
        magnitude = np.hypot(data[0], data[1])
        finite = magnitude[np.isfinite(magnitude)]
        return float(finite.max()) if finite.size else 0.0

    def _rgba(
        self, data: np.ndarray, ctx: RenderContext
    ) -> "np.ndarray | None":
        if data.ndim < 3 or data.shape[0] < 2:
            return None
        dx, dy = np.nan_to_num(data[0]), np.nan_to_num(data[1])
        magnitude = np.hypot(dx, dy)
        peak = self.peak_magnitude()
        saturation = (
            np.clip(magnitude / peak, 0.0, 1.0)
            if peak > 0
            else np.zeros_like(magnitude)
        )
        # Screen coordinates run downwards, so negate dy to make the hue follow
        # the direction a viewer sees the pixel travel.
        hue = (np.arctan2(-dy, dx) / (2.0 * np.pi)) % 1.0
        rgb = _hsv_to_rgb(hue, saturation, np.ones_like(saturation))
        return _opaque(rgb, self.alpha, np.isfinite(data[:2]).all(axis=0))


class NormalMap(ArrayField):
    """Three channels of unit vector per pixel, drawn as the usual pastel RGB.

    Surface normals live in ``[-1, 1]``; the convention is to shift them into
    ``[0, 1]`` and read xyz as RGB, which is why normal maps all look alike.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import Image, NormalMap
        >>> normals = np.zeros((3, 4, 5), np.float32)
        >>> normals[2] = 1.0  # all facing the camera
        >>> Image(np.zeros((8, 10, 3), np.uint8)).add(
        ...     NormalMap(values=normals)
        ... ).render().shape
        (8, 10, 4)

    """

    def _rgba(
        self, data: np.ndarray, ctx: RenderContext
    ) -> "np.ndarray | None":
        if data.ndim < 3 or data.shape[0] < 3:
            return None
        vectors = np.clip(np.nan_to_num(data[:3]), -1.0, 1.0)
        rgb = np.round((np.moveaxis(vectors, 0, -1) + 1.0) * 0.5 * 255).astype(
            np.uint8
        )
        return _opaque(rgb, self.alpha, np.isfinite(data[:3]).all(axis=0))


class ArrayImage(ArrayField):
    """An array that already *is* a picture, drawn as-is.

    For a restoration or generative model's output, a reprojected view, or any
    other three-channel array whose values are colors rather than measurements.
    Accepts ``[0, 1]`` floats or ``[0, 255]`` bytes, telling them apart by range.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import ArrayImage, Image
        >>> pixels = np.full((3, 4, 5), 0.5, np.float32)
        >>> Image(np.zeros((8, 10, 3), np.uint8)).add(
        ...     ArrayImage(values=pixels)
        ... ).render().shape
        (8, 10, 4)

    """

    def _rgba(
        self, data: np.ndarray, ctx: RenderContext
    ) -> "np.ndarray | None":
        if data.ndim < 3 or data.shape[0] < 3:
            return None
        pixels = np.nan_to_num(np.moveaxis(data[:3], 0, -1))
        finite = pixels[np.isfinite(pixels)]
        # A field whose values all sit inside [0, 1] is unit-scaled; anything
        # reaching past it was already stored as bytes.
        if finite.size and float(finite.max()) <= 1.0:
            pixels = pixels * 255.0
        rgb = np.clip(np.round(pixels), 0, 255).astype(np.uint8)
        return _opaque(rgb, self.alpha, np.isfinite(data[:3]).all(axis=0))


class SegmentationScores(ArrayField):
    """One score plane per class, drawn as the segmentation it argmaxes to.

    The shape every semantic-segmentation head emits. The picture worth drawing
    is not the scores themselves but the label each pixel won, so this resolves
    the stack to a class-index map and hands it to `SemanticMask`, which already
    knows how to fill and outline one and colors it by class name.

    Attributes:
        names: Class name per channel, in channel order. Supplying them is what
            makes the mask's colors match the rest of the scene, since the
            palette keys on names.
        ignore_index: Channel indices never drawn; ``0`` keeps the usual
            background class invisible. See `SemanticMask.ignore_index`.
        color_map: Explicit color per channel index, overriding the palette.
        fill_alpha: Fill opacity override.
        contour: Whether to stroke class outlines.
        weight_by_confidence: When ``True`` each pixel's opacity scales with
            how sure the winning class was, so the mask keeps its class colors
            but fades wherever the model was guessing. Reads both halves of
            the stack at once — `confidence` shows the certainty alone, and a
            flat mask hides it entirely.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import SegmentationScores
        >>> scores = np.zeros((3, 2, 2), np.float32)
        >>> scores[1, 0] = 5.0  # class 1 wins the top row
        >>> scores[2, 1] = 5.0  # class 2 wins the bottom row
        >>> SegmentationScores(
        ...     values=scores, names=["bg", "road", "sky"]
        ... ).labels().tolist()
        [[1, 1], [2, 2]]

    """

    names: "dict[int, str] | list[str] | None" = None
    ignore_index: "int | list[int]" = 0
    color_map: "dict[int, ColorLike] | None" = None
    fill_alpha: float | None = None
    contour: bool = True
    weight_by_confidence: bool = False

    def labels(self) -> np.ndarray:
        """Resolve the stack to the winning class index per pixel."""
        data = self.field()
        if data is None:
            return np.zeros((0, 0), np.int32)
        if data.ndim == 2:
            # A single plane is already an index map; combine_to_numpy stores
            # everything as float, so round rather than truncate.
            return np.rint(np.nan_to_num(data)).astype(np.int32)
        return np.argmax(np.nan_to_num(data), axis=0).astype(np.int32)

    def probabilities(self, data: np.ndarray) -> np.ndarray:
        """Read the stack as a per-pixel distribution over the classes.

        Heads emit either probabilities or raw logits, and the two cannot be
        told apart by shape. A stack whose channels are non-negative and already
        sum to one per pixel is taken as-is; anything else is softmaxed, since a
        bare logit of ``9.0`` says nothing on its own — only its size *relative
        to the other classes* does.
        """
        scores = np.nan_to_num(data)
        totals = scores.sum(axis=0)
        if np.all(scores >= 0.0) and np.allclose(
            totals, 1.0, atol=_DISTRIBUTION_TOLERANCE
        ):
            return scores
        # Shift by the per-pixel max before exponentiating: mathematically a
        # no-op, but it keeps exp() from overflowing on large logits.
        exponent = np.exp(scores - scores.max(axis=0, keepdims=True))
        return exponent / exponent.sum(axis=0, keepdims=True)

    def confidence(self, **fields: object) -> ScalarField:
        """Build the field of how sure the model was, pixel by pixel.

        The probability the winning class took. Where it is low the model was
        guessing between classes, which a flat segmentation map cannot show at
        all — every pixel there looks equally decided.

        The range is pinned to ``[1 / n_classes, 1]``, the band such a value can
        actually occupy: a uniform guess over ``C`` classes still awards the
        winner ``1 / C``. Auto-scaling instead would stretch a uniformly
        confident frame to look full of doubt. Pass ``vmin``/``vmax`` to override.

        Args:
            **fields: Any `ScalarField` field, such as ``gradient`` or ``vmin``.

        Returns:
            A `ScalarField` over the winning class's probability.

        """
        data = self.field()
        if data is None:
            peak = np.zeros((0, 0), np.float32)
        elif data.ndim == 2:
            # Already one value per pixel: a certainty map, not a stack.
            peak = data
        else:
            peak = self.probabilities(data).max(axis=0)
            fields.setdefault("vmin", 1.0 / data.shape[0])
            fields.setdefault("vmax", 1.0)
        return ScalarField(
            values=peak.astype(np.float32),
            alpha=self.alpha,
            **fields,  # pyright: ignore[reportArgumentType]
        )

    def _mask(self) -> SemanticMask:
        """Build the `SemanticMask` that does the actual drawing."""
        return SemanticMask(
            labels=self.labels(),
            names=self.names,
            ignore_index=self.ignore_index,
            color_map=self.color_map,
            fill_alpha=self.fill_alpha
            if self.fill_alpha is not None
            else self.alpha,
            style=self.style,
            style_overrides=self.style_overrides,
            palette=self.palette,
        )

    def _certainty_weights(self, data: np.ndarray) -> np.ndarray:
        """Rescale the winning probability to ``[0, 1]`` opacity weights.

        Stretched over ``[1 / n_classes, 1]`` rather than ``[0, 1]``: a pixel the
        model split evenly between ``C`` classes still scores ``1 / C``, and that
        is the one that should vanish entirely. Mapping from zero instead would
        leave even a coin-flip looking half-solid.
        """
        classes = data.shape[0]
        peak = self.probabilities(data).max(axis=0)
        floor = 1.0 / classes
        if floor >= 1.0:
            return np.ones_like(peak)
        return np.clip((peak - floor) / (1.0 - floor), 0.0, 1.0)

    def _weighted_fill(self, ctx: RenderContext, style: Style) -> None:
        """Paint class colors at an opacity set by how sure each pixel was.

        The two readings at once: hue says which class won, solidity says how
        much it won by. Boundaries and confusions dissolve into the image instead
        of being stated as flatly as the parts the model was sure of.
        """
        data = self.field()
        if data is None or data.ndim != 3:
            return
        canvas = ctx.canvas
        mask = self._mask()
        palette = mask.resolved_palette(ctx)
        alpha = (
            style.mask_alpha if self.fill_alpha is None else self.fill_alpha
        )
        labels = resize_mask(self.labels(), canvas.width, canvas.height)
        weights = _resize_field(
            self._certainty_weights(data), canvas.width, canvas.height
        )
        rgb = np.zeros((canvas.height, canvas.width, 3), np.uint8)
        drawn = np.zeros((canvas.height, canvas.width), bool)
        ignored = mask._ignored()
        for class_id in np.unique(labels):
            index = int(class_id)
            if index in ignored:
                continue
            selected = labels == class_id
            own = mask._color(index, palette)
            rgb[selected] = (own.r, own.g, own.b)
            drawn |= selected
        band = np.clip(weights, 0.0, 1.0) * drawn * alpha * 255
        canvas.blit(np.dstack([rgb, band.astype(np.uint8)]), 0, 0)

    def draw_fill(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Fill each class region, flat or weighted by certainty."""
        if self.weight_by_confidence:
            self._weighted_fill(ctx, style)
        else:
            self._mask().draw_fill(ctx, style, color)

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Stroke the class outlines, delegating to `SemanticMask`."""
        if self.contour:
            self._mask().draw(ctx, style, color)
