"""The `Image` composition root.

An `Image` wraps a base raster and a list of annotations (the scene graph).
Annotations are collected with `Image.add`; nothing is drawn until
`Image.render` is called, which rasterizes the base plus every annotation
in one pass.
"""

import hashlib
from collections.abc import Hashable
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from . import io
from .annotations.base import Annotation, RenderContext
from .annotations.layout import LabelLayout
from .canvas import Canvas
from .geometry import Rect
from .hitmap import HitMap
from .options import RenderOptions, current_options
from .style import Theme
from .tooltip import Tooltip

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from .convert import RenderableLDF
    from .frame import Frame
    from .gradient import Gradient
    from .io import ImageSource
    from .panel import PanelData

#: Canvas short-side (px) at which styles render at their nominal size; larger
#: canvases scale labels/strokes up proportionally (clamped to the range below).
#: Kept a touch below a typical frame so type reads a bit larger relative to the
#: image on medium and large canvases.
_STYLE_REFERENCE_PX = 700.0
_STYLE_SCALE_RANGE = (1.0, 3.0)


def _qualname(value: object) -> str:
    """Fully-qualified ``module.QualName`` of ``value``'s type."""
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _freeze_ndarray(array: np.ndarray) -> Hashable:
    """Freeze an array by dtype, shape, and a content hash."""
    contiguous = np.ascontiguousarray(array)
    return (
        "ndarray",
        contiguous.dtype.str,
        contiguous.shape,
        hashlib.sha256(contiguous.tobytes()).digest(),
    )


def _freeze_fields(value: object, names: "list[str]") -> Hashable:
    """Freeze an object by ``(qualname, ((field, frozen), ...))``."""
    return (
        _qualname(value),
        tuple(
            (name, _freeze_render_state(getattr(value, name)))
            for name in names
        ),
    )


def _freeze_dict(value: dict) -> Hashable:
    """Freeze a dict into a repr-sorted tuple of frozen key/value pairs."""
    items = [
        (_freeze_render_state(key), _freeze_render_state(item))
        for key, item in value.items()
    ]
    return tuple(sorted(items, key=lambda item: repr(item[0])))


def _freeze_leaf(value: object) -> Hashable:
    """Freeze a set, enum, ``__dict__`` object, or scalar (the non-container tail)."""
    if isinstance(value, set):
        return tuple(
            sorted((_freeze_render_state(item) for item in value), key=repr)
        )
    if isinstance(value, Enum):
        return (type(value).__qualname__, value.value)
    if hasattr(value, "__dict__"):
        return (_qualname(value), _freeze_render_state(vars(value)))
    scalar = (str, int, float, bool, bytes, type(None))
    return value if isinstance(value, scalar) else repr(value)


def _freeze_render_state(value: object) -> Hashable:
    """Convert mutable render state into a stable, equality-safe value."""
    if isinstance(value, np.ndarray):
        return _freeze_ndarray(value)
    if isinstance(value, Annotation):
        return _freeze_fields(value, list(type(value).model_fields))
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze_fields(value, [f.name for f in fields(value)])
    if isinstance(value, dict):
        return _freeze_dict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_render_state(item) for item in value)
    return _freeze_leaf(value)


def _render_signature(
    annotations: list[Annotation],
    theme: Theme,
    gradient: "Gradient | str | None" = None,
) -> bytes:
    """Return a digest of all mutable state that can affect rendered pixels."""
    state = _freeze_render_state((annotations, theme, gradient))
    return hashlib.sha256(repr(state).encode("utf-8")).digest()


def _style_scale(width: int, height: int) -> float:
    """Resolution-aware style multiplier for a canvas of ``(width, height)``."""
    lo, hi = _STYLE_SCALE_RANGE
    return max(lo, min(hi, min(width, height) / _STYLE_REFERENCE_PX))


class Image:
    """Collect annotations over a base raster and render them as one scene.

    `Image` is mutable: `add` and `render_at` update the scene and return
    ``self`` for chaining. Rendering is lazy and cached by output size and
    annotation state. Export methods return fresh arrays or objects, so callers
    can modify their outputs without changing the `Image`.

    Attributes:
        width: Source-raster width in pixels.
        height: Source-raster height in pixels.
        annotations: Live list of top-level render annotations.
        theme: Explicit per-image theme, or ``None`` for the process default.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import BBox, Image
        >>> image = Image(np.zeros((60, 100, 3), np.uint8))
        >>> image.add(BBox(x=0.1, y=0.2, w=0.4, h=0.5, label="car"))
        Image(size=100x60, annotations=1)
        >>> image.render().shape
        (60, 100, 4)

    """

    def __init__(
        self,
        source: "ImageSource",
        *,
        mode: str = "rgb",
        options: RenderOptions | None = None,
        render_size: tuple[int, int] | None = None,
    ) -> None:
        """Create an image from any supported source.

        Args:
            source: A NumPy array, Pillow image, Torch tensor, or image path.
                NumPy arrays may be ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)``, or
                ``(H, W, 4)``; tensors may also be channel-first. See
                `vizlab.io.load_rgba`.
            mode: Channel order for array/tensor sources, ``"rgb"`` (default) or
                ``"bgr"`` (e.g. the output of ``cv2.imread``).
            options: `RenderOptions` supplying the theme (style/palette/background),
                the default gradient, and the LDF-adapter behavior used when LDF
                objects are added via `add`. ``None`` uses the options in effect
                for the current scope (see `vizlab.default_options`).
            render_size: Optional ``(width, height)`` display size. Mask fills are
                painted at the source resolution and the raster is scaled to this
                size once; strokes and labels are then drawn crisply at it (see
                `render`). ``None`` renders at the source resolution.

        Raises:
            TypeError: If ``source`` is not a supported image type.
            ValueError: If an array/tensor shape or channel mode is unsupported.
            FileNotFoundError: If an image path cannot be read or decoded.

        """
        self._rgba = io.load_rgba(source, mode)
        self._annotations: list[Annotation] = []
        self._options = options
        self._render_size = (
            None
            if render_size is None
            else (int(render_size[0]), int(render_size[1]))
        )
        self._cache: np.ndarray | None = None
        self._cache_key: tuple[tuple[int, int], bytes] | None = None

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._rgba.shape[1]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._rgba.shape[0]

    @property
    def annotations(self) -> list[Annotation]:
        """Return the live list of top-level annotations.

        Mutating this list or an annotation changes the next render; the render
        cache tracks the mutable scene state automatically. Use `copy` first
        when a separate top-level list is required.

        Returns:
            The mutable annotation list owned by this image.

        """
        return self._annotations

    @property
    def theme(self) -> Theme:
        """The theme this image renders with (its options', else the scope's)."""
        return self._resolve_options().theme

    @property
    def options(self) -> RenderOptions | None:
        """The explicit `RenderOptions` set on this image, or ``None``."""
        return self._options

    def _resolve_options(self) -> RenderOptions:
        """Return this image's options, falling back to the current scope's."""
        return (
            self._options if self._options is not None else current_options()
        )

    def base_rgba(self) -> np.ndarray:
        """Return a copy of the base raster, without any annotations drawn.

        Returns:
            A fresh ``(H, W, 4)`` ``uint8`` RGBA array of the underlying image.

        """
        return self._rgba.copy()

    def add(
        self,
        annotation: "Annotation | RenderableLDF",
        *,
        options: RenderOptions | None = None,
    ) -> Self:
        """Collect an annotation (native or LDF) to be drawn at render time.

        Nothing is rasterized here; the annotation is appended to the scene graph
        and this image is returned so calls can be chained.

        Accepts either a native vizlab `Annotation`
        or a Luxonis Data Format object — a
        `Detection`, a
        `DatasetRecord`, or a single annotation
        model — which is converted to render annotations natively (see
        `luxonis_ml.vizlab.convert`).

        Args:
            annotation: The annotation or LDF object to add.
            options: `RenderOptions` for converting LDF objects; falls back to the
                options passed to `Image`, then to the current scope's. Ignored
                for native annotations.

        Returns:
            This image, to allow ``img.add(...).add(...)`` chaining.

        """
        if isinstance(annotation, Annotation):
            self._annotations.append(annotation)
        else:
            from . import convert

            self._annotations.extend(
                convert.to_render_annotations(
                    annotation, options or self._resolve_options()
                )
            )
        self._cache = None
        return self

    def render_at(self, size: tuple[int, int] | None) -> Self:
        """Set the display render size and return ``self`` for chaining.

        This changes the default used by `render`; passing ``size`` directly to
        `render` overrides it for that call without updating this setting.

        Args:
            size: ``(width, height)`` to render at, or ``None`` for the source
                resolution. See `render` for how the size is used.

        Returns:
            This image, to allow fluent chaining.

        """
        self._render_size = (
            None if size is None else (int(size[0]), int(size[1]))
        )
        self._cache = None
        return self

    def copy(self) -> "Image":
        """Return a shallow clone sharing the base raster.

        The clone gets its own top-level annotation list, so adding or removing
        annotations on the clone does not affect the original. Annotation objects,
        nested children, the base raster, and options remain shared; mutate an
        annotation only when that change should be visible from both images.

        Returns:
            A new `Image` with the same base pixels and a copied annotation list.

        """
        clone = Image.__new__(Image)
        clone._rgba = self._rgba
        clone._annotations = list(self._annotations)
        clone._options = self._options
        clone._render_size = self._render_size
        clone._cache = None
        clone._cache_key = None
        return clone

    def render(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Rasterize the base image and all annotations, in two passes.

        Raster fills (mask overlays) are painted first, on a native-resolution
        canvas. The canvas is then scaled once to the display size, and the sharp
        vector layer (box strokes, mask contours, keypoints, label chips) is
        drawn on top at that size. This keeps labels and outlines crisp when the
        image is scaled for display while painting heavy fills only once, at the
        source resolution.

        The result is cached per size, active theme, and mutable scene-graph
        state. A copy is returned on every call.

        Args:
            size: ``(width, height)`` to render at; ``None`` uses the size set via
                `render_at` (the source resolution if unset).

        Returns:
            A fresh ``(H, W, 4)`` ``uint8`` RGBA array. The caller may mutate it
            freely; the cache holds a separate copy.

        """
        return self._render(size, capture=False)[0]

    def render_hits(
        self, size: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, HitMap]:
        """Render like `render`, also returning a hover `HitMap`.

        The map holds the display-pixel region of every annotation that carries a
        `Tooltip` (see `Annotation.tooltip`), so an interactive viewer can resolve
        the annotation under the cursor. Unlike `render`, this call is not cached;
        it is meant to run once per displayed frame.

        Args:
            size: ``(width, height)`` to render at; ``None`` uses the size set via
                `render_at` (the source resolution if unset).

        Returns:
            A ``(rgba, hitmap)`` pair. The RGBA array matches what `render` would
            return for the same ``size``.

        """
        rgba, hits = self._render(size, capture=True)
        return rgba, hits if hits is not None else HitMap.empty()

    def frame(self) -> "Frame":
        """Pair this image with its hover `HitMap` as a `Frame`.

        Captures the display-pixel region of every tooltip-bearing annotation
        (see `render_hits`) at the native render size and bundles it with the
        image, ready to hand to a `Viewer`. The map matches what `render` (no
        size) returns, which is the size the viewer renders before screen-fitting.

        Returns:
            A `Frame` wrapping this image and its hit map.

        """
        from .frame import Frame

        _, hitmap = self.render_hits()
        return Frame(self, hitmap)

    def _render(
        self, size: tuple[int, int] | None, *, capture: bool
    ) -> tuple[np.ndarray, HitMap | None]:
        """Shared render body for `render` (cached) and `render_hits` (uncapped).

        When ``capture`` is ``False`` the result is served from / stored in the
        render cache exactly as before and the second tuple element is ``None``.
        When ``capture`` is ``True`` the cache is bypassed and a `HitMap` of every
        tooltip-bearing annotation's region is collected instead.
        """
        target = size if size is not None else self._render_size
        render_size = (
            target if target is not None else (self.width, self.height)
        )
        options = self._resolve_options()
        theme = options.theme
        gradient = options.gradient
        key = (
            render_size,
            _render_signature(self._annotations, theme, gradient),
        )
        if not capture and self._cache is not None and self._cache_key == key:
            return self._cache.copy(), None

        # Background layers (semantic segmentation) render beneath every other
        # spatial annotation; overlays (image-level chrome) render on top. A
        # stable sort keeps add-order within each tier.
        spatial = sorted(
            (a for a in self._annotations if not a.OVERLAY),
            key=lambda a: not a.BACKGROUND,
        )
        overlays = [a for a in self._annotations if a.OVERLAY]

        # Pass 1: raster fills at the source resolution.
        canvas = self._render_fills(spatial, theme, gradient)
        # Scale the filled raster once to the display size.
        if target is not None and target != (canvas.width, canvas.height):
            canvas = canvas.scaled(target[0], target[1])
        # Pass 2: sharp vector content at the display resolution.
        hits: list[tuple[Rect, Tooltip]] | None = [] if capture else None
        self._render_vectors(canvas, spatial, overlays, theme, gradient, hits)

        rgba = canvas.to_rgba()
        if not capture:
            self._cache = rgba
            self._cache_key = key
            return rgba.copy(), None
        return rgba.copy(), HitMap(hits if hits is not None else [])

    def _render_fills(
        self,
        spatial: list[Annotation],
        theme: Theme,
        gradient: "Gradient | str | None",
    ) -> Canvas:
        """First pass: paint every annotation's raster fill at source resolution."""
        canvas = Canvas.from_rgba(self._rgba)
        ctx = RenderContext(
            canvas=canvas, depth=0, theme=theme, gradient=gradient
        )
        for annotation in spatial:
            annotation.render_fill(ctx)
        return canvas

    def _render_vectors(
        self,
        canvas: Canvas,
        spatial: list[Annotation],
        overlays: list[Annotation],
        theme: Theme,
        gradient: "Gradient | str | None",
        hits: list[tuple[Rect, Tooltip]] | None = None,
    ) -> None:
        """Second pass: crisp vector content and label chips at display resolution.

        Overlay label positions are reserved first so spatial labels avoid them,
        then spatial shapes, then their chips on top (so a later box never covers
        an earlier one's chip), then the overlays on top of everything. When
        ``hits`` is given, tooltip-bearing annotations append their region to it
        during the label pass (all in this canvas's display pixels).
        """
        ctx = RenderContext(
            canvas=canvas,
            depth=0,
            layout=LabelLayout(canvas.width, canvas.height),
            theme=theme,
            style_scale=_style_scale(canvas.width, canvas.height),
            hits=hits,
            gradient=gradient,
        )
        for annotation in overlays:
            annotation.reserve(ctx)
        for annotation in spatial:
            annotation.render(ctx)
        for annotation in spatial:
            annotation.render_labels(ctx)
        for annotation in overlays:
            annotation.render(ctx)

    def to_numpy(self, mode: str = "rgb") -> np.ndarray:
        """Render and return the result as a numpy array.

        Args:
            mode: Output layout: ``"rgb"`` (default), ``"bgr"``, ``"rgba"``, or
                ``"bgra"``.

        Returns:
            The rendered image in the requested channel layout.

        Raises:
            ValueError: If ``mode`` is not one of the supported layouts.

        """
        return io.export(self.render(), mode)

    def to_pil(self) -> "PILImage.Image":
        """Render and return the result as a PIL image.

        Returns:
            A PIL ``Image`` in RGBA mode.

        Raises:
            ImportError: If Pillow is not installed.

        """
        return io.to_pil(self.render())

    def save(self, path: str | Path, *, quality: int = 95) -> Self:
        """Render and write the image to a file.

        Args:
            path: Destination path; the format is inferred from the extension.
            quality: Encoder quality for lossy formats (0-100).

        Returns:
            This image, to allow chaining.

        Raises:
            ValueError: If the destination extension is not PNG, JPEG, or WebP.

        """
        io.save(self.render(), path, quality=quality)
        return self

    def show(self) -> None:
        """Render and open the image in Pillow's default viewer.

        Raises:
            ImportError: If Pillow is not installed.

        """
        self.to_pil().show()

    def blend(self, other: "Image", alpha: float = 0.3) -> "Image":
        """Blend this image with another (mixup), returning a new image.

        Only the base rasters are mixed; both images' annotations are carried
        onto the result and drawn crisply when it renders. Differently sized
        inputs are padded at the bottom and right, and their spatial annotations
        are transformed to remain aligned. See `vizlab.compose.blend`.

        Args:
            other: The image whose base is blended on top, weighted by ``alpha``.
            alpha: Weight of ``other`` in ``[0, 1]``.

        Returns:
            A new `Image` (blended base plus both label sets); neither input
            is mutated.

        """
        from . import compose

        return compose.blend(self, other, alpha)

    def with_panel(
        self,
        data: "PanelData",
        *,
        side: str = "right",
        width: float | None = None,
        title: str | None = None,
    ) -> "Image":
        """Append a metadata panel showing ``data`` and return a new image.

        Nested mappings and sequences are formatted as an indented tree. The
        panel is placed outside the rendered image, so it cannot cover pixels or
        labels. See `vizlab.panel.with_panel`.

        Args:
            data: JSON-like metadata (mapping/sequence/scalar, nested arbitrarily).
            side: Which edge to attach the panel to: ``"right"``, ``"left"``, or
                ``"bottom"``.
            width: Panel width in pixels; ``None`` auto-sizes from the content.
            title: Optional bold heading drawn above the tree.

        Returns:
            A new `Image` of this image plus the panel; not mutated.

        """
        from . import panel

        return panel.with_panel(
            self, data, side=side, width=width, title=title
        )

    def __repr__(self) -> str:
        """Return a compact source-size and annotation-count summary."""
        return f"Image(size={self.width}x{self.height}, annotations={len(self._annotations)})"
