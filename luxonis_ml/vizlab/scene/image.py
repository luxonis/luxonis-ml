"""The `Image` composition root.

An `Image` wraps a base raster and a list of annotations (the scene graph).
Annotations are collected with `Image.add`; nothing is drawn until
`Image.render` is called, which rasterizes the base plus every annotation
in one pass.
"""

import hashlib
from collections.abc import Callable, Hashable, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from luxonis_ml.vizlab import io
from luxonis_ml.vizlab.annotations.base import Annotation, RenderContext
from luxonis_ml.vizlab.annotations.layout import LabelLayout
from luxonis_ml.vizlab.canvas import Canvas
from luxonis_ml.vizlab.hitmap import ClickMap, HitMap, InteractionCapture
from luxonis_ml.vizlab.options import RenderOptions, current_options
from luxonis_ml.vizlab.render import RenderEnvironment
from luxonis_ml.vizlab.style import Theme

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from luxonis_ml.vizlab.adapters.ldf import RenderableLDF
    from luxonis_ml.vizlab.frame import Frame
    from luxonis_ml.vizlab.gradient import Gradient
    from luxonis_ml.vizlab.io import ImageSource
    from luxonis_ml.vizlab.panel import PanelData

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


def _freeze_dict(value: Mapping[object, object]) -> Hashable:
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
    options: RenderOptions,
    environment: RenderEnvironment,
) -> bytes:
    """Return a digest of all mutable state that can affect rendered pixels."""
    state = _freeze_render_state(
        (
            annotations,
            options,
            environment,
        )
    )
    return hashlib.sha256(repr(state).encode("utf-8")).digest()


def _style_scale(width: int, height: int) -> float:
    """Resolution-aware style multiplier for a canvas of ``(width, height)``."""
    lo, hi = _STYLE_SCALE_RANGE
    return max(lo, min(hi, min(width, height) / _STYLE_REFERENCE_PX))


class Renderable:
    """A scene that can render to pixels or SVG — an `Image` or a `Composite`.

    Subclasses supply their pixel `width`/`height`, their render options
    (`_resolve_options`), and how they paint themselves onto a canvas
    (`_draw_onto`). This base provides `render`, `render_svg`, and `save`; both
    output formats use the same scene-drawing path.
    """

    _render_size: tuple[int, int] | None = None
    _options: RenderOptions | None = None
    #: A precomputed hover map (in this scene's own pixels), carried by composites
    #: whose tiles' tooltips no longer live on annotations — see `with_hitmap`.
    _hits: "HitMap | None" = None

    @property
    def width(self) -> int:
        """Scene width in pixels."""
        raise NotImplementedError

    @property
    def height(self) -> int:
        """Scene height in pixels."""
        raise NotImplementedError

    def copy(self) -> "Renderable":
        """Return an independent clone that renders the same scene."""
        raise NotImplementedError

    @property
    def options(self) -> RenderOptions | None:
        """The explicit `RenderOptions` set on this scene, or ``None``."""
        return self._options

    @property
    def theme(self) -> Theme:
        """The theme this scene renders with (its options', else the scope's)."""
        return self._resolve_options().theme

    def _resolve_options(self) -> RenderOptions:
        """Return this scene's options, falling back to the current scope's."""
        return (
            self._options if self._options is not None else current_options()
        )

    def _draw_onto(
        self,
        canvas: Canvas,
        x: float,
        y: float,
        size: tuple[int, int],
        *,
        environment: RenderEnvironment,
        capture: InteractionCapture | None = None,
    ) -> None:
        """Draw the whole scene into ``canvas`` at the rect ``(x, y, size)``."""
        raise NotImplementedError

    def _invalidate(self) -> None:
        """Drop any cached render (a no-op unless a subclass caches)."""

    def render_at(self, size: tuple[int, int] | None) -> Self:
        """Set the display render size and return ``self`` for chaining.

        Changes the default used by `render`; passing ``size`` directly to
        `render` overrides it for that call without updating this setting.

        Args:
            size: ``(width, height)`` to render at, or ``None`` for the natural
                size.

        Returns:
            This scene, to allow fluent chaining.

        """
        self._render_size = (
            None if size is None else (int(size[0]), int(size[1]))
        )
        self._invalidate()
        return self

    def _resolved_size(self, size: tuple[int, int] | None) -> tuple[int, int]:
        """Return the render size for ``size`` (falling back to `render_at`)."""
        target = size if size is not None else self._render_size
        return target if target is not None else (self.width, self.height)

    def _draw(
        self,
        size: tuple[int, int] | None,
        *,
        capture: bool,
        svg: bool = False,
        text_as_paths: bool = True,
        environment: RenderEnvironment | None = None,
    ) -> "tuple[Canvas, InteractionCapture | None, tuple[int, int]]":
        """Draw the scene and return its canvas, interactions, and render size.

        Raster and SVG output differ only in the blank canvas created here. The
        scene itself always draws through `_draw_onto`.
        """
        environment = environment or RenderEnvironment.current()
        render_size = self._resolved_size(size)
        antialias = self._resolve_options().antialias
        canvas = (
            Canvas.svg(
                *render_size,
                antialias=antialias,
                text_as_paths=text_as_paths,
            )
            if svg
            else Canvas.blank(*render_size, antialias=antialias)
        )
        interactions = InteractionCapture() if capture else None
        self._draw_onto(
            canvas,
            0,
            0,
            render_size,
            environment=environment,
            capture=interactions,
        )
        return canvas, interactions, render_size

    def render(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Rasterize the scene to an ``(H, W, 4)`` ``uint8`` RGBA array.

        Args:
            size: ``(width, height)`` to render at; ``None`` uses the `render_at`
                size (the natural size if unset).

        Returns:
            A fresh RGBA array; the caller may mutate it freely.

        """
        canvas, _, _ = self._draw(size, capture=False)
        return canvas.to_rgba()

    def _render_in(
        self,
        environment: RenderEnvironment,
        size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Render using an already-resolved ambient-style environment."""
        canvas, _, _ = self._draw(
            size,
            capture=False,
            environment=environment,
        )
        return canvas.to_rgba()

    def render_svg(
        self,
        size: tuple[int, int] | None = None,
        *,
        text_as_paths: bool = True,
    ) -> bytes:
        """Render the scene as an SVG: vectors over embedded raster bases.

        Each image's photo (plus mask fills) embeds once as a base64 ``<image>``;
        every other mark — box strokes, mask contours, keypoints, label chips, and
        all panel/grid chrome — is emitted as true SVG vector elements, crisp at
        any zoom.

        Args:
            size: ``(width, height)`` to render at; ``None`` uses the `render_at`
                size (the natural size if unset).
            text_as_paths: Emit glyphs as outlines so the SVG renders identically
                anywhere without the fonts installed (default); turn off to keep
                selectable ``<text>`` that depends on the viewer's fonts.

        Returns:
            The SVG document as UTF-8 bytes.

        """
        canvas, _, _ = self._draw(
            size,
            capture=False,
            svg=True,
            text_as_paths=text_as_paths,
        )
        return canvas.finish_svg()

    def save(self, path: str | Path, *, quality: int = 95) -> Self:
        """Render and write the scene to a file (format from the extension).

        A ``.svg`` destination writes a vector render (`render_svg`); every other
        extension writes a raster encode of `render`.

        Args:
            path: Destination path; the format is inferred from the extension.
            quality: Encoder quality for lossy raster formats (0-100).

        Returns:
            This scene, to allow chaining.

        Raises:
            ValueError: If the destination extension is not SVG, PNG, JPEG, or
                WebP.

        """
        if Path(path).suffix.lower() == ".svg":
            Path(path).write_bytes(self.render_svg())
        else:
            io.save(self.render(), path, quality=quality)
        return self

    def to_numpy(self, mode: str = "rgb") -> np.ndarray:
        """Render and return the scene as a NumPy array.

        Args:
            mode: Output layout: ``"rgb"`` (default), ``"bgr"``, ``"rgba"``, or
                ``"bgra"``.

        Returns:
            The rendered scene in the requested channel layout.

        Raises:
            ValueError: If ``mode`` is unsupported.

        """
        return io.export(self.render(), mode)

    def to_pil(self) -> "PILImage.Image":
        """Render and return the scene as a Pillow RGBA image."""
        return io.to_pil(self.render())

    def show(self) -> None:
        """Render and open the scene in Pillow's default viewer."""
        self.to_pil().show()

    def render_hits(
        self, size: tuple[int, int] | None = None
    ) -> "tuple[np.ndarray, HitMap]":
        """Render the scene, also returning a hover `HitMap`.

        The map holds the display-pixel region of every annotation that carries a
        `Tooltip`, so an interactive viewer can resolve the annotation under the
        cursor. Not cached (meant to run once per displayed frame).

        Args:
            size: ``(width, height)`` to render at; ``None`` uses the `render_at`
                size (the natural size if unset).

        Returns:
            A ``(rgba, hitmap)`` pair. The RGBA matches what `render` would return.

        """
        canvas, interactions, _ = self._draw(size, capture=True)
        rgba = canvas.to_rgba()
        hitmap = HitMap(interactions.hover if interactions is not None else [])
        return rgba, hitmap

    def with_hitmap(self, hitmap: HitMap) -> Self:
        """Attach a precomputed hover `HitMap` (in this scene's pixels).

        For composites whose tiles were flattened so their tooltips no longer
        live on annotations: the map is remembered and returned by `render_hits`
        / `frame`, scaled to the render size like an annotation's would be.

        Args:
            hitmap: The hover map, in this scene's current pixel coordinates.

        Returns:
            This scene, to allow chaining.

        """
        self._hits = hitmap
        return self

    def frame(self) -> "Frame":
        """Capture this scene's interactions as a `Frame` for a `Viewer`."""
        from luxonis_ml.vizlab.frame import Frame

        environment = RenderEnvironment.current()
        _, interactions, _ = self._draw(
            None,
            capture=True,
            environment=environment,
        )
        assert interactions is not None
        return Frame(
            self,
            HitMap(interactions.hover),
            ClickMap(interactions.clicks),
            environment,
        )

    def _capture_carried(
        self,
        capture: InteractionCapture | None,
        x: float,
        y: float,
        size: tuple[int, int],
    ) -> None:
        """Emit a legacy map attached with `with_hitmap` into ``capture``."""
        if capture is None or self._hits is None:
            return
        factor_x = size[0] / self.width if self.width else 1.0
        factor_y = size[1] / self.height if self.height else 1.0
        capture.transformed(x, y, factor_x, factor_y).add_hitmap(self._hits)

    def with_panel(
        self,
        data: "PanelData",
        *,
        side: str = "right",
        width: float | None = None,
        title: str | None = None,
    ) -> "Renderable":
        """Append a metadata panel showing ``data`` and return a composed scene.

        Nested mappings and sequences are formatted as an indented tree. The
        panel is placed outside the rendered scene, so it cannot cover pixels or
        labels. See `vizlab.panel.with_panel`.

        Args:
            data: JSON-like metadata (mapping/sequence/scalar, nested arbitrarily).
            side: Which edge to attach the panel to: ``"right"``, ``"left"``, or
                ``"bottom"``.
            width: Panel width in pixels; ``None`` auto-sizes from the content.
            title: Optional bold heading drawn above the tree.

        Returns:
            A `Composite` of this scene plus the panel — renders to raster or SVG.

        """
        from luxonis_ml.vizlab import panel

        return panel.with_panel(
            self, data, side=side, width=width, title=title
        )


class Image(Renderable):
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
        # A precomputed hover map (in this image's own pixels), carried by
        # composites whose tiles were baked to pixels — see `with_hitmap`.
        self._hits: HitMap | None = None

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
            from luxonis_ml.vizlab import convert

            self._annotations.extend(
                convert.to_render_annotations(
                    annotation, options or self._resolve_options()
                )
            )
        self._cache = None
        return self

    def _invalidate(self) -> None:
        """Drop the render cache (called when the render size changes)."""
        self._cache = None

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
        clone._hits = self._hits
        return clone

    def render(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Rasterize the base image and all annotations, in two passes.

        Raster fills (mask overlays) are painted first, on a native-resolution
        canvas. The canvas is then scaled once to the display size, and the sharp
        vector layer (box strokes, mask contours, keypoints, label chips) is
        drawn on top at that size. This keeps labels and outlines crisp when the
        image is scaled for display while painting heavy fills only once, at the
        source resolution.

        The result is cached per size, resolved render options, scoped style, and
        mutable scene-graph state. A copy is returned on every call.

        Args:
            size: ``(width, height)`` to render at; ``None`` uses the size set via
                `render_at` (the source resolution if unset).

        Returns:
            A fresh ``(H, W, 4)`` ``uint8`` RGBA array. The caller may mutate it
            freely; the cache holds a separate copy.

        """
        # Cache by render size and every resolved/ambient pixel input;
        # render_hits (the capture path) runs once per frame and never consults
        # this cache.
        options = self._resolve_options()
        environment = RenderEnvironment.current()
        key = (
            self._resolved_size(size),
            _render_signature(self._annotations, options, environment),
        )
        if self._cache is not None and self._cache_key == key:
            return self._cache.copy()
        canvas, _, _ = self._draw(
            size,
            capture=False,
            environment=environment,
        )
        rgba = canvas.to_rgba()
        self._cache = rgba
        self._cache_key = key
        return rgba.copy()

    def _draw_onto(
        self,
        canvas: Canvas,
        x: float,
        y: float,
        size: tuple[int, int],
        *,
        environment: RenderEnvironment,
        capture: InteractionCapture | None = None,
    ) -> None:
        """Draw this image's scene into ``canvas`` at the rect ``(x, y, size)``.

        Two passes, backend-agnostic: the base (photo plus mask fills, painted at
        the source resolution) is laid into the rect via `Canvas.blit_scaled` — a
        resample on a raster canvas, one embedded ``<image>`` on an SVG one — and
        the crisp vector layer (strokes, contours, keypoints, label chips) is drawn
        over it in a `Canvas.viewport` local to the rect. This is how a composite
        draws a sub-image, and how `_draw` draws the whole image (rect ``(0, 0)``
        to the render size). ``capture`` collects interaction regions and maps
        them into the final output coordinates when given.
        """
        options = self._resolve_options()
        theme = options.theme
        gradient = options.gradient
        # Background layers (semantic segmentation) render beneath every other
        # spatial annotation; overlays (image-level chrome) render on top. A
        # stable sort keeps add-order within each tier.
        spatial = sorted(
            (a for a in self._annotations if not a.OVERLAY),
            key=lambda a: not a.BACKGROUND,
        )
        overlays = [a for a in self._annotations if a.OVERLAY]
        base = self._render_fills(
            spatial,
            theme,
            gradient,
            environment,
            options.antialias,
        )
        canvas.blit_scaled(base.to_rgba(), x, y, size[0], size[1])
        local_capture = (
            capture.transformed(x, y) if capture is not None else None
        )
        with canvas.viewport(x, y, size[0], size[1]) as region:
            self._render_vectors(
                region,
                spatial,
                overlays,
                theme,
                gradient,
                environment,
                local_capture,
            )
        self._capture_carried(capture, x, y, size)

    def _render_fills(
        self,
        spatial: list[Annotation],
        theme: Theme,
        gradient: "Gradient | str | None",
        environment: RenderEnvironment,
        antialias: bool = True,
    ) -> Canvas:
        """First pass: paint every annotation's raster fill at source resolution.

        The ``antialias`` flag is set on the canvas here and carried through the
        display-scaled canvas (`Canvas.scaled`) into the vector pass, so it
        governs every shape fill and stroke in the render.
        """
        canvas = Canvas.from_rgba(self._rgba, antialias=antialias)
        ctx = RenderContext(
            canvas=canvas,
            depth=0,
            theme=theme,
            environment=environment,
            gradient=gradient,
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
        environment: RenderEnvironment,
        capture: InteractionCapture | None = None,
    ) -> None:
        """Second pass: crisp vector content and label chips at display resolution.

        Overlay label positions are reserved first so spatial labels avoid them,
        then spatial shapes, then their chips on top (so a later box never covers
        an earlier one's chip), then the overlays on top of everything. When
        ``capture`` is given, tooltip-bearing annotations emit their regions
        during the label pass.
        """
        ctx = RenderContext(
            canvas=canvas,
            depth=0,
            layout=LabelLayout(canvas.width, canvas.height),
            theme=theme,
            style_scale=_style_scale(canvas.width, canvas.height),
            capture=capture,
            environment=environment,
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
        from luxonis_ml.vizlab import compose

        return compose.blend(self, other, alpha)

    def __repr__(self) -> str:
        """Return a compact source-size and annotation-count summary."""
        return f"Image(size={self.width}x{self.height}, annotations={len(self._annotations)})"


#: A composite's scene painter: draws the whole layout, at natural coordinates,
#: onto the canvas it is given (child images via their own `Image._draw_onto`,
#: chrome via vector primitives).
ScenePaint = Callable[
    ["Canvas", RenderEnvironment, InteractionCapture | None],
    None,
]


class Composite(Renderable):
    """A renderable assembled from placed child scenes plus vector chrome.

    Produced by the composition helpers (`luxonis_ml.vizlab.with_panel`, the grid builders):
    it holds a natural pixel size and a `ScenePaint` that draws the whole layout —
    child images via their own `Image._draw_onto` (so their annotations stay
    vector in an SVG) and every border, panel, title, and legend as vector
    primitives. A composite renders to raster or SVG and can save either format;
    rendering at a size other than its natural one scales the whole layout
    uniformly.

    Attributes:
        width: Composite width in pixels.
        height: Composite height in pixels.

    """

    def __init__(
        self,
        size: tuple[int, int],
        paint: ScenePaint,
        *,
        options: RenderOptions | None = None,
    ) -> None:
        """Assemble a composite.

        Args:
            size: The composite's natural ``(width, height)`` in pixels.
            paint: Draws the layout at natural coordinates onto a given canvas.
            options: Render options (theme/antialias); ``None`` uses the scope's.

        """
        self._size = (int(size[0]), int(size[1]))
        self._scene = paint
        self._options = options
        self._render_size: tuple[int, int] | None = None

    @property
    def width(self) -> int:
        """Composite width in pixels."""
        return self._size[0]

    @property
    def height(self) -> int:
        """Composite height in pixels."""
        return self._size[1]

    def _draw_onto(
        self,
        canvas: Canvas,
        x: float,
        y: float,
        size: tuple[int, int],
        *,
        environment: RenderEnvironment,
        capture: InteractionCapture | None = None,
    ) -> None:
        """Draw the composite into ``canvas`` at ``(x, y, size)``.

        The scene is painted at its natural coordinates in a `Canvas.viewport`
        that maps them onto the rect (scaling uniformly when ``size`` differs from
        the natural size). The same transform is applied to interaction regions
        emitted by nested scenes.
        """
        scene_capture = (
            capture.transformed(
                x,
                y,
                size[0] / self.width if self.width else 1.0,
                size[1] / self.height if self.height else 1.0,
            )
            if capture is not None
            else None
        )
        with canvas.viewport(
            x, y, size[0], size[1], logical=self._size
        ) as region:
            self._scene(region, environment, scene_capture)
        self._capture_carried(capture, x, y, size)

    def copy(self) -> "Composite":
        """Return a clone sharing the scene painter but with its own state."""
        clone = Composite(self._size, self._scene, options=self._options)
        clone._render_size = self._render_size
        clone._hits = self._hits
        return clone

    def __repr__(self) -> str:
        """Return a compact size summary."""
        return f"Composite(size={self.width}x{self.height})"
