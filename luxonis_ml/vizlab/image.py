"""The `Image` composition root.

An `Image` wraps a base raster and a list of annotations (the scene graph).
Annotations are collected with `Image.add`; nothing is drawn until
`Image.render` is called, which rasterizes the base plus every annotation
in one pass.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from . import io
from .annotations.base import Annotation, RenderContext
from .annotations.layout import LabelLayout
from .canvas import Canvas
from .style import Theme, get_default_theme

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from .convert import VizConfig

#: Canvas short-side (px) at which styles render at their nominal size; larger
#: canvases scale labels/strokes up proportionally (clamped to the range below).
_STYLE_REFERENCE_PX = 760.0
_STYLE_SCALE_RANGE = (1.0, 3.0)


def _style_scale(width: int, height: int) -> float:
    """Resolution-aware style multiplier for a canvas of ``(width, height)``."""
    lo, hi = _STYLE_SCALE_RANGE
    return max(lo, min(hi, min(width, height) / _STYLE_REFERENCE_PX))


class Image:
    """A base image plus a collected list of annotations to draw on it."""

    def __init__(
        self,
        source: object,
        *,
        mode: str = "rgb",
        theme: Theme | None = None,
        config: "VizConfig | None" = None,
        render_size: tuple[int, int] | None = None,
    ) -> None:
        """Create an image from any supported source.

        Args:
            source: A numpy array, PIL image, torch tensor, or file path. See
                `vizlab.io.load_rgba` for the accepted shapes.
            mode: Channel order for array/tensor sources, ``"rgb"`` (default) or
                ``"bgr"`` (e.g. the output of ``cv2.imread``).
            theme: Theme supplying default style/palette; ``None`` uses the
                process-wide default (see `vizlab.style.set_default_theme`).
            config: Default `VizConfig` used when
                LDF objects are added via `add` without an explicit config.
            render_size: Optional ``(width, height)`` display size. Mask fills are
                painted at the source resolution and the raster is scaled to this
                size once; strokes and labels are then drawn crisply at it (see
                `render`). ``None`` renders at the source resolution.

        """
        self._rgba = io.load_rgba(source, mode)
        self._annotations: list[Annotation] = []
        self._theme = theme
        self._config = config
        self._render_size = (
            None
            if render_size is None
            else (int(render_size[0]), int(render_size[1]))
        )
        self._cache: np.ndarray | None = None
        self._cache_key: tuple[int, int] | None = None

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
        """The list of top-level annotations collected so far."""
        return self._annotations

    @property
    def theme(self) -> Theme | None:
        """The explicit theme set on this image, or ``None`` to use the default."""
        return self._theme

    def base_rgba(self) -> np.ndarray:
        """Return a copy of the base raster, without any annotations drawn.

        Returns:
            A fresh ``(H, W, 4)`` ``uint8`` RGBA array of the underlying image.

        """
        return self._rgba.copy()

    def add(
        self, annotation: object, *, config: "VizConfig | None" = None
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
            config: Rendering context for LDF objects; falls back to the config
                passed to `Image`, then to defaults. Ignored for native
                annotations.

        Returns:
            This image, to allow ``img.add(...).add(...)`` chaining.

        """
        if isinstance(annotation, Annotation):
            self._annotations.append(annotation)
        else:
            from . import convert

            self._annotations.extend(
                convert.to_render_annotations(
                    annotation, config or self._config
                )
            )
        self._cache = None
        return self

    def render_at(self, size: tuple[int, int] | None) -> Self:
        """Set the display render size and return ``self`` for chaining.

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

        The clone gets its own annotation list, so adding to it does not affect the
        original — useful for reusing a base image across several visualizations.

        Returns:
            A new `Image` with the same base pixels and a copied annotation list.

        """
        clone = Image.__new__(Image)
        clone._rgba = self._rgba
        clone._annotations = list(self._annotations)
        clone._theme = self._theme
        clone._config = self._config
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

        The result is cached per size until the scene graph changes (via `add`).

        Args:
            size: ``(width, height)`` to render at; ``None`` uses the size set via
                `render_at` (the source resolution if unset).

        Returns:
            A fresh ``(H, W, 4)`` ``uint8`` RGBA array. The caller may mutate it
            freely; the cache holds a separate copy.

        """
        target = size if size is not None else self._render_size
        key = target if target is not None else (self.width, self.height)
        if self._cache is not None and self._cache_key == key:
            return self._cache.copy()

        theme = self._theme if self._theme is not None else get_default_theme()
        spatial = [a for a in self._annotations if not a.OVERLAY]
        overlays = [a for a in self._annotations if a.OVERLAY]

        # Pass 1: raster fills at the source resolution.
        canvas = Canvas.from_rgba(self._rgba)
        fill_ctx = RenderContext(canvas=canvas, depth=0, theme=theme)
        for annotation in spatial:
            annotation.render_fill(fill_ctx)

        # Scale the filled raster once to the display size.
        if target is not None and target != (canvas.width, canvas.height):
            canvas = canvas.scaled(target[0], target[1])

        # Pass 2: sharp vector content at the display resolution. Style metrics
        # scale with the canvas so labels stay proportionate and readable.
        layout = LabelLayout(canvas.width, canvas.height)
        ctx = RenderContext(
            canvas=canvas,
            depth=0,
            layout=layout,
            theme=theme,
            style_scale=_style_scale(canvas.width, canvas.height),
        )
        # Reserve overlay label positions first so spatial labels avoid them,
        # draw the spatial shapes, then their label chips on top (so a later box
        # never covers an earlier one's chip), then overlays on top of all.
        for annotation in overlays:
            annotation.reserve(ctx)
        for annotation in spatial:
            annotation.render(ctx)
        for annotation in spatial:
            annotation.render_labels(ctx)
        for annotation in overlays:
            annotation.render(ctx)

        cache = canvas.to_rgba()
        self._cache = cache
        self._cache_key = key
        return cache.copy()

    def to_numpy(self, mode: str = "rgb") -> np.ndarray:
        """Render and return the result as a numpy array.

        Args:
            mode: Output layout: ``"rgb"`` (default), ``"bgr"``, ``"rgba"``, or
                ``"bgra"``.

        Returns:
            The rendered image in the requested channel layout.

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

        """
        io.save(self.render(), path, quality=quality)
        return self

    def show(self) -> None:
        """Render and open the image in the default viewer (needs Pillow)."""
        self.to_pil().show()

    def blend(self, other: "Image", alpha: float = 0.3) -> "Image":
        """Blend this image with another (mixup), returning a new image.

        Only the base rasters are mixed; both images' annotations are carried onto
        the result and drawn crisply when it renders. See
        `vizlab.compose.blend` for details.

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
        data: object,
        *,
        side: str = "right",
        width: float | None = None,
        title: str | None = None,
    ) -> "Image":
        """Append a metadata sidebar showing ``data``, returning a new image.

        See `vizlab.panel.with_panel` for details.

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
        return f"Image(size={self.width}x{self.height}, annotations={len(self._annotations)})"
