"""The `Image` composition root.

An `Image` wraps a base raster and a list of annotations (the scene graph).
Annotations are collected with `Image.add`; nothing is drawn until
`Image.render` is called, which rasterizes the base plus every annotation
in one pass.
"""

import hashlib
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from luxonis_ml.vizlab import io
from luxonis_ml.vizlab.annotations.base import Annotation, RenderContext
from luxonis_ml.vizlab.annotations.layout import LabelLayout
from luxonis_ml.vizlab.options import RenderOptions, current_options
from luxonis_ml.vizlab.render import RenderEnvironment
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.render.capture import (
    HitMap,
    InteractionCapture,
)
from luxonis_ml.vizlab.render.context import LayerMask
from luxonis_ml.vizlab.scene.html import Nav, html_document
from luxonis_ml.vizlab.style import Theme, style_scale

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from luxonis_ml.vizlab.adapters.ldf import RenderableLDF
    from luxonis_ml.vizlab.gradient import Gradient
    from luxonis_ml.vizlab.interaction.frame import Frame
    from luxonis_ml.vizlab.io import ImageSource
    from luxonis_ml.vizlab.layout.panel import PanelData


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


def _paints_fill(
    annotations: "Iterable[Annotation]", layers: "LayerMask"
) -> bool:
    """Whether any annotation ``layers`` admits has a raster fill to paint.

    Mirrors the walk `Annotation.render_fill` performs — the mask gates an
    annotation's own marks but never the recursion into its children — so this
    answers exactly what that pass would do without running it.

    Args:
        annotations: The scene's spatial annotations, in draw order.
        layers: The mask in effect.

    Returns:
        Whether the fill pass has anything to draw.

    """
    return any(
        (
            layers.draws(a.LAYER, a.label)
            and type(a).draw_fill is not Annotation.draw_fill
        )
        or _paints_fill(a.children, layers)
        for a in annotations
    )


def _tight_crop(rgba: np.ndarray) -> "tuple[np.ndarray, int, int] | None":
    """Reduce a raster to the region that is not fully transparent.

    Args:
        rgba: An ``(H, W, 4)`` ``uint8`` array.

    Returns:
        The cropped view and its ``(left, top)`` in the original, or ``None``
        when nothing at all is opaque.

    """
    alpha = rgba[..., 3]
    rows = np.flatnonzero(alpha.any(axis=1))
    if not rows.size:
        return None
    cols = np.flatnonzero(alpha.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    return rgba[top:bottom, left:right], left, top


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


class Renderable:
    """A scene that can render to pixels or SVG — an `Image` or a `Composite`.

    Subclasses supply their pixel `width`/`height`, their render options
    (`_resolve_options`), and how they paint themselves onto a canvas
    (`_draw_onto`). This base provides `render`, `render_svg`, `render_html`,
    and `save`; every output format uses the same scene-drawing path.
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

    def render_html(
        self,
        size: tuple[int, int] | None = None,
        *,
        title: str = "vizlab",
        text_as_paths: bool = True,
        controls: bool = True,
        nav: "Nav | None" = None,
    ) -> str:
        """Render the scene as one self-contained, interactive HTML page.

        The vector render of `render_svg` and the hover regions of `render_hits`
        come out of a single draw pass and are welded into one document, so every
        annotation carrying a `Tooltip` shows its card under the cursor in a
        browser — the same content the desktop viewer draws. The page is
        shareable as a single file: markup, styling, script, and the embedded
        bases are all inline, and it makes no external request.

        With ``controls`` the scene is additionally drawn once per layer and
        class, each pass under a
        `luxonis_ml.vizlab.render.context.layer_scope` that suppresses
        everything else. Each result becomes a targetable ``<g>``, so
        checkboxes in front of the figure can switch layers, classes and label
        chips off through CSS alone. Because the mask travels in the render
        environment rather than filtering a list of annotations, this works
        through composites — grids and panelled scenes included — whose
        children a caller cannot reach.

        The drawing is re-rooted on a ``viewBox`` and the hover regions ride
        inside it, so both scale with the page and stay aligned at any width.

        Args:
            size: ``(width, height)`` to render at; ``None`` uses the `render_at`
                size (the natural size if unset).
            title: The page's ``<title>``, as plain text.
            text_as_paths: Emit glyphs as outlines so the page renders identically
                anywhere without the fonts installed (default); turn off to keep
                selectable ``<text>`` that depends on the viewer's fonts.
            controls: Whether to emit the layer controls. ``False`` writes the
                scene as a single flat layer, in one pass.
            nav: Links to this page's neighbours, when it is one of a series
                written to a directory.

        Returns:
            The HTML document.

        """
        from luxonis_ml.vizlab.render.context import (
            label_plan,
            layer_inventory,
            layer_scope,
        )
        from luxonis_ml.vizlab.scene.html import (
            LAYERS,
            class_slugs,
            layered_document,
        )
        from luxonis_ml.vizlab.scene.html import (
            draws_anything as _draws_anything,
        )

        def page(svg: bytes, hits: HitMap, size_px: tuple[int, int]) -> str:
            return html_document(
                svg, hits, size_px, self.theme, nav=nav, title=title
            )

        if not controls:
            canvas, interactions, render_size = self._draw(
                size, capture=True, svg=True, text_as_paths=text_as_paths
            )
            assert interactions is not None
            return page(
                canvas.finish_svg(), HitMap(interactions.hover), render_size
            )

        # Every pass below places every label chip, painted or not, so that
        # splitting the scene into layers cannot move the chips that remain.
        # Placing is the most expensive part of a pass and always lands on the
        # same answer, so the first pass records it and the rest look it up. The
        # single-pass render above stays outside, both because it has nothing to
        # reuse and so that it remains an independent check on this one.
        with label_plan():
            # This pass's pixels are thrown away -- every layer is drawn again
            # on its own below -- and it serves only to collect the hover
            # regions and to see which layers the scene contains, which a
            # composite cannot be asked. A raster canvas answers both, and
            # spares a base64 PNG of the photo nobody would read.
            with layer_inventory() as found:
                _, interactions, render_size = self._draw(size, capture=True)
            assert interactions is not None
            hits = HitMap(interactions.hover)

            if not found:
                # Nothing to layer after all, and the pass above drew to raster.
                canvas, _, _ = self._draw(
                    render_size,
                    capture=False,
                    svg=True,
                    text_as_paths=text_as_paths,
                )
                return page(canvas.finish_svg(), hits, render_size)

            def draw(**scope: object) -> bytes:
                with layer_scope(**scope):  # type: ignore[arg-type]
                    return self.render_svg(
                        render_size, text_as_paths=text_as_paths
                    )

            # The picture and its chrome, drawn once, with no annotation on it.
            base = draw(kinds=(), labels=False)

            fragments: list[tuple[str, bytes]] = []
            classes: list[str] = []
            present: set[str] = set()
            # One unique slug per class name, shared by the fragments and the
            # controls; `slug` alone can fold "Car" and "car" into one id.
            slugs = class_slugs(
                dict.fromkeys(label for _, label in found if label is not None)
            )
            for layer, label in found:
                marks = draw(
                    kinds={layer}, classes={label}, labels=False, chrome=False
                )
                # An annotation can register a layer and still paint nothing — a
                # keypoint set with no visible joints, say. Its control would be
                # dead, so neither it nor its class earns one.
                if not _draws_anything(marks):
                    continue
                present.add(layer)
                css = ["vl-layer", f"vl-{layer}"]
                if label is not None:
                    css.append(f"vl-cls-{slugs[label]}")
                    if label not in classes:
                        classes.append(label)
                fragments.append((" ".join(css), marks))
            # Chips are their own layer so they can be switched off wholesale, and
            # split per class so hiding a class takes its chip with it. Only the
            # classes that have one contribute; an unlabeled annotation has none.
            labelled = False
            for label in classes:
                chips = draw(
                    kinds=(), classes={label}, labels=True, chrome=False
                )
                if not _draws_anything(chips):
                    continue
                labelled = True
                fragments.append(
                    (
                        f"vl-layer vl-label vl-cls-{slugs[label]}",
                        chips,
                    )
                )

            # LAYERS holds (slug, label) pairs and fixes the control order; only
            # layers that put marks on the page get a toggle, so a control never
            # switches nothing.
            kinds = [
                slug_name for slug_name, _ in LAYERS if slug_name in present
            ]
            palette = self.theme.palette
            swatches = {}
            for name in classes:
                tint = palette.color_for(name)
                swatches[name] = f"#{tint.r:02x}{tint.g:02x}{tint.b:02x}"
            return layered_document(
                base,
                fragments,
                hits,
                render_size,
                self.theme,
                kinds=[*kinds, "label"] if labelled else kinds,
                classes=classes,
                swatches=swatches,
                slugs=slugs,
                nav=nav,
                title=title,
            )

    def save(self, path: str | Path, *, quality: int = 95) -> Self:
        """Render and write the scene to a file (format from the extension).

        A ``.svg`` destination writes a vector render (`render_svg`), a ``.html``
        or ``.htm`` one an interactive page (`render_html`), and every other
        extension a raster encode of `render`.

        Args:
            path: Destination path; the format is inferred from the extension.
            quality: Encoder quality for lossy raster formats (0-100).

        Returns:
            This scene, to allow chaining.

        Raises:
            ValueError: If the destination extension is not SVG, HTML, PNG, JPEG,
                or WebP.

        """
        suffix = Path(path).suffix.lower()
        if suffix == ".svg":
            Path(path).write_bytes(self.render_svg())
        elif suffix in (".html", ".htm"):
            Path(path).write_text(self.render_html(), encoding="utf-8")
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

        For a scene baked to pixels — a composite flattened with
        ``Image(other.render())`` — whose tooltips no longer live on
        annotations. The map is remembered and returned by `render_hits` /
        `frame`, scaled to the render size like an annotation's would be.

        Args:
            hitmap: The hover map, in this scene's current pixel coordinates.

        Returns:
            This scene, to allow chaining.

        """
        self._hits = hitmap
        return self

    def capture(
        self, size: tuple[int, int] | None = None
    ) -> "tuple[InteractionCapture, RenderEnvironment]":
        """Draw the scene solely to collect its interaction regions.

        The public seam a `luxonis_ml.vizlab.Frame` is built from: it returns
        the raw collector plus the style snapshot the regions were captured
        under, and discards the pixels.

        Args:
            size: ``(width, height)`` to capture at; ``None`` uses the
                `render_at` size (the natural size if unset).

        Returns:
            The filled `InteractionCapture` and the `RenderEnvironment` it ran
            under.

        """
        environment = RenderEnvironment.current()
        _, interactions, _ = self._draw(
            size,
            capture=True,
            environment=environment,
        )
        assert interactions is not None
        return interactions, environment

    def frame(self) -> "Frame":
        """Capture this scene's interactions as a `Frame` for a `Viewer`.

        Sugar for `luxonis_ml.vizlab.Frame.capture`, which is where the pairing
        actually lives — this scene only knows how to hand over its `capture`.

        Returns:
            A `Frame` a `Viewer` can display.

        """
        from luxonis_ml.vizlab.interaction.frame import Frame

        return Frame.capture(self)

    def with_panel(
        self,
        data: "PanelData",
        *,
        side: str = "right",
        width: float | None = None,
        title: str | None = None,
    ) -> "Renderable":
        """Append a metadata panel showing ``data`` and return a composed scene.

        Sugar for `luxonis_ml.vizlab.with_panel`, which owns the layout; the
        panel is placed outside the rendered scene, so it cannot cover pixels or
        labels, and this scene is not mutated.

        Args:
            data: JSON-like metadata (mapping/sequence/scalar, nested arbitrarily).
            side: Which edge to attach the panel to: ``"right"``, ``"left"``, or
                ``"bottom"``.
            width: Panel width in pixels; ``None`` auto-sizes from the content.
            title: Optional bold heading drawn above the tree.

        Returns:
            A `Composite` of this scene plus the panel.

        """
        from luxonis_ml.vizlab.layout.panel import with_panel

        return with_panel(self, data, side=side, width=width, title=title)

    def _capture_carried(
        self,
        capture: InteractionCapture | None,
        x: float,
        y: float,
        size: tuple[int, int],
    ) -> None:
        """Emit a map attached with `with_hitmap` into ``capture``."""
        if capture is None or self._hits is None:
            return
        factor_x = size[0] / self.width if self.width else 1.0
        factor_y = size[1] / self.height if self.height else 1.0
        capture.transformed(x, y, factor_x, factor_y).add_hitmap(self._hits)


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
        `luxonis_ml.vizlab.adapters.ldf`).

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
            from luxonis_ml.vizlab.adapters import ldf

            self._annotations.extend(
                ldf.to_render_annotations(
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
        if base is not None:
            # The base may be a crop -- see `_render_fills` -- so it is placed
            # at the sub-rect its source-pixel origin maps to, not the whole one.
            rgba, left, top = base
            scale_x = size[0] / self.width
            scale_y = size[1] / self.height
            canvas.blit_scaled(
                rgba,
                x + left * scale_x,
                y + top * scale_y,
                rgba.shape[1] * scale_x,
                rgba.shape[0] * scale_y,
            )
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
    ) -> "tuple[np.ndarray, int, int] | None":
        """First pass: paint every annotation's raster fill at source resolution.

        The ``antialias`` flag is set on the canvas here and carried through the
        display-scaled canvas (`Canvas.scaled`) into the vector pass, so it
        governs every shape fill and stroke in the render.

        Returns:
            The base raster and its ``(left, top)`` origin in source pixels, or
            ``None`` when the pass paints nothing but transparency.

            Both concessions serve the layered export, which draws the scene
            once per layer with chrome suppressed. A box or keypoint layer has
            no fill at all, so it returns ``None`` rather than blitting an empty
            raster; a mask layer covers a car in a car park, so it returns that
            car's pixels rather than a full frame that is transparent
            everywhere else. Either way the cost of the blit — a resample on a
            raster canvas, a PNG encode on an SVG one — follows what the layer
            actually painted instead of the frame size.

            With chrome on, the photo covers the frame and the crop is the whole
            of it, so the ordinary render is unaffected.

        """
        # With chrome suppressed the photo itself is not drawn, but the fills
        # painted over it still are -- a mask or field layer *is* pixels, so
        # dropping the whole raster would drop the layer with it.
        chrome = environment.layers.chrome
        if not chrome and not _paints_fill(spatial, environment.layers):
            return None
        canvas = (
            Canvas.from_rgba(self._rgba, antialias=antialias)
            if chrome
            else Canvas.blank(self.width, self.height, antialias=antialias)
        )
        ctx = RenderContext(
            canvas=canvas,
            depth=0,
            theme=theme,
            environment=environment,
            gradient=gradient,
        )
        for annotation in spatial:
            annotation.render_fill(ctx)
        rgba = canvas.to_rgba()
        if chrome:
            return rgba, 0, 0
        return _tight_crop(rgba)

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
            style_scale=style_scale(canvas.width, canvas.height),
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

        Sugar for `luxonis_ml.vizlab.blend`, which owns the compositing: only
        the base rasters are mixed, both images' annotations are carried onto
        the result, and neither input is mutated.

        Args:
            other: The image whose base is blended on top, weighted by ``alpha``.
            alpha: Weight of ``other`` in ``[0, 1]``.

        Returns:
            A new `Image` — the blended base plus both label sets.

        """
        from luxonis_ml.vizlab.layout.compose import blend

        return blend(self, other, alpha)

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
