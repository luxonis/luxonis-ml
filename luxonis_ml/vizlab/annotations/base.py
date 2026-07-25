"""The drawable annotation base and the render context passed down the scene graph.

Every drawable label is an `Annotation`. vizlab annotations reuse the
Luxonis Data Format data models: the spatial ones (`BBox`, `Keypoints`, `Mask`)
subclass their `luxonis_ml.ldf` counterpart and inherit its fields and parsing
(RLE decoding, normalized coordinates, keypoint layout), adding only the
rendering state — class label, confidence, generic payload, color/style
overrides, palette, and nested children — and the drawing itself.

Annotations form a tree: a box may carry child boxes (sub-labels), and children
are rendered after their parent with a `RenderContext` that carries the
parent's resolved color and style, so an unstyled child derives its look from its
parent (see `luxonis_ml.vizlab.style.derive`).
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.style import (
    DEFAULT_PALETTE,
    DEFAULT_STYLE,
    Palette,
    Style,
    Theme,
    derive_child_color,
    derive_child_style,
)

from .layout import LabelLayout

if TYPE_CHECKING:
    from luxonis_ml.vizlab.canvas import Canvas


@dataclass
class RenderContext:
    """State threaded through a single render pass.

    Attributes:
        canvas: The canvas being drawn on.
        depth: Nesting depth of the annotation currently being drawn (0 at the top
            level, incremented for each level of sub-labels).
        parent_color: The immediate parent's resolved color, or ``None`` at the top
            level. Used to derive an unstyled child's color.
        parent_style: The immediate parent's resolved style, or ``None`` at the top
            level. Used to derive an unstyled child's style.
        layout: Shared label-placement state for the whole render pass, used to keep
            label chips from overlapping. ``None`` outside an actual render.
        theme: The active theme, supplying default style/palette when an annotation
            sets none. ``None`` falls back to the library defaults.
        style_scale: Multiplier applied to every resolved style so pixel metrics
            (stroke, typography, chip padding) track the canvas resolution. ``1.0``
            leaves styles at their nominal sizes.

    """

    canvas: "Canvas"
    depth: int = 0
    parent_color: Color | None = None
    parent_style: Style | None = None
    layout: LabelLayout | None = None
    theme: Theme | None = None
    style_scale: float = 1.0

    def descend(self, color: Color, style: Style) -> "RenderContext":
        """Return a child context one level deeper, carrying resolved parent look.

        Args:
            color: The resolved color of the annotation being descended from.
            style: The resolved style of the annotation being descended from.

        Returns:
            A new `RenderContext` sharing the canvas, layout, and theme with
            ``depth + 1``.

        """
        return RenderContext(
            canvas=self.canvas,
            depth=self.depth + 1,
            parent_color=color,
            parent_style=style,
            layout=self.layout,
            theme=self.theme,
            style_scale=self.style_scale,
        )


class Annotation(BaseModel):
    """Abstract base for every drawable label.

    This is the vizlab *rendering* base. The spatial annotations subclass their
    Luxonis Data Format data model as well (e.g. ``BBox(BBoxAnnotation,
    Annotation)``), so their coordinate/mask data and parsing come from
    `luxonis_ml.ldf`; this base only adds the rendering state below.

    Attributes:
        label: Class name; shown on the label chip and used to pick a palette color.
        score: Confidence in ``[0, 1]``, rendered as a percentage on the chip.
        payload: Arbitrary value (e.g. transcribed OCR text) shown on the chip.
        color: Explicit color override; any :data:`ColorLike`.
        style: Style override; falls back to a parent-derived or the theme's style.
        palette: Palette used to pick a color from ``label``; ``None`` uses the
            theme's palette.
        children: Nested sub-label annotations, drawn on top of this one.

    Examples:
        Fluent helpers mutate and return the same annotation:

        >>> from luxonis_ml.vizlab import BBox
        >>> box = BBox(x=0.1, y=0.2, w=0.3, h=0.4)
        >>> box.tag("car", score=0.9).caption("track 7") is box
        True

        Nest a child annotation so it derives its parent's color and style:

        >>> box.add(BBox(x=0.2, y=0.3, w=0.1, h=0.1, label="plate")) is box
        True

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str | None = None
    score: float | None = None
    payload: str | int | float | None = None
    color: ColorLike | None = None
    style: Style | None = None
    palette: Palette | None = None
    children: list["Annotation"] = Field(default_factory=list)

    #: Whether this is image-level chrome drawn on top of everything (e.g. a tag).
    #: Overlays reserve their label positions first, then render last.
    OVERLAY: ClassVar[bool] = False

    def add(self, child: "Annotation") -> Self:
        """Attach a nested sub-label and return ``self`` for chaining.

        The child is rendered after its parent. Without explicit color/style
        overrides, it receives a derived color and style that visually indicates
        nesting.

        Args:
            child: The child annotation to nest inside this one.

        Returns:
            This annotation, to allow fluent chaining.

        """
        self.children.append(child)
        return self

    def tag(self, label: str, *, score: float | None = None) -> Self:
        """Set the class label (and optional score) and return ``self``.

        The label selects a palette color and is shown on the annotation's label
        chip. Setting ``score`` adds a percentage to that chip.

        Args:
            label: The class name.
            score: Optional confidence in ``[0, 1]``.

        Returns:
            This annotation, to allow fluent chaining.

        """
        self.label = label
        self.score = score
        return self

    def caption(self, value: str | float) -> Self:
        """Set the generic payload and return ``self``.

        Payload appears after the label and score on the annotation's chip. This
        is useful for recognized text, track identifiers, or another compact
        per-instance value.

        Args:
            value: The text/int/float value to show on the label chip.

        Returns:
            This annotation, to allow fluent chaining.

        """
        self.payload = value
        return self

    def with_style(self, style: Style) -> Self:
        """Set an explicit style override and return ``self``.

        Args:
            style: The style to draw this annotation with.

        Returns:
            This annotation, to allow fluent chaining.

        """
        self.style = style
        return self

    def resolved_palette(self, ctx: RenderContext) -> Palette:
        """Resolve the palette: explicit, else the theme's, else the library default.

        Args:
            ctx: The current render context.

        Returns:
            The `Palette` to draw with.

        """
        if self.palette is not None:
            return self.palette
        if ctx.theme is not None:
            return ctx.theme.palette
        return DEFAULT_PALETTE

    def resolve_style(self, ctx: RenderContext) -> Style:
        """Resolve the style: explicit, then parent-derived, then theme, then default.

        The resolved style is scaled by ``ctx.style_scale`` so pixel metrics track
        the canvas resolution. The parent-derived branch is left unscaled because
        it derives from the parent's already-scaled style.

        Args:
            ctx: The current render context.

        Returns:
            The `Style` to draw with.

        """
        if self.style is not None:
            return self.style.scaled(ctx.style_scale)
        if ctx.parent_style is not None:
            return derive_child_style(ctx.parent_style)
        if ctx.theme is not None:
            return ctx.theme.style.scaled(ctx.style_scale)
        return DEFAULT_STYLE.scaled(ctx.style_scale)

    def resolve_color(self, ctx: RenderContext) -> Color:
        """Resolve the color: override, then parent-derived, then label, then hash.

        A nested annotation without an explicit color derives from its parent (so a
        sub-label reads as a lighter shade of it); a top-level one uses its palette
        color from ``label``, or a stable per-object fallback when unlabeled.

        Args:
            ctx: The current render context.

        Returns:
            The resolved `Color`.

        """
        if self.color is not None:
            return Color.parse(self.color)
        if ctx.parent_color is not None:
            return derive_child_color(ctx.parent_color)
        palette = self.resolved_palette(ctx)
        if self.label is not None:
            return palette.color_for(self.label)
        return palette.color_for(f"{type(self).__name__}@{id(self):x}")

    def reserve(self, ctx: RenderContext) -> None:
        """Reserve any fixed label positions before spatial labels are placed.

        Overlays (image-level chrome) implement this so that boxes and masks avoid
        their chips even though the overlay is drawn last. The default is a no-op.

        Args:
            ctx: The current render context.

        """

    @abstractmethod
    def extent(self) -> Rect | None:
        """Return the annotation's axis-aligned bounds, if it has any.

        Returns:
            The bounding `Rect`, or ``None`` for
            annotations without a spatial extent (e.g. an image-level tag).

        """

    @abstractmethod
    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Draw the sharp (vector) layer of this annotation onto the canvas.

        This is the strokes, outlines, keypoints, and label chips — everything
        that must stay crisp. It runs in the second render pass, on the
        (possibly scaled) display-size canvas. Raster fills belong in
        `draw_fill` instead.

        Args:
            ctx: The current render context.
            style: This annotation's resolved style.
            color: This annotation's resolved color.

        """

    def draw_fill(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Draw the raster (fill) layer of this annotation onto the canvas.

        This runs in the first render pass, on the native-resolution canvas
        (before any display scaling), so pixel-heavy fills such as mask overlays
        are painted once at source resolution rather than resampled per draw.
        The default paints nothing; annotations with a raster fill (masks)
        override it. Vector content (strokes, chips) stays in `draw`.

        Args:
            ctx: The current render context (native resolution).
            style: This annotation's resolved style.
            color: This annotation's resolved color.

        """

    def draw_label(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Draw only this annotation's label chip, after all shapes are drawn.

        Runs in a pass after every annotation's `draw`, so label chips always
        sit on top of the boxes and masks — never behind a later shape. The
        default draws nothing; annotations with a label chip (boxes, masks)
        override it and omit the chip from `draw`.

        Args:
            ctx: The current render context.
            style: This annotation's resolved style.
            color: This annotation's resolved color.

        """

    def render(self, ctx: RenderContext) -> None:
        """Resolve this annotation, draw its vector layer, then render its children.

        Args:
            ctx: The current render context.

        """
        style = self.resolve_style(ctx)
        color = self.resolve_color(ctx)
        self.draw(ctx, style, color)
        child_ctx = ctx.descend(color, style)
        for child in self.children:
            child.render(child_ctx)

    def render_fill(self, ctx: RenderContext) -> None:
        """Resolve this annotation, draw its raster fill, then recurse into children.

        The first render pass, mirroring `render` but for the fill layer.

        Args:
            ctx: The current render context (native resolution).

        """
        style = self.resolve_style(ctx)
        color = self.resolve_color(ctx)
        self.draw_fill(ctx, style, color)
        child_ctx = ctx.descend(color, style)
        for child in self.children:
            child.render_fill(child_ctx)

    def render_labels(self, ctx: RenderContext) -> None:
        """Resolve this annotation, draw its label chip, then recurse children.

        The label pass, mirroring `render` but for the chip layer, run after all
        shapes so chips are never hidden behind a box or mask.

        Args:
            ctx: The current render context.

        """
        style = self.resolve_style(ctx)
        color = self.resolve_color(ctx)
        self.draw_label(ctx, style, color)
        child_ctx = ctx.descend(color, style)
        for child in self.children:
            child.render_labels(child_ctx)
