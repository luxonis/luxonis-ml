"""The drawable annotation base and the render context passed down the scene graph.

Every drawable label is an `Annotation`. vizlab annotations reuse the
Luxonis Data Format data models: the spatial ones (`BBox`, `Keypoints`, `Mask`)
subclass their `luxonis_ml.ldf` counterpart and inherit its fields and parsing
(RLE decoding, normalized coordinates, keypoint layout), adding only the
rendering state — class label, confidence, generic payload, color/style
overrides, palette, and nested children — and the drawing itself.

Annotations form a tree: a box may carry child boxes (sub-labels), and children
are rendered after their parent with a `RenderContext` that carries the parent's
resolved color and style. A labeled sub-label is its own class, so it keeps its
own palette color (a ``driver`` and a ``plate`` in a ``car`` are colored apart)
while it derives the parent's dashed sub-label style and traces its outline in the
parent's color to stay visibly nested; an *unlabeled* child derives the parent's
color outright (see `luxonis_ml.vizlab.style.derive` and `Annotation.outline_color`).
"""

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.hitmap import InteractionCapture
from luxonis_ml.vizlab.render import RenderEnvironment
from luxonis_ml.vizlab.style import (
    DEFAULT_PALETTE,
    DEFAULT_STYLE,
    Palette,
    Style,
    Theme,
    derive_child_color,
    derive_child_style,
)
from luxonis_ml.vizlab.tooltip import Tooltip

from .layout import LabelLayout

if TYPE_CHECKING:
    from luxonis_ml.vizlab.canvas import Canvas
    from luxonis_ml.vizlab.gradient import Gradient


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
        hits: Collector for hover hit entries, or ``None`` to capture none. When
            set, annotations that carry a `Tooltip` append their ``(region,
            tooltip)`` to it during the label pass (see `emit_hit`); the regions
            are in final display-pixel coordinates.
        capture: Render-time interaction collector. New rendering paths use this
            instead of ``hits`` so nested composites can transform hover and
            click regions with their pixels.
        environment: Scoped style state resolved once at the render boundary.
        gradient: Default colormap for heatmaps that set none (from the render
            options); ``None`` falls back to the library default gradient.

    """

    canvas: "Canvas"
    depth: int = 0
    parent_color: Color | None = None
    parent_style: Style | None = None
    layout: LabelLayout | None = None
    theme: Theme | None = None
    style_scale: float = 1.0
    hits: list[tuple[Rect, Tooltip]] | None = None
    capture: InteractionCapture | None = None
    environment: RenderEnvironment = dataclass_field(
        default_factory=RenderEnvironment.current
    )
    gradient: "Gradient | str | None" = None

    def descend(self, color: Color, style: Style) -> "RenderContext":
        """Return a child context one level deeper, carrying resolved parent look.

        Args:
            color: The resolved color of the annotation being descended from.
            style: The resolved style of the annotation being descended from.

        Returns:
            A new `RenderContext` sharing the canvas, layout, theme, and hit
            collector with ``depth + 1``.

        """
        return RenderContext(
            canvas=self.canvas,
            depth=self.depth + 1,
            parent_color=color,
            parent_style=style,
            layout=self.layout,
            theme=self.theme,
            style_scale=self.style_scale,
            hits=self.hits,
            capture=self.capture,
            environment=self.environment,
            gradient=self.gradient,
        )

    def emit_hit(self, region: Rect, tooltip: Tooltip | None) -> None:
        """Record a hover region for the current annotation, if collecting.

        A no-op unless a collector is attached (`hits`) and the annotation
        carries a `Tooltip`. ``region`` must be in the canvas's final display
        pixels (the label pass runs on the already-scaled canvas).

        Args:
            region: The annotation's axis-aligned bounds in display pixels.
            tooltip: The hover content to associate with ``region``, if any.

        """
        if tooltip is None:
            return
        if self.capture is not None:
            self.capture.add_hover(region, tooltip)
        elif self.hits is not None:
            self.hits.append((region, tooltip))


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
        style: Full style override that *replaces* the resolved theme/parent style
            (set via ``styled`` with a `Style`). ``None`` falls back to the
            parent-derived or theme style. For a partial tweak that keeps the rest
            of the theme, pass field overrides to ``styled`` instead.
        style_overrides: Partial style fields layered over the resolved base style
            (set via ``styled`` with keyword fields), so only the named fields
            change. Applied on top of any `luxonis_ml.vizlab.Style.override` scope;
            values are in display pixels.
        palette: Palette used to pick a color from ``label``; ``None`` uses the
            theme's palette.
        tooltip: Optional hover content surfaced by an interactive viewer when the
            annotation is hovered (see `luxonis_ml.vizlab.viewer`). Non-rendering
            state: it never draws, but `Image.render_hits` reports its region.
        children: Nested sub-label annotations, drawn on top of this one.

    Examples:
        Fluent helpers mutate and return the same annotation:

        >>> from luxonis_ml.vizlab import BBox
        >>> box = BBox(x=0.1, y=0.2, w=0.3, h=0.4)
        >>> box.tag("car", score=0.9).caption("track 7") is box
        True

        Nest a child annotation; a labeled sub-label keeps its own color and
        derives the parent's dashed style:

        >>> box.add(BBox(x=0.2, y=0.3, w=0.1, h=0.1, label="plate")) is box
        True

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str | None = None
    score: float | None = None
    payload: str | int | float | None = None
    color: ColorLike | None = None
    style: Style | None = None
    style_overrides: dict[str, Any] = Field(default_factory=dict)
    palette: Palette | None = None
    tooltip: Tooltip | None = None
    children: list["Annotation"] = Field(default_factory=list)

    #: Whether this is image-level chrome drawn on top of everything (e.g. a tag).
    #: Overlays reserve their label positions first, then render last.
    OVERLAY: ClassVar[bool] = False

    #: Whether this is a background layer drawn beneath every other spatial
    #: annotation (e.g. a semantic-segmentation map), so boxes, instance masks,
    #: and keypoints are never hidden under it.
    BACKGROUND: ClassVar[bool] = False

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

    def styled(
        self,
        style: "Style | Mapping[str, Any] | None" = None,
        /,
        **fields: Any,
    ) -> Self:
        """Set this annotation's style and return ``self``.

        Two modes, chosen by the argument:

        - **Field overrides** (keyword arguments, or a mapping) *layer* over the
          resolved theme/parent style — only the named fields change, everything
          else still comes from the theme. This is the usual case
          (``box.styled(stroke_width=6)``); values are in display pixels.
        - A full `Style` *replaces* the base style, ignoring the theme
          (``box.styled(Style(...))``). Any field overrides — passed here as
          keywords, or from an enclosing `luxonis_ml.vizlab.Style.override`
          scope — still layer on top of it.

        Args:
            style: A `Style` to use as the base (replacing the theme), or a mapping
                of `Style` field overrides to layer. ``None`` layers only
                ``fields``.
            **fields: `Style` field overrides layered on top.

        Returns:
            This annotation, to allow fluent chaining.

        Examples:
            >>> from luxonis_ml.vizlab import BBox, Style
            >>> box = BBox(x=0.1, y=0.2, w=0.3, h=0.4)
            >>> box.styled(stroke_width=6.0) is box  # layer over the theme
            True
            >>> box.styled(
            ...     Style(shadow=False)
            ... ) is box  # replace the base style
            True

        """
        if isinstance(style, Style):
            self.style = style.merge(**fields) if fields else style
        else:
            merged = {**self.style_overrides}
            if style is not None:
                merged.update(style)
            merged.update(fields)
            self.style_overrides = merged
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
        """Resolve the style, layering overrides over the base style.

        The base is a full `styled` style override (which replaces the theme), else
        the parent-derived style (when nested), else the scoped
        `luxonis_ml.vizlab.Style.as_default` style, else the theme's style, else the
        library default — scaled by ``ctx.style_scale`` so pixel metrics track the
        canvas resolution (the parent-derived branch is already scaled). Any
        `luxonis_ml.vizlab.Style.override` scope and this annotation's ``styled``
        field overrides are then layered on top, in display pixels.

        Args:
            ctx: The current render context.

        Returns:
            The `Style` to draw with.

        """
        if self.style is not None:
            base = self.style.scaled(ctx.style_scale)
        elif ctx.parent_style is not None:
            base = derive_child_style(ctx.parent_style)
        else:
            ambient = ctx.environment.default_style
            if ambient is not None:
                base = ambient.scaled(ctx.style_scale)
            elif ctx.theme is not None:
                base = ctx.theme.style.scaled(ctx.style_scale)
            else:
                base = DEFAULT_STYLE.scaled(ctx.style_scale)
        overrides = {
            **ctx.environment.style_overrides,
            **self.style_overrides,
        }
        return base.merge(**overrides) if overrides else base

    def resolve_color(self, ctx: RenderContext) -> Color:
        """Resolve the fill/chip color: override, then own class, then parent, then hash.

        A **labeled** annotation always takes its own class color from the palette,
        even when nested: a sub-label is a class in its own right, so a ``driver``
        and a ``plate`` inside a ``car`` are colored independently rather than as
        shades of the car. Only an *unlabeled* nested annotation (e.g. keypoints
        attached to a box) derives the parent's color, so it reads as part of the
        parent. The visual tie for labeled sub-labels is carried by the outline
        instead (see `outline_color`) and by the derived, dashed sub-label style.

        Args:
            ctx: The current render context.

        Returns:
            The resolved `Color`.

        """
        if self.color is not None:
            return Color.parse(self.color)
        palette = self.resolved_palette(ctx)
        if self.label is not None:
            return palette.color_for(self.label)
        if ctx.parent_color is not None:
            return derive_child_color(ctx.parent_color)
        return palette.color_for(f"{type(self).__name__}@{id(self):x}")

    def outline_color(self, ctx: RenderContext, color: Color) -> Color:
        """Resolve the color for this annotation's outline (box stroke, mask contour).

        A labeled sub-label keeps its own ``color`` for its fill and label chip but
        traces its outline in the parent's color, so it reads as its own class while
        still visibly belonging to its parent. Top-level annotations, and ones with
        an explicit ``color`` override, outline in their own ``color``.

        Args:
            ctx: The current render context.
            color: This annotation's own resolved (fill/chip) color.

        Returns:
            The color to stroke the outline with.

        """
        if (
            self.color is None
            and self.label is not None
            and ctx.parent_color is not None
        ):
            return ctx.parent_color
        return color

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
