"""Keypoints with an optional skeleton of limb edges.

`Keypoints` subclasses the Luxonis Data Format
`KeypointAnnotation`, reusing its normalized
``(x, y, visibility)`` keypoint list (COCO visibility ``0``/``1``/``2``) and
adding rendering: joints, and — given ``edges`` — the limbs connecting them. A
dataset's skeleton definition is a ``(labels, edges)`` pair per task (as
``LuxonisDataset.get_skeletons`` returns); pass its ``edges`` and, for named
point labels, its ``labels`` as ``keypoint_names``.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from luxonis_ml.ldf import KeypointAnnotation
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import XY, Rect
from luxonis_ml.vizlab.style import Palette, Style

from .base import Annotation, RenderContext

if TYPE_CHECKING:
    from luxonis_ml.vizlab.render.canvas import Canvas

_WHITE = Color(255, 255, 255)
#: Dark outline for occluded joints (vs the white outline of visible ones).
_DARK = Color(25, 25, 25)


#: A diamond (occluded joint) is scaled up so it reads as visually heavy as the
#: circle (visible joint) of the same nominal radius.
_DIAMOND_SCALE = 1.3
#: Fill opacity of an occluded joint, so it also reads dimmer than a visible one.
_OCCLUDED_ALPHA = 0.55

#: Dash pattern of a limb running to a joint the data does not place.
_ABSENT_DASH = (5.0, 4.0)
#: How far short of the mark that limb stops, in joint radii, so the mark reads
#: as a joint of its own rather than a bead threaded onto the limb.
_ABSENT_GAP = 2.0
#: Half-diagonal of the cross standing in for an unplaced joint, in joint radii.
_ABSENT_ARM = 1.25
_ABSENT_LIMB_ALPHA = 0.45
_ABSENT_MARK_ALPHA = 0.9

PointLabelMode = Literal["none", "numbers", "names", "full"]
"""How to label each keypoint: nothing, its index, its name, or ``index:name``."""


def _stop_short(start: XY, end: XY, gap: float) -> XY | None:
    """Pull ``end`` back along the segment by ``gap``, or ``None`` if too short.

    Examples:
        >>> _stop_short((0.0, 0.0), (10.0, 0.0), 4.0)
        (6.0, 0.0)
        >>> _stop_short((0.0, 0.0), (3.0, 0.0), 4.0) is None
        True

    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length <= gap:
        return None
    scale = (length - gap) / length
    return (start[0] + dx * scale, start[1] + dy * scale)


class Keypoints(KeypointAnnotation, Annotation):
    """A set of keypoints, optionally wired into a skeleton.

    Reuses the normalized ``(x, y, visibility)`` ``keypoints`` list of
    `KeypointAnnotation`.

    .. image:: TODO-HOST/masks_keypoints.png
       :alt: Keypoints alongside instance, polygon, and semantic masks.

    A joint the data does not place — the ``(0, 0, 0)`` a dataset writes for one
    it never labeled, and a model for one it did not predict — is marked rather
    than hidden, whenever ``edges`` say where it belongs. Hiding it would
    silently amputate the skeleton: the limb stops, anything past the gap floats
    free, and the pose reads as a *different* pose rather than an incomplete
    one. Instead the joint is drawn as a cross at the position the skeleton
    implies, joined to its present neighbors by dashed limbs, so the pose stays
    whole while the cross says the joint was never placed rather than claiming
    it sits there. See `Keypoints._absent_positions` for which gaps the skeleton
    can answer for; the rest stay hidden, since a guess drawn confidently is
    worse than an admitted gap.

    Attributes:
        edges: Pairs of keypoint indices to connect with a limb; empty draws
            joints only.
        keypoint_names: Optional per-keypoint names (index order), used by the
            ``"names"``/``"full"`` point-label modes.
        visibility_threshold: Points whose visibility is ``<=`` this are hidden.
            With COCO visibility, a joint marked visible (``2``) is a bright dot
            with a white outline; one marked occluded (``1``) is a dimmer diamond
            with a dark outline, so it clearly recedes.
        point_labels: How to label each joint — ``"none"`` (default),
            ``"numbers"`` (its index), ``"names"`` (its keypoint name), or
            ``"full"`` (``index:name``). ``"names"``/``"full"`` fall back to the
            index when no name is available.
        point_colors: Optional per-joint color overrides, index-aligned to
            ``keypoints`` (a ``None`` entry, or a short list, falls back to the
            instance color). A limb between two differently colored joints is
            drawn as a gradient from one color to the other. Useful for grading
            individual joints — see `luxonis_ml.vizlab.comparison.render`. The
            marks on absent joints keep the instance color either way: a joint
            neither side placed has no verdict worth painting.

    See `Annotation` for the shared
    ``label``, ``color``, ``style``, and ``palette`` fields.

    Examples:
        >>> kp = Keypoints(keypoints=[(0.1, 0.2, 2), (0.3, 0.4, 0)])
        >>> xy, vis = kp._resolve(200, 100)
        >>> xy.tolist()
        [[20.0, 20.0], [60.0, 40.0]]
        >>> vis.tolist()
        [2.0, 0.0]

    """

    LAYER: ClassVar[str] = "keypoint"

    edges: list[tuple[int, int]] = []
    keypoint_names: list[str] | None = None
    visibility_threshold: float = 0.0
    point_labels: PointLabelMode = "none"
    point_colors: list[ColorLike | None] | None = None

    @classmethod
    def from_ldf(
        cls,
        annotation: KeypointAnnotation,
        *,
        edges: Iterable[tuple[int, int]] = (),
        keypoint_names: Iterable[str] | None = None,
        point_labels: PointLabelMode = "none",
        label: str | None = None,
        palette: Palette | None = None,
    ) -> "Keypoints":
        """Build renderable keypoints from an LDF `KeypointAnnotation`.

        Reuses the annotation's ``keypoints`` directly; absent points (visibility
        ``0``) are hidden by the default ``visibility_threshold``.

        Args:
            annotation: The LDF keypoint annotation.
            edges: Keypoint-index pairs to connect with limbs; empty draws only
                joints.
            keypoint_names: Per-keypoint names (index order) for named labels.
            point_labels: How to label each joint (see the class ``point_labels``
                attribute).
            label: Class label used for the palette color.
            palette: Palette used to color the joints from ``label``.

        Returns:
            The equivalent `Keypoints`.

        Examples:
            The LDF ``(x, y, visibility)`` points are reused as-is; the skeleton
            ``edges`` and label are rendering state added on top:

            >>> from luxonis_ml.ldf import KeypointAnnotation
            >>> ann = KeypointAnnotation.model_validate(
            ...     {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 0)]}
            ... )
            >>> kp = Keypoints.from_ldf(ann, edges=[(0, 1)], label="pose")
            >>> len(kp.keypoints)
            2
            >>> (
            ...     kp.keypoints[0][2],
            ...     kp.keypoints[1][2],
            ... )  # visibility preserved
            (2, 0)
            >>> (kp.edges, kp.label)
            ([(0, 1)], 'pose')

        """
        return cls(
            **annotation.model_dump(),
            edges=[(int(a), int(b)) for a, b in edges],
            keypoint_names=(
                list(keypoint_names) if keypoint_names is not None else None
            ),
            point_labels=point_labels,
            label=label,
            palette=palette,
        )

    def _point_label(self, index: int) -> str | None:
        """Return the label text for the keypoint at ``index``, or ``None``.

        Args:
            index: 0-based keypoint index.

        Returns:
            The text to draw beside the joint, or ``None`` when labels are off.

        """
        mode = self.point_labels
        if mode == "none":
            return None
        names = self.keypoint_names
        name = (
            names[index] if names is not None and index < len(names) else None
        )
        if mode == "numbers":
            return str(index)
        if mode == "names":
            return name if name is not None else str(index)
        return f"{index}:{name}" if name is not None else str(index)

    def _resolve(
        self, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve normalized points to pixel ``xy`` and a visibility vector.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        Returns:
            A ``(xy, visibility)`` pair: an ``(K, 2)`` float array and a length-``K``
            visibility array.

        """
        arr = np.asarray(self.keypoints, dtype=float).reshape(-1, 3)
        xy = arr[:, :2].copy()
        xy[:, 0] *= width
        xy[:, 1] *= height
        return xy, arr[:, 2].copy()

    def extent(self) -> Rect | None:
        """Return ``None``: normalized keypoints have no pixel extent until render.

        Returns:
            Always ``None``.

        """
        return None

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Draw limbs then joints (and optional names) onto the canvas.

        The joints the skeleton places but the data does not come first, so the
        real skeleton wins wherever the two overlap.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved instance color.

        """
        canvas = ctx.canvas
        xy, vis = self._resolve(canvas.width, canvas.height)
        visible = vis > self.visibility_threshold

        self._draw_absent(ctx, xy, visible, style, color)
        colors = self._joint_colors(color, len(xy))
        self._draw_limbs(ctx, xy, visible, colors, style)
        radius = style.keypoint_radius
        for i in range(len(xy)):
            if not visible[i]:
                continue
            center = (float(xy[i, 0]), float(xy[i, 1]))
            joint = colors[i]
            # COCO visibility: 2 (visible) is a white-outlined dot; anything
            # less (1, labeled but occluded) is a diamond, so a glance tells
            # them apart by shape alone.
            if float(vis[i]) < 2:
                self._draw_diamond(canvas, center, radius, joint, style)
            else:
                canvas.circle(
                    center,
                    radius,
                    fill=joint,
                    stroke=_WHITE,
                    stroke_width=style.keypoint_outline_width,
                )
            text = self._point_label(i)
            if text is not None:
                canvas.markup(
                    (
                        center[0] + radius + 3.0,
                        center[1] + style.font_size * 0.35,
                    ),
                    text,
                    size=style.font_size * 0.72,
                    color=_WHITE,
                    weight=style.font_weight,
                )

    def draw_label(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Emit the hover region for the joints, if this set carries a tooltip.

        Keypoints have no label chip; this override exists only so a tooltip-
        bearing set participates in hover hit-testing. The region is the bounds
        of the visible joints, padded by the joint radius, in display pixels.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved instance color.

        """
        if self.tooltip is None:
            return
        region = self._hit_region(ctx.canvas.width, ctx.canvas.height, style)
        if region is not None:
            ctx.emit_hit(region, self.tooltip, self.source)

    def region_at(self, width: int, height: int) -> Rect | None:
        """Return the bounds of the visible joints, in canvas pixels.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        Returns:
            The bounding `Rect` of the joints above ``visibility_threshold``, or
            ``None`` when none of them are visible.

        """
        xy, vis = self._resolve(width, height)
        points = xy[vis > self.visibility_threshold]
        if len(points) == 0:
            return None
        return Rect(
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )

    def _hit_region(
        self, width: int, height: int, style: Style
    ) -> Rect | None:
        """Return the padded bounds of the visible joints, or ``None`` if none."""
        region = self.region_at(width, height)
        if region is None:
            return None
        pad = (
            style.keypoint_radius * _DIAMOND_SCALE
            + style.keypoint_outline_width
        )
        return Rect(
            region.left - pad,
            region.top - pad,
            region.right + pad,
            region.bottom + pad,
        )

    def _draw_diamond(
        self,
        canvas: "Canvas",
        center: XY,
        radius: float,
        color: Color,
        style: Style,
    ) -> None:
        """Draw a filled diamond (an occluded joint) at ``center``.

        Stacks three cues against a visible joint's white-outlined dot — a
        diamond shape, a dimmed fill, and a dark outline — so it reads as
        clearly recessed at a glance.
        """
        r = radius * _DIAMOND_SCALE
        cx, cy = center
        canvas.polygon(
            [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
            fill=color.with_alpha(_OCCLUDED_ALPHA),
            stroke=_DARK,
            stroke_width=style.keypoint_outline_width,
        )

    def _joint_colors(self, color: Color, count: int) -> list[Color]:
        """Per-joint colors: ``point_colors`` where set, else the instance color."""
        if not self.point_colors:
            return [color] * count
        resolved: list[Color] = []
        for i in range(count):
            override = (
                self.point_colors[i] if i < len(self.point_colors) else None
            )
            resolved.append(
                Color.parse(override) if override is not None else color
            )
        return resolved

    def _adjacency(self) -> dict[int, set[int]]:
        """Neighbor sets per keypoint index, over the in-range ``edges``."""
        count = len(self.keypoints)
        adjacency: dict[int, set[int]] = {}
        for a, b in self.edges:
            if 0 <= a < count and 0 <= b < count:
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
        return adjacency

    def _absent_positions(
        self, xy: np.ndarray, visible: np.ndarray
    ) -> dict[int, XY]:
        """Place the absent joints the skeleton can place, in canvas pixels.

        Two shapes of gap have an answer that follows from the skeleton rather
        than from a pose prior, and only those are placed:

        - A joint with exactly two present neighbors is a gap *in* a chain, so
          it goes at their midpoint — unless those two are joined to each other
          as well, which makes the three a triangle rather than a chain and puts
          the midpoint right on top of the limb already drawn between them.
        - A joint with one present neighbor that is itself a pass-through of the
          chain (exactly one other present neighbor) hangs off the *end* of one,
          so the chain continues by one more limb of the same length.

        Everything else is left out: a joint hanging off a fork has no implied
        direction, and one whose neighbors are absent too would only stack a
        guess on a guess. Those stay hidden, and the joint count is what reports
        them.
        """
        if not self.edges:
            return {}
        adjacency = self._adjacency()
        positions: dict[int, XY] = {}
        for index in range(len(xy)):
            if visible[index]:
                continue
            near = sorted(n for n in adjacency.get(index, ()) if visible[n])
            if len(near) == 2:
                a, b = near
                if b not in adjacency.get(a, ()):
                    positions[index] = (
                        float(xy[a, 0] + xy[b, 0]) / 2.0,
                        float(xy[a, 1] + xy[b, 1]) / 2.0,
                    )
            elif len(near) == 1:
                anchor = near[0]
                back = sorted(
                    n
                    for n in adjacency.get(anchor, ())
                    if n != index and visible[n]
                )
                if len(back) == 1:
                    positions[index] = (
                        float(2.0 * xy[anchor, 0] - xy[back[0], 0]),
                        float(2.0 * xy[anchor, 1] - xy[back[0], 1]),
                    )
        return positions

    def _draw_absent(
        self,
        ctx: RenderContext,
        xy: np.ndarray,
        visible: np.ndarray,
        style: Style,
        color: Color,
    ) -> None:
        """Draw the unplaced joints as crosses on dashed limbs."""
        positions = self._absent_positions(xy, visible)
        if not positions:
            return
        canvas = ctx.canvas
        adjacency = self._adjacency()
        radius = style.keypoint_radius
        gap = radius * _ABSENT_GAP
        limb = color.with_alpha(_ABSENT_LIMB_ALPHA)
        mark = color.with_alpha(_ABSENT_MARK_ALPHA)
        for index, point in positions.items():
            for neighbor in sorted(adjacency.get(index, ())):
                if not visible[neighbor]:
                    continue
                start: XY = (float(xy[neighbor, 0]), float(xy[neighbor, 1]))
                end = _stop_short(start, point, gap)
                if end is not None:
                    canvas.polygon(
                        [start, end],
                        stroke=limb,
                        stroke_width=style.stroke_width,
                        dash=_ABSENT_DASH,
                        closed=False,
                    )
            arm = radius * _ABSENT_ARM
            x, y = point
            width = style.keypoint_outline_width * 1.5
            canvas.line((x - arm, y - arm), (x + arm, y + arm), mark, width)
            canvas.line((x - arm, y + arm), (x + arm, y - arm), mark, width)

    def _draw_limbs(
        self,
        ctx: RenderContext,
        xy: np.ndarray,
        visible: np.ndarray,
        colors: list[Color],
        style: Style,
    ) -> None:
        """Draw skeleton edges between visible endpoints.

        A limb whose endpoints share a color is a solid line; one spanning two
        different joint colors is drawn as a linear-gradient stroke between them.

        Args:
            ctx: The current render context.
            xy: Resolved ``(K, 2)`` pixel coordinates.
            visible: Boolean visibility mask over the points.
            colors: Per-joint colors (index-aligned to ``xy``).
            style: The resolved style.

        """
        if not self.edges:
            return
        for a, b in self.edges:
            if a < len(xy) and b < len(xy) and visible[a] and visible[b]:
                p1: XY = (float(xy[a, 0]), float(xy[a, 1]))
                p2: XY = (float(xy[b, 0]), float(xy[b, 1]))
                ca, cb = colors[a].with_alpha(0.85), colors[b].with_alpha(0.85)
                if ca == cb:
                    ctx.canvas.line(p1, p2, ca, width=style.stroke_width)
                else:
                    ctx.canvas.gradient_line(
                        p1, p2, ca, cb, width=style.stroke_width
                    )
