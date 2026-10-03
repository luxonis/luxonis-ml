"""Keypoints with an optional skeleton of limb edges.

`Keypoints` subclasses the Luxonis Data Format
`KeypointAnnotation`, reusing its normalized
``(x, y, visibility)`` keypoints keyed by name (COCO visibility ``0``/``1``/``2``)
and its ``edges``, and adding rendering: joints, and the limbs connecting them.
A dataset keeps the names and edges of each task as a `KeypointMetadata` (see
``LuxonisDataset.get_keypoint_metadata``); `Keypoints.from_ldf` applies it.
"""

from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from luxonis_ml.ldf import KeypointAnnotation, KeypointMetadata
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import XY, Rect
from luxonis_ml.vizlab.style import Palette, Style

from .base import Annotation, RenderContext

if TYPE_CHECKING:
    from luxonis_ml.vizlab.render.canvas import Canvas

_WHITE = Color(255, 255, 255)
#: Dark outline that keeps a joint name readable over the image.
_HALO = Color(16, 20, 30, 200)
#: Joint radius per pixel of pose height, and its floor: a far-away pose
#: gets small dots, a close one the full style radius.
_RADIUS_PER_HEIGHT = 0.025
_MIN_RADIUS = 1.5
#: Below this radius a dot has no room for the white ring.
_RING_MIN_RADIUS = 3.0
#: Poses shorter than this get no joint names: they would cover the pose.
_NAMED_MIN_HEIGHT = 120.0
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
"""How to label each keypoint, with nothing, its index, its name, or both.

``"none"`` draws nothing, ``"numbers"`` the index, ``"names"`` the name, and
``"full"`` draws ``index:name``.
"""


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


def _pose_height(xy: np.ndarray, visible: np.ndarray) -> float:
    """Return the pixel height of the visible joints, or of all when none is.

    Examples:
        >>> xy = np.array([[0.0, 10.0], [5.0, 70.0], [9.0, 400.0]])
        >>> _pose_height(xy, np.array([True, True, False]))
        60.0
        >>> _pose_height(xy, np.zeros(3, bool))
        390.0

    """
    points = xy[visible] if visible.any() else xy
    return float(np.ptp(points[:, 1])) if len(points) else 0.0


def _joint_radius(style: Style, height: float) -> float:
    """Return the joint radius for a pose ``height`` pixels tall.

    The radius grows with the pose from `_MIN_RADIUS` up to the style's
    ``keypoint_radius``, so a far-away pose is not a cluster of large dots.

    Examples:
        >>> from luxonis_ml.vizlab.style import DEFAULT_STYLE
        >>> _joint_radius(DEFAULT_STYLE, 20.0)
        1.5
        >>> _joint_radius(
        ...     DEFAULT_STYLE, 2000.0
        ... ) == DEFAULT_STYLE.keypoint_radius
        True

    """
    return min(
        style.keypoint_radius, max(_MIN_RADIUS, _RADIUS_PER_HEIGHT * height)
    )


class Keypoints(KeypointAnnotation, Annotation):
    """A set of keypoints, optionally wired into a skeleton.

    Reuses the normalized ``(x, y, visibility)`` ``keypoints`` of
    `KeypointAnnotation`, keyed by name, and its ``edges``.

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
        visibility_threshold: Points whose visibility is ``<=`` this are hidden.
            With COCO visibility, a joint marked visible (``2``) is a bright dot
            with a white outline; one marked occluded (``1``) is a dimmer diamond
            with a dark outline, so it clearly recedes.
        point_labels: How to label each joint — ``"none"`` (default),
            ``"numbers"`` (its index), ``"names"`` (its keypoint name), or
            ``"full"`` (``index:name``). ``"names"``/``"full"`` fall back to the
            index for keypoints keyed by position.
        point_colors: Optional per-joint color overrides, index-aligned to
            ``keypoints`` (a ``None`` entry, or a short list, falls back to the
            instance color). A limb between two differently colored joints is
            drawn as a gradient from one color to the other. Useful for grading
            individual joints — see `luxonis_ml.vizlab.comparison.render`. The
            marks on absent joints keep the instance color either way: a joint
            neither side placed has no verdict worth painting.

    See `KeypointAnnotation` for ``keypoints`` and ``edges``, and
    `Annotation` for the shared ``label``, ``color``, ``style``, and
    ``palette`` fields.

    Examples:
        >>> kp = Keypoints(keypoints=[(0.1, 0.2, 2), (0.3, 0.4, 0)])
        >>> xy, vis = kp._resolve(200, 100)
        >>> xy.tolist()
        [[20.0, 20.0], [60.0, 40.0]]
        >>> vis.tolist()
        [2.0, 0.0]

    """

    LAYER: ClassVar[str] = "keypoint"

    visibility_threshold: float = 0.0
    point_labels: PointLabelMode = "none"
    point_colors: list[ColorLike | None] | None = None

    @classmethod
    def from_ldf(
        cls,
        annotation: KeypointAnnotation,
        *,
        metadata: KeypointMetadata | None = None,
        point_labels: PointLabelMode = "none",
        label: str | None = None,
        palette: Palette | None = None,
    ) -> "Keypoints":
        """Build renderable keypoints from an LDF `KeypointAnnotation`.

        Absent points (visibility ``0``) are hidden by the default
        ``visibility_threshold``.

        Args:
            annotation: The LDF keypoint annotation.
            metadata: The keypoint metadata of the annotation's task. Its
                names key the keypoints, and its edges become the limbs when
                the annotation declares none. Stored edges that point past
                the keypoints are dropped, as an older dataset can hold them.
            point_labels: How to label each joint (see the class ``point_labels``
                attribute).
            label: Class label used for the palette color.
            palette: Palette used to color the joints from ``label``.

        Returns:
            The equivalent `Keypoints`.

        Examples:
            A loader row is keyed by position. The task's metadata names the
            keypoints and supplies the limbs:

            >>> from luxonis_ml.ldf import KeypointAnnotation, KeypointMetadata
            >>> ann = KeypointAnnotation.model_validate(
            ...     {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 0)]}
            ... )
            >>> metadata = KeypointMetadata(
            ...     labels=["nose", "eye"], edges=[(0, 1)]
            ... )
            >>> kp = Keypoints.from_ldf(ann, metadata=metadata, label="pose")
            >>> list(kp.keypoints)
            ['nose', 'eye']
            >>> (kp.edges, kp.label)
            ([(0, 1)], 'pose')

        """
        keypoints = dict(annotation.keypoints)
        edges = annotation.edges
        if metadata is not None:
            keypoints = metadata.align(keypoints)
            if metadata.has_names:
                keypoints = dict(
                    zip(metadata.labels, keypoints.values(), strict=True)
                )
            count = len(keypoints)
            edges = edges or [
                (a, b) for a, b in metadata.edges if a < count and b < count
            ]
        return cls(
            keypoints=keypoints,
            edges=edges,
            point_labels=point_labels,
            label=label,
            palette=palette,
        )

    def _point_label(self, index: int, names: list[str]) -> str | None:
        """Return the label text for the keypoint at ``index``, or ``None``.

        Args:
            index: 0-based keypoint index.
            names: The keypoint names, empty when the keys are positions.

        Returns:
            The text to draw beside the joint, or ``None`` when labels are off.

        """
        mode = self.point_labels
        if mode == "none":
            return None
        if mode == "numbers" or not names:
            return str(index)
        if mode == "names":
            return names[index]
        return f"{index}:{names[index]}"

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
        arr = self.to_numpy().reshape(-1, 3)
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
        """Draw limbs then joints onto the canvas.

        The joints the skeleton places but the data does not come first, so the
        real skeleton wins wherever the two overlap. The joint names are drawn
        later, in `draw_label`.

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
        radius = _joint_radius(style, _pose_height(xy, visible))
        ring = (
            style.keypoint_outline_width if radius >= _RING_MIN_RADIUS else 0.0
        )
        for i in range(len(xy)):
            if not visible[i]:
                continue
            center = (float(xy[i, 0]), float(xy[i, 1]))
            joint = colors[i]
            # COCO visibility: 2 (visible) is a white-outlined dot; anything
            # less (1, labeled but occluded) is a diamond, so a glance tells
            # them apart by shape alone.
            if float(vis[i]) < 2:
                self._draw_diamond(canvas, center, radius, joint, ring)
            else:
                canvas.circle(
                    center,
                    radius,
                    fill=joint,
                    stroke=_WHITE if ring else None,
                    stroke_width=ring,
                )

    def _draw_names(
        self,
        canvas: "Canvas",
        xy: np.ndarray,
        visible: np.ndarray,
        radius: float,
        style: Style,
    ) -> None:
        """Name the visible joints on the side away from the body.

        Outer joints are named first, as they have room; a name that would
        overlap one already drawn is left out.

        Args:
            canvas: The canvas to draw on.
            xy: ``(K, 2)`` joint positions in pixels.
            visible: Length-``K`` mask of the joints to name.
            radius: The drawn joint radius.
            style: The resolved style.

        """
        shown = np.flatnonzero(visible)
        if len(shown) == 0:
            return
        declared = self.declared_metadata()
        names = declared.labels if declared is not None else []
        mid_x, mid_y = np.median(xy[shown], axis=0)
        order = sorted(
            shown,
            key=lambda i: -float(np.hypot(xy[i, 0] - mid_x, xy[i, 1] - mid_y)),
        )
        size = style.font_size * 0.72
        taken: list[Rect] = []
        for i in order:
            text = self._point_label(int(i), names)
            if text is None:
                continue
            metrics = canvas.measure_markup(
                text, size, weight=style.font_weight
            )
            x, y = float(xy[i, 0]), float(xy[i, 1])
            gap = radius + 3.0
            left = x - gap - metrics.width if x < mid_x else x + gap
            baseline = y + metrics.ascent / 2
            rect = Rect(
                left - 1.0,
                baseline - metrics.ascent - 1.0,
                left + metrics.width + 1.0,
                baseline + 1.0,
            )
            if any(rect.intersection(other) for other in taken):
                continue
            taken.append(rect)
            canvas.markup(
                (left, baseline),
                text,
                size=size,
                color=_WHITE,
                weight=style.font_weight,
                halo=_HALO,
            )

    def draw_label(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Draw the joint names, and emit the hover region for the joints.

        The names are drawn in the label pass, after every shape, so no limb
        or contour drawn later covers them. A pose shorter than
        ``_NAMED_MIN_HEIGHT`` pixels gets no names. The hover region is the
        bounds of the visible joints, padded by the joint radius, in display
        pixels; it is emitted only when this set carries a tooltip.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved instance color.

        """
        canvas = ctx.canvas
        if self.point_labels != "none" and ctx.environment.layers.labels:
            xy, vis = self._resolve(canvas.width, canvas.height)
            visible = vis > self.visibility_threshold
            height = _pose_height(xy, visible)
            if height >= _NAMED_MIN_HEIGHT:
                radius = _joint_radius(style, height)
                self._draw_names(canvas, xy, visible, radius, style)
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
        outline: float,
    ) -> None:
        """Draw a filled diamond (an occluded joint) at ``center``.

        Stacks three cues against a visible joint's white-outlined dot — a
        diamond shape, a dimmed fill, and a dark outline — so it reads as
        clearly recessed at a glance. A small joint has no ``outline``, like
        a small dot has no ring.
        """
        r = radius * _DIAMOND_SCALE
        cx, cy = center
        canvas.polygon(
            [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
            fill=color.with_alpha(_OCCLUDED_ALPHA),
            stroke=_DARK if outline else None,
            stroke_width=outline,
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
