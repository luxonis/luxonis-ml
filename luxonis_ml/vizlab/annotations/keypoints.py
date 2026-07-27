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
from typing import TYPE_CHECKING, Literal

import numpy as np

from luxonis_ml.ldf import KeypointAnnotation
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import XY, Rect
from luxonis_ml.vizlab.style import Palette, Style

from .base import Annotation, RenderContext

if TYPE_CHECKING:
    from luxonis_ml.vizlab.canvas import Canvas

_WHITE = Color(255, 255, 255)
#: Dark outline for occluded joints (vs the white outline of visible ones).
_DARK = Color(25, 25, 25)


#: A diamond (occluded joint) is scaled up so it reads as visually heavy as the
#: circle (visible joint) of the same nominal radius.
_DIAMOND_SCALE = 1.3
#: Fill opacity of an occluded joint, so it also reads dimmer than a visible one.
_OCCLUDED_ALPHA = 0.55

PointLabelMode = Literal["none", "numbers", "names", "full"]
"""How to label each keypoint: nothing, its index, its name, or ``index:name``."""


class Keypoints(KeypointAnnotation, Annotation):
    """A set of keypoints, optionally wired into a skeleton.

    Reuses the normalized ``(x, y, visibility)`` ``keypoints`` list of
    `KeypointAnnotation`.

    .. image:: TODO-HOST/masks_keypoints.png
       :alt: Keypoints alongside instance, polygon, and semantic masks.

    Attributes:
        edges: Pairs of keypoint indices to connect with a limb; empty draws
            joints only.
        keypoint_names: Optional per-keypoint names (index order), used by the
            ``"names"``/``"full"`` point-label modes.
        visibility_threshold: Points whose visibility is ``<=`` this are hidden.
            With COCO visibility, a joint marked visible (``2``) is a bright dot
            with a white outline; one marked occluded (``1``) is a dimmer diamond
            with a dark outline, so it clearly recedes. A ``0..1`` confidence
            column instead scales the dot.
        point_labels: How to label each joint — ``"none"`` (default),
            ``"numbers"`` (its index), ``"names"`` (its keypoint name), or
            ``"full"`` (``index:name``). ``"names"``/``"full"`` fall back to the
            index when no name is available.
        point_colors: Optional per-joint color overrides, index-aligned to
            ``keypoints`` (a ``None`` entry, or a short list, falls back to the
            instance color). A limb between two differently colored joints is
            drawn as a gradient from one color to the other. Useful for grading
            individual joints — see `luxonis_ml.vizlab.compare`.

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

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved instance color.

        """
        canvas = ctx.canvas
        xy, vis = self._resolve(canvas.width, canvas.height)
        visible = vis > self.visibility_threshold

        # Treat a 0..1 visibility column as confidence and scale joints by it.
        seen = vis[visible]
        confidence_like = len(seen) > 0 and bool(
            np.all((seen > 0) & (seen <= 1.0))
        )

        colors = self._joint_colors(color, len(xy))
        self._draw_limbs(ctx, xy, visible, colors, style)
        for i in range(len(xy)):
            if not visible[i]:
                continue
            v = float(vis[i])
            radius = style.keypoint_radius
            if confidence_like:
                radius *= 0.55 + 0.45 * v
            center = (float(xy[i, 0]), float(xy[i, 1]))
            joint = colors[i]
            # COCO visibility: 2 (visible) is a filled dot; 1 (labeled but
            # occluded) is a filled diamond. Both keep the same fill and white
            # outline, so a glance tells them apart by shape alone. Confidence-
            # style visibility is always a dot.
            if not confidence_like and v < 2:
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
                canvas.text(
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
            ctx.emit_hit(region, self.tooltip)

    def _hit_region(
        self, width: int, height: int, style: Style
    ) -> Rect | None:
        """Return the padded bounds of the visible joints, or ``None`` if none."""
        xy, vis = self._resolve(width, height)
        points = xy[vis > self.visibility_threshold]
        if len(points) == 0:
            return None
        pad = (
            style.keypoint_radius * _DIAMOND_SCALE
            + style.keypoint_outline_width
        )
        return Rect(
            float(points[:, 0].min()) - pad,
            float(points[:, 1].min()) - pad,
            float(points[:, 0].max()) + pad,
            float(points[:, 1].max()) + pad,
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
