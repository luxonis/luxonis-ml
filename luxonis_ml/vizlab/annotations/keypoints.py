"""Keypoints and skeletons.

`Keypoints` subclasses the Luxonis Data Format
`KeypointAnnotation`, reusing its normalized
``(x, y, visibility)`` keypoint list (COCO visibility ``0``/``1``/``2``) and
adding rendering: joints, and — given a `Skeleton` — the limbs connecting
them. Build a `Skeleton` from a dataset's definition with
`Skeleton.from_ldf`.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from luxonis_ml.ldf import KeypointAnnotation
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import XY, Rect
from luxonis_ml.vizlab.style import Palette, Style

from .base import Annotation, RenderContext

_WHITE = Color(255, 255, 255)

PointLabelMode = Literal["none", "numbers", "names", "full"]
"""How to label each keypoint: nothing, its index, its name, or ``index:name``."""


@dataclass(frozen=True)
class Skeleton:
    """A set of limb connections (and optional joint names) over keypoint indices.

    Attributes:
        edges: Pairs of keypoint indices to connect with a limb.
        names: Optional per-keypoint names, ``K`` long, used for point labels.

    """

    edges: tuple[tuple[int, int], ...]
    names: tuple[str, ...] | None = None

    @classmethod
    def from_ldf(
        cls,
        labels: Iterable[str] | None,
        edges: Iterable[tuple[int, int]],
    ) -> "Skeleton":
        """Build a skeleton from an LDF dataset's skeleton definition.

        LDF stores skeletons in dataset metadata as ``(labels, edges)`` (see
        ``LuxonisDataset.get_skeletons``), not on the annotations themselves.

        Args:
            labels: Per-keypoint names in index order, or ``None``.
            edges: 0-based keypoint index pairs to connect.

        Returns:
            The equivalent `Skeleton`.

        """
        return cls(
            edges=tuple((int(a), int(b)) for a, b in edges),
            names=tuple(labels) if labels else None,
        )


class Keypoints(KeypointAnnotation, Annotation):
    """A set of keypoints, optionally wired into a skeleton.

    Reuses the normalized ``(x, y, visibility)`` ``keypoints`` list of
    `KeypointAnnotation`.

    Attributes:
        skeleton: Limb/name definition; without it only joints are drawn.
        visibility_threshold: Points whose visibility is ``<=`` this are hidden.
        point_labels: How to label each joint — ``"none"`` (default),
            ``"numbers"`` (its index), ``"names"`` (its skeleton name), or
            ``"full"`` (``index:name``). ``"names"``/``"full"`` fall back to the
            index when no name is available.

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

    skeleton: Skeleton | None = None
    visibility_threshold: float = 0.0
    point_labels: PointLabelMode = "none"

    @classmethod
    def from_ldf(
        cls,
        annotation: KeypointAnnotation,
        *,
        skeleton: Skeleton | None = None,
        point_labels: PointLabelMode = "none",
        label: str | None = None,
        palette: Palette | None = None,
    ) -> "Keypoints":
        """Build renderable keypoints from an LDF `KeypointAnnotation`.

        Reuses the annotation's ``keypoints`` directly; absent points (visibility
        ``0``) are hidden by the default ``visibility_threshold``.

        Args:
            annotation: The LDF keypoint annotation.
            skeleton: Limb/name definition; without it only joints are drawn.
            point_labels: How to label each joint (see the class ``point_labels``
                attribute).
            label: Class label used for the palette color.
            palette: Palette used to color the joints from ``label``.

        Returns:
            The equivalent `Keypoints`.

        """
        return cls(
            **annotation.model_dump(),
            skeleton=skeleton,
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
        names = self.skeleton.names if self.skeleton is not None else None
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

        self._draw_limbs(ctx, xy, visible, color, style)
        for i in range(len(xy)):
            if not visible[i]:
                continue
            radius = style.keypoint_radius
            if confidence_like:
                radius *= 0.55 + 0.45 * float(vis[i])
            center = (float(xy[i, 0]), float(xy[i, 1]))
            canvas.circle(
                center,
                radius,
                fill=color,
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

    def _draw_limbs(
        self,
        ctx: RenderContext,
        xy: np.ndarray,
        visible: np.ndarray,
        color: Color,
        style: Style,
    ) -> None:
        """Draw skeleton edges between visible endpoints.

        Args:
            ctx: The current render context.
            xy: Resolved ``(K, 2)`` pixel coordinates.
            visible: Boolean visibility mask over the points.
            color: Limb color.
            style: The resolved style.

        """
        if self.skeleton is None:
            return
        limb = color.with_alpha(0.85)
        for a, b in self.skeleton.edges:
            if a < len(xy) and b < len(xy) and visible[a] and visible[b]:
                p1: XY = (float(xy[a, 0]), float(xy[a, 1]))
                p2: XY = (float(xy[b, 0]), float(xy[b, 1]))
                ctx.canvas.line(p1, p2, limb, width=style.stroke_width)
