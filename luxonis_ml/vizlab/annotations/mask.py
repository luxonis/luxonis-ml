"""Instance and semantic masks.

`Mask` subclasses the Luxonis Data Format
`InstanceSegmentationAnnotation`, so it reuses that model's
storage and RLE decoding (``to_numpy()``) and adds only the rendering: a
translucent fill plus an optional contour. `SemanticMask` renders a whole
dense label map, coloring each class id from the palette — a purely visual
construct with no single LDF annotation counterpart. Contours use OpenCV when it
is installed and are skipped otherwise (the fill still draws).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from luxonis_ml.ldf import InstanceSegmentationAnnotation
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.style import Palette, Style

from .base import Annotation, RenderContext
from .chip import place_label

if TYPE_CHECKING:
    from luxonis_ml.ldf import SegmentationAnnotation

_WHITE = Color(255, 255, 255)


def _mask_contours(mask_bool: np.ndarray) -> list[np.ndarray]:
    """Return external contours of a boolean mask, or ``[]`` without OpenCV.

    Args:
        mask_bool: An ``(H, W)`` boolean array.

    Returns:
        A list of ``(N, 2)`` float contour point arrays; empty if OpenCV is not
        installed.

    """
    try:
        import cv2
    except ImportError:
        return []
    m = np.ascontiguousarray(mask_bool.astype(np.uint8))
    contours, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [c.reshape(-1, 2).astype(float) for c in contours if len(c) >= 2]


def _nonzero_bounds(binary: np.ndarray) -> Rect | None:
    """Return the bounding rect of a binary mask's set pixels, or ``None`` if empty."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    return Rect(
        float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    )


class Mask(InstanceSegmentationAnnotation, Annotation):
    """A single instance mask: a translucent fill with an optional contour.

    Reuses `InstanceSegmentationAnnotation`, so the mask is
    supplied and stored exactly as in LDF (a binary array, polygon ``points``, or
    COCO RLE ``counts`` + ``height``/``width``) and decoded to a dense ``(H, W)``
    array with the inherited ``to_numpy()`` — no separate decoding here.

    Attributes:
        fill_alpha: Fill opacity override; falls back to ``style.mask_alpha``.
        contour: Whether to stroke the mask outline.

    See `Annotation` for the shared
    ``label``, ``score``, ``payload``, ``color``, ``style``, and ``palette`` fields.

    Examples:
        >>> import numpy as np
        >>> Mask(mask=np.array([[0, 1], [1, 0]], np.uint8)).to_numpy().tolist()
        [[0, 1], [1, 0]]

    """

    fill_alpha: float | None = None
    contour: bool = True

    @classmethod
    def from_ldf(
        cls,
        annotation: "SegmentationAnnotation",
        *,
        label: str | None = None,
        score: float | None = None,
        palette: Palette | None = None,
    ) -> "Mask":
        """Build a renderable instance mask from an LDF segmentation annotation.

        Reuses the annotation's RLE storage directly; only rendering state is added.

        Args:
            annotation: An LDF instance- or single-class segmentation annotation.
            label: Class label for the chip and palette color.
            score: Optional confidence shown on the chip.
            palette: Palette used to color the mask from ``label``.

        Returns:
            The equivalent `Mask`.

        """
        return cls(
            **annotation.model_dump(),
            label=label,
            score=score,
            palette=palette,
        )

    def extent(self) -> Rect | None:
        """Return the mask's pixel bounds, or ``None`` when empty."""
        return _nonzero_bounds(self.to_numpy())

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Draw the fill, optional contour, and label chip onto the canvas.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved fill color.

        """
        alpha = (
            style.mask_alpha if self.fill_alpha is None else self.fill_alpha
        )
        binary = self.to_numpy()
        canvas = ctx.canvas
        canvas.overlay_mask(binary, color, alpha=alpha)
        if self.contour:
            for ring in _mask_contours(binary > 0):
                canvas.polygon(
                    [(float(x), float(y)) for x, y in ring],
                    stroke=color,
                    stroke_width=style.stroke_width,
                    dash=style.dash,
                )
        region = _nonzero_bounds(binary)
        if region is not None:
            place_label(
                ctx, region, self.label, self.score, self.payload, color, style
            )


class SemanticMask(Annotation):
    """A dense label map, colored per class id from the palette.

    A rendering-only construct: LDF has no single semantic-label-map annotation
    (semantic segmentation is stored per class as
    `SegmentationAnnotation`), so `from_ldf` combines
    those into the ``(H, W)`` id map this draws.

    Attributes:
        labels: An ``(H, W)`` integer array of class ids.
        names: Optional id-to-name mapping (or a list indexed by id) used both for
            stable palette colors and future legends.
        ignore_index: Class id(s) treated as background and left undrawn.
        color_map: Optional explicit id-to-color mapping; overrides the palette.
        fill_alpha: Fill opacity override; falls back to ``style.mask_alpha``.

    See `Annotation` for the shared
    ``style`` and ``palette`` fields (``label``/``color`` do not apply — colors are
    per class id).

    """

    labels: np.ndarray | None = None
    names: dict[int, str] | list[str] | None = None
    ignore_index: int | list[int] = 0
    color_map: dict[int, ColorLike] | None = None
    fill_alpha: float | None = None

    @classmethod
    def from_ldf(
        cls,
        segmentations: Sequence[tuple[str | None, "SegmentationAnnotation"]],
        *,
        palette: Palette | None = None,
    ) -> "SemanticMask":
        """Aggregate LDF semantic-segmentation annotations into one label map.

        Each ``(class_name, annotation)`` pair is decoded to a dense ``(H, W)``
        mask and painted into a shared id map (id ``0`` is background, ids
        ``1..K`` are assigned per class in first-seen order, first mask wins on
        overlap), mirroring ``SegmentationAnnotation.combine_to_numpy``.

        Args:
            segmentations: Pairs of class name and LDF segmentation annotation.
            palette: Palette used to color each class id by name.

        Returns:
            A `SemanticMask` over the combined id map, or an empty one
            when ``segmentations`` is empty.

        """
        if not segmentations:
            return cls(palette=palette)
        masks = [(name, ann.to_numpy()) for name, ann in segmentations]
        height, width = masks[0][1].shape
        id_map = np.zeros((height, width), dtype=np.int32)
        names: dict[int, str] = {}
        ids: dict[str, int] = {}
        assigned = np.zeros((height, width), dtype=bool)
        for name, mask in masks:
            key = name if name is not None else "unknown"
            class_id = ids.setdefault(key, len(ids) + 1)
            names[class_id] = key
            region = (mask > 0) & (~assigned)
            id_map[region] = class_id
            assigned |= region
        return cls(labels=id_map, names=names, palette=palette)

    def _name(self, class_id: int) -> str:
        """Return the palette key for a class id (its name, or the id as a string)."""
        names = self.names
        if isinstance(names, dict):
            return names.get(class_id, str(class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return names[class_id]
        return str(class_id)

    def _color(self, class_id: int, palette: Palette) -> Color:
        """Resolve the color for a class id from ``color_map`` or the palette."""
        if self.color_map is not None and class_id in self.color_map:
            return Color.parse(self.color_map[class_id])
        return palette.color_for(self._name(class_id))

    def resolve_color(self, ctx: RenderContext) -> Color:
        """Semantic masks color per class id, so no single color is resolved.

        Args:
            ctx: The current render context (unused).

        Returns:
            An unused placeholder color (white); the real colors come from
            `_color` per class id.

        """
        return _WHITE

    def extent(self) -> Rect | None:
        """Semantic masks cover the image and have no local extent.

        Returns:
            Always ``None``.

        """
        return None

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Overlay one translucent color per class id.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: Unused (colors are per class id).

        """
        if self.labels is None:
            return
        canvas = ctx.canvas
        palette = self.resolved_palette(ctx)
        alpha = (
            style.mask_alpha if self.fill_alpha is None else self.fill_alpha
        )
        ignore = (
            {self.ignore_index}
            if isinstance(self.ignore_index, int)
            else set(self.ignore_index)
        )

        labels = np.asarray(self.labels)
        for class_id in np.unique(labels):
            cid = int(class_id)
            if cid in ignore:
                continue
            region = labels == class_id
            class_color = self._color(cid, palette)
            canvas.overlay_mask(region, class_color, alpha=alpha)
            if style.stroke_width > 0:
                for ring in _mask_contours(region):
                    canvas.polygon(
                        [(float(x), float(y)) for x, y in ring],
                        stroke=class_color,
                        stroke_width=style.stroke_width,
                    )
