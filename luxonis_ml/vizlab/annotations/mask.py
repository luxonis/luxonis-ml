"""Instance and semantic masks.

`Mask` subclasses the Luxonis Data Format
`InstanceSegmentationAnnotation`, so it reuses that model's
storage and RLE decoding (``to_numpy()``) and adds only the rendering: a
translucent fill plus an optional contour. `SemanticMask` renders a whole
dense label map, coloring each class id from the palette — a purely visual
construct with no single LDF annotation counterpart. Contours use OpenCV when it
is installed and are skipped otherwise (the fill still draws).
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import PrivateAttr

from luxonis_ml.ldf import InstanceSegmentationAnnotation
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.style import MaskOutline, Palette, Style

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
    # Cast and contiguity in one pass; a mask that is already contiguous uint8
    # (an instance mask straight from ``to_numpy``) is used without a copy.
    m = np.ascontiguousarray(mask_bool, dtype=np.uint8)
    contours, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rings: list[np.ndarray] = []
    for c in contours:
        if len(c) < 2:
            continue
        # Drop the sub-pixel staircase so the outline is cleaner once scaled up.
        c = cv2.approxPolyDP(c, 1.0, True)
        rings.append(c.reshape(-1, 2).astype(float))
    return rings


#: A vertex whose outline turns at least this much (≈50°) is kept sharp; gentler
#: bends are rounded. This keeps rectangular masks rectangular (their ~90°
#: corners survive) while still smoothing the facets of curved outlines.
_KEEP_CORNER_ANGLE = math.radians(50.0)


def _smooth_ring(points: np.ndarray, iterations: int) -> np.ndarray:
    """Round a closed polygon by corner-cutting, preserving sharp corners.

    A Chaikin-style pass, but a vertex is only cut where the outline bends
    gently; sharp corners (turn ``>= _KEEP_CORNER_ANGLE``) are left in place. So
    the pixel-staircase facets of a curve round out, while a rectangle keeps its
    corners instead of rounding into an ellipse. Cheap and OpenCV-free.

    Args:
        points: An ``(N, 2)`` array of closed-ring vertices.
        iterations: Number of corner-cutting passes.

    Returns:
        The smoothed ring (variable length; sharp corners keep one vertex,
        cut corners become two).

    """
    for _ in range(iterations):
        prev = np.roll(points, 1, axis=0)
        nxt = np.roll(points, -1, axis=0)
        incoming = points - prev
        outgoing = nxt - points
        # Turn angle at each vertex: 0 straight, pi doubling back.
        dot = (incoming * outgoing).sum(axis=1)
        norms = np.linalg.norm(incoming, axis=1) * np.linalg.norm(
            outgoing, axis=1
        )
        turn = np.arccos(np.clip(dot / (norms + 1e-9), -1.0, 1.0))
        cut_before = 0.75 * points + 0.25 * prev
        cut_after = 0.75 * points + 0.25 * nxt
        out: list[np.ndarray] = []
        for i in range(len(points)):
            if turn[i] >= _KEEP_CORNER_ANGLE:
                out.append(points[i])
            else:
                out.append(cut_before[i])
                out.append(cut_after[i])
        points = np.asarray(out, dtype=float)
    return points


def _contour_path(
    ring: np.ndarray, sx: float, sy: float, smoothing: int
) -> list[tuple[float, float]]:
    """Scale a contour ring to the canvas and optionally smooth it for drawing."""
    points = ring * np.array([sx, sy])
    if smoothing > 0 and len(points) >= 3:
        points = _smooth_ring(points, smoothing)
    return [(float(x), float(y)) for x, y in points]


def _contour_smoothing(sx: float, sy: float) -> int:
    """Chaikin passes to use for a contour scaled by ``(sx, sy)``.

    Only smooths when the mask is upscaled (where the pixel staircase becomes
    visible); downscaling already hides it, and native rendering is left as-is.
    """
    scale = max(sx, sy)
    if scale <= 1.15:
        return 0
    return 3 if scale >= 2.0 else 2


def _contour_transform(
    canvas: Canvas, source: np.ndarray, style: Style
) -> tuple[float, float, int]:
    """Scale factors and smoothing to trace ``source`` onto ``canvas``."""
    mask_h, mask_w = source.shape[:2]
    sx = canvas.width / mask_w
    sy = canvas.height / mask_h
    smoothing = (
        _contour_smoothing(sx, sy)
        if style.mask_outline is MaskOutline.SMOOTH
        else 0
    )
    return sx, sy, smoothing


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbor resize a mask to ``(height, width)``.

    Masks are stored at the source image's resolution, but the canvas may be a
    different size (e.g. the image was scaled for display). Nearest-neighbor
    keeps class ids / binary values intact and is dtype-agnostic (no OpenCV
    needed).

    Args:
        mask: An ``(H, W)`` array of any dtype.
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        The mask resampled to ``(height, width)`` (the original if it already
        matches).

    """
    src_h, src_w = mask.shape[:2]
    if (src_h, src_w) == (height, width):
        return mask
    ys = (np.arange(height) * (src_h / height)).astype(np.intp)
    xs = (np.arange(width) * (src_w / width)).astype(np.intp)
    return mask[ys[:, None], xs[None, :]]


def _nonzero_bounds(binary: np.ndarray) -> Rect | None:
    """Return the bounding rect of a binary mask's set pixels, or ``None`` if empty.

    Reduces along each axis (``any``) and reads the first/last set row and column,
    which avoids materializing the coordinate arrays of every set pixel that
    ``np.nonzero`` would build for a large mask.
    """
    rows = np.any(binary, axis=1)
    if not rows.any():
        return None
    cols = np.any(binary, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return Rect(float(x0), float(y0), float(x1), float(y1))


class Mask(InstanceSegmentationAnnotation, Annotation):
    """A single instance mask: a translucent fill with an optional contour.

    Reuses `InstanceSegmentationAnnotation`, so the mask is
    supplied and stored exactly as in LDF (a binary array, polygon ``points``, or
    COCO RLE ``counts`` + ``height``/``width``) and decoded to a dense ``(H, W)``
    array with the inherited ``to_numpy()`` — no separate decoding here.

    .. image:: TODO-HOST/masks_keypoints.png
       :alt: Instance, polygon, and semantic masks alongside keypoints.

    Attributes:
        fill_alpha: Fill opacity override; falls back to ``style.mask_alpha``.
        contour: Whether to stroke the mask outline.
        label_chip: Whether to draw the class label chip. The fill and contour
            still draw and keep the class color — this only hides the chip, e.g.
            when a box already labels the same class (see
            `luxonis_ml.vizlab.adapters.ldf.blend_records_to_annotations`).

    See `Annotation` for the shared
    ``label``, ``score``, ``payload``, ``color``, ``style``, and ``palette`` fields.

    Examples:
        >>> import numpy as np
        >>> Mask(mask=np.array([[0, 1], [1, 0]], np.uint8)).to_numpy().tolist()
        [[0, 1], [1, 0]]

    """

    LAYER: ClassVar[str] = "mask"

    fill_alpha: float | None = None
    contour: bool = True
    label_chip: bool = True

    #: Memoized dense mask. A single render decodes the RLE three times (fill,
    #: contour, and label passes); caching it turns that into one decode. The
    #: mask data is treated as immutable once the annotation is built.
    _dense_cache: np.ndarray | None = PrivateAttr(default=None)
    #: Memoized `extent`, boxed so that an empty mask's ``None`` is a computed
    #: answer rather than a cache miss. A layered export draws the scene once
    #: per layer and class, so without this the same reduction over the same
    #: pixels runs a dozen times over.
    _extent_cache: "tuple[Rect | None] | None" = PrivateAttr(default=None)

    def _dense(self) -> np.ndarray:
        """Return the decoded ``(H, W)`` mask, decoding once and caching it.

        The decode (pycocotools) yields a Fortran-ordered array; it is made
        C-contiguous here once so the per-pixel passes over it — the bounds
        reduction, the fill, the contour extraction — all run on cache-friendly
        row-major memory instead of paying a strided copy each time.
        """
        if self._dense_cache is None:
            self._dense_cache = np.ascontiguousarray(self.to_numpy())
        return self._dense_cache

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

        Examples:
            The LDF mask storage is reused directly; ``to_numpy()`` decodes it back
            to the same dense array:

            >>> import numpy as np
            >>> from luxonis_ml.ldf import InstanceSegmentationAnnotation
            >>> ann = InstanceSegmentationAnnotation.model_validate(
            ...     {"mask": np.eye(3, dtype=np.uint8)}
            ... )
            >>> mask = Mask.from_ldf(ann, label="cell")
            >>> mask.to_numpy().tolist()
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            >>> mask.label
            'cell'

        """
        return cls(
            **annotation.model_dump(),
            label=label,
            score=score,
            palette=palette,
        )

    def extent(self) -> Rect | None:
        """Return the mask's pixel bounds, or ``None`` when empty."""
        if self._extent_cache is None:
            self._extent_cache = (_nonzero_bounds(self._dense()),)
        return self._extent_cache[0]

    def draw_fill(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Overlay the translucent mask fill at native resolution (first pass).

        Args:
            ctx: The current render context (native resolution).
            style: The resolved style.
            color: The resolved fill color.

        """
        alpha = (
            style.mask_alpha if self.fill_alpha is None else self.fill_alpha
        )
        canvas = ctx.canvas
        binary = resize_mask(self._dense(), canvas.width, canvas.height)
        canvas.overlay_mask(binary, color, alpha=alpha)

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Draw the sharp contour (the label chip is drawn later, on top).

        The contour is traced from the source-resolution mask and scaled to the
        canvas, so it stays crisp regardless of any display scaling.

        Args:
            ctx: The current render context (display resolution).
            style: The resolved style.
            color: The resolved fill color.

        """
        if not self.contour or style.mask_outline is MaskOutline.NONE:
            return
        canvas = ctx.canvas
        binary = self._dense()
        sx, sy, smoothing = _contour_transform(canvas, binary, style)
        # A nested sub-label fills in its own class color but traces its contour in
        # the parent's, so it stays tied to the parent (see outline_color).
        stroke = self.outline_color(ctx, color)
        for ring in _mask_contours(binary):
            canvas.polygon(
                _contour_path(ring, sx, sy, smoothing),
                stroke=stroke,
                stroke_width=style.stroke_width,
                dash=style.dash,
            )

    def draw_label(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Place the mask's label chip on top of all shapes.

        Args:
            ctx: The current render context.
            style: The resolved style.
            color: The resolved fill color.

        """
        canvas = ctx.canvas
        mask_h, mask_w = self._dense().shape[:2]
        region = self.extent()
        if region is not None:
            sx = canvas.width / mask_w
            sy = canvas.height / mask_h
            region = Rect(
                region.left * sx,
                region.top * sy,
                region.right * sx,
                region.bottom * sy,
            )
            ctx.emit_hit(region, self.tooltip)
            if self.label_chip:
                place_label(
                    ctx,
                    region,
                    self.label,
                    self.score,
                    self.payload,
                    color,
                    style,
                    id(self),
                )


class SemanticMask(Annotation):
    """A dense label map, colored per class id from the palette.

    A rendering-only construct: LDF has no single semantic-label-map annotation
    (semantic segmentation is stored per class as
    `SegmentationAnnotation`), so `from_ldf` combines
    those into the ``(H, W)`` id map this draws.

    .. image:: TODO-HOST/masks_keypoints.png
       :alt: A semantic mask beside instance and polygon masks and keypoints.

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

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import Image, SemanticMask
        >>> labels = np.array([[0, 1], [2, 2]], dtype=np.uint8)
        >>> mask = SemanticMask(labels=labels, names={1: "road", 2: "car"})
        >>> Image(np.zeros((2, 2, 3), np.uint8)).add(mask).render().shape
        (2, 2, 4)

    """

    #: Semantic segmentation is a background layer: it always renders beneath
    #: boxes, instance masks, and keypoints, never on top of them.
    BACKGROUND: ClassVar[bool] = True
    LAYER: ClassVar[str] = "mask"

    labels: np.ndarray | None = None
    names: dict[int, str] | list[str] | None = None
    ignore_index: int | list[int] = 0
    color_map: dict[int, ColorLike] | None = None
    fill_alpha: float | None = None

    #: Memoized sorted class ids present in ``labels``. Both the fill pass and the
    #: contour pass iterate them, and finding them sorts the whole label map, so
    #: it is computed once. The label map is treated as immutable once built.
    _ids_cache: np.ndarray | None = PrivateAttr(default=None)

    def _ids(self) -> np.ndarray:
        """Return the unique class ids in ``labels`` (computed once, cached)."""
        if self._ids_cache is None:
            self._ids_cache = np.unique(np.asarray(self.labels))
        return self._ids_cache

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

        Examples:
            Each per-class LDF `SegmentationAnnotation` becomes one id in the
            combined map, in first-seen order (id 0 stays background):

            >>> import numpy as np
            >>> from luxonis_ml.ldf import SegmentationAnnotation
            >>> road = SegmentationAnnotation.model_validate(
            ...     {"mask": np.array([[1, 1], [0, 0]], np.uint8)}
            ... )
            >>> sky = SegmentationAnnotation.model_validate(
            ...     {"mask": np.array([[0, 0], [1, 1]], np.uint8)}
            ... )
            >>> sm = SemanticMask.from_ldf([("road", road), ("sky", sky)])
            >>> sm.names
            {1: 'road', 2: 'sky'}
            >>> sm.labels.tolist()
            [[1, 1], [2, 2]]

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

    def _ignored(self) -> set[int]:
        """Return the set of class ids treated as background."""
        return (
            {self.ignore_index}
            if isinstance(self.ignore_index, int)
            else set(self.ignore_index)
        )

    def draw_fill(
        self, ctx: RenderContext, style: Style, color: Color
    ) -> None:
        """Overlay one translucent color per class id at native resolution.

        Args:
            ctx: The current render context (native resolution).
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
        ignore = self._ignored()
        labels = resize_mask(
            np.asarray(self.labels), canvas.width, canvas.height
        )
        for class_id in self._ids():
            cid = int(class_id)
            if cid in ignore:
                continue
            canvas.overlay_mask(
                labels == class_id, self._color(cid, palette), alpha=alpha
            )

    def draw(self, ctx: RenderContext, style: Style, color: Color) -> None:
        """Stroke the sharp per-class contours (second, display-size pass).

        Contours are traced from the source-resolution label map and scaled to
        the canvas so they stay crisp regardless of display scaling.

        Args:
            ctx: The current render context (display resolution).
            style: The resolved style.
            color: Unused (colors are per class id).

        """
        if (
            self.labels is None
            or style.stroke_width <= 0
            or style.mask_outline is MaskOutline.NONE
        ):
            return
        canvas = ctx.canvas
        palette = self.resolved_palette(ctx)
        ignore = self._ignored()
        labels = np.asarray(self.labels)
        sx, sy, smoothing = _contour_transform(canvas, labels, style)
        for class_id in self._ids():
            cid = int(class_id)
            if cid in ignore:
                continue
            class_color = self._color(cid, palette)
            for ring in _mask_contours(labels == class_id):
                canvas.polygon(
                    _contour_path(ring, sx, sy, smoothing),
                    stroke=class_color,
                    stroke_width=style.stroke_width,
                )
