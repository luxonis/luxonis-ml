"""Combine several images: blend (mixup), stack, and grid.

Every function here renders its inputs at native resolution (so text and edges
stay crisp) and returns a brand-new `Image` wrapping the
composited raster. The inputs are never mutated — combining multiple images makes
purity unambiguous, unlike the in-place ``Image.add``.
"""

import math
from collections.abc import Sequence

import numpy as np

from .annotations import Annotation, BBox, Keypoints, Mask, SemanticMask
from .annotations.mask import _resize_mask
from .canvas import Canvas
from .color import Color, ColorLike
from .geometry import Rect
from .image import Image
from .style import DARK_THEME, DEFAULT_STYLE, Style

_DEFAULT_BG = DARK_THEME.background
"""Brand dark background painted behind stacked/gridded cells and pad gaps
(the default theme's background, so composites match single-image renders)."""

_TITLE_PAD = 6.0
_MEASURE = Canvas.blank(1, 1)


def blend(
    base: Image,
    other: Image,
    alpha: float = 0.3,
    *,
    bg: ColorLike = _DEFAULT_BG,
) -> Image:
    """Blend two images for mixup, keeping both label sets crisp.

    Only the *base rasters* are mixed (``(1 - alpha) * base + alpha * other``); the
    annotations of both images are carried onto the result as a live scene graph and
    drawn — at full opacity by default — when the result renders. This means labels
    stay sharp instead of being alpha-faded into the background, and the shared
    collision-aware layout keeps the two images' labels from landing on top of each
    other. To fade labels anyway, give them a style with ``label_alpha < 1``.

    Mismatched sizes are padded at the bottom and right to the larger canvas.
    Boxes and keypoints are renormalized, masks are resized to their source image
    and padded, and nested annotations are transformed recursively. Image-level
    overlays remain anchored to the resulting canvas.

    Args:
        base: The first image.
        other: The second image, weighted by ``alpha``.
        alpha: Weight of ``other``'s base raster in ``[0, 1]``.
        bg: Background color used when padding mismatched sizes.

    Returns:
        A new `Image`: the blended base plus both annotation
        lists. Neither input is mutated.

    Examples:
        >>> from luxonis_ml.vizlab.image import Image
        >>> import numpy as np
        >>> a = Image(np.zeros((4, 4, 3), np.uint8))
        >>> b = Image(np.full((4, 4, 3), 100, np.uint8))
        >>> int(blend(a, b, alpha=0.5).render()[0, 0, 0])
        50

    """
    first = base.base_rgba()
    second = other.base_rgba()
    base_annotations = base.annotations
    other_annotations = other.annotations
    if first.shape != second.shape:
        height = max(first.shape[0], second.shape[0])
        width = max(first.shape[1], second.shape[1])
        fill = Color.parse(bg)
        base_annotations = [
            _pad_annotation(
                annotation,
                source_width=first.shape[1],
                source_height=first.shape[0],
                width=width,
                height=height,
            )
            for annotation in base.annotations
        ]
        other_annotations = [
            _pad_annotation(
                annotation,
                source_width=second.shape[1],
                source_height=second.shape[0],
                width=width,
                height=height,
            )
            for annotation in other.annotations
        ]
        first = _pad(first, width, height, fill)
        second = _pad(second, width, height, fill)
    mixed = (1.0 - alpha) * first.astype(np.float32) + alpha * second.astype(
        np.float32
    )
    result = Image(np.clip(mixed, 0, 255).astype(np.uint8), theme=base.theme)
    for annotation in (*base_annotations, *other_annotations):
        result.add(annotation)
    return result


def _pad_annotation(
    annotation: Annotation,
    *,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
) -> Annotation:
    """Copy an annotation from a top-left anchored source onto a padded canvas."""
    clone = annotation.model_copy(deep=False)
    scale_x = source_width / width
    scale_y = source_height / height

    if isinstance(clone, BBox):
        clone.x *= scale_x
        clone.y *= scale_y
        clone.w *= scale_x
        clone.h *= scale_y
    elif isinstance(clone, Keypoints):
        clone.keypoints = [
            (x * scale_x, y * scale_y, visibility)
            for x, y, visibility in clone.keypoints
        ]
    elif isinstance(clone, Mask):
        from luxonis_ml.ldf import SegmentationAnnotation

        source_mask = _resize_mask(
            clone.to_numpy(), source_width, source_height
        )
        padded = np.zeros((height, width), dtype=source_mask.dtype)
        padded[:source_height, :source_width] = source_mask
        rle = SegmentationAnnotation._numpy_to_rle(padded)
        clone.height = rle["height"]
        clone.width = rle["width"]
        clone.counts = rle["counts"].encode("utf-8")
    elif isinstance(clone, SemanticMask) and clone.labels is not None:
        labels = _resize_mask(
            np.asarray(clone.labels), source_width, source_height
        )
        ignored = clone._ignored()
        background = next(iter(ignored), 0)
        padded = np.full((height, width), background, dtype=labels.dtype)
        padded[:source_height, :source_width] = labels
        clone.labels = padded

    clone.children = [
        _pad_annotation(
            child,
            source_width=source_width,
            source_height=source_height,
            width=width,
            height=height,
        )
        for child in annotation.children
    ]
    return clone


def hstack(
    images: Sequence[Image],
    *,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
) -> Image:
    """Render images into a new, left-to-right row.

    Each input is rendered before composition. Cells use the tallest input
    height, and smaller images are centered within their cell. The inputs and
    their scene graphs are not mutated.

    Args:
        images: The images to place.
        pad: Gap between cells and outer margin, in pixels.
        bg: Background color.
        titles: Optional per-image titles drawn above each cell.
        style: Style whose font is used for titles.

    Returns:
        A new `Image` of the row.

    Raises:
        ValueError: If ``images`` is empty.

    Examples:
        >>> from luxonis_ml.vizlab.image import Image
        >>> import numpy as np
        >>> cells = [Image(np.zeros((10, 20, 3), np.uint8)) for _ in range(2)]
        >>> hstack(cells, pad=5).render().shape
        (20, 55, 4)

    """
    return _grid(
        images,
        ncols=len(list(images)) or 1,
        pad=pad,
        bg=bg,
        titles=titles,
        style=style,
    )[0]


def vstack(
    images: Sequence[Image],
    *,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
) -> Image:
    """Render images into a new, top-to-bottom column.

    Cells use the widest input width, and smaller images are centered within
    their cell. The inputs and their scene graphs are not mutated.

    Args:
        images: The images to place.
        pad: Gap between cells and outer margin, in pixels.
        bg: Background color.
        titles: Optional per-image titles drawn above each cell.
        style: Style whose font is used for titles.

    Returns:
        A new `Image` of the column.

    Raises:
        ValueError: If ``images`` is empty.

    """
    return _grid(images, ncols=1, pad=pad, bg=bg, titles=titles, style=style)[
        0
    ]


def grid(
    images: Sequence[Image],
    *,
    ncols: int | None = None,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
) -> Image:
    """Render images into a new row-major grid of uniform cells.

    Cell width and height come from the largest rendered input. Smaller images
    are centered, optional titles occupy a shared title band, and unused cells in
    the final row are omitted.

    Args:
        images: The images to place, filled row-major.
        ncols: Positive number of columns; defaults to ``ceil(sqrt(n))``.
        pad: Gap between cells and outer margin, in pixels.
        bg: Background color.
        titles: Optional per-image titles drawn above each cell.
        style: Style whose font is used for titles.

    Returns:
        A new `Image` of the grid.

    Raises:
        ValueError: If ``images`` is empty.

    Examples:
        >>> from luxonis_ml.vizlab.image import Image
        >>> import numpy as np
        >>> grid([Image(np.zeros((10, 10, 3), np.uint8))] * 4, ncols=2, pad=4).render().shape
        (32, 32, 4)

    """
    return grid_placed(
        images, ncols=ncols, pad=pad, bg=bg, titles=titles, style=style
    )[0]


def grid_placed(
    images: Sequence[Image],
    *,
    ncols: int | None = None,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
) -> tuple[Image, list[tuple[int, int, int, int]]]:
    """Like `grid`, but also return each tile's ``(x, y, w, h)`` placement.

    The placements (one per input image, in order) give where each tile's raster
    landed in the composite, so callers can map tile-local coordinates — e.g.
    detection boxes — into the composed image (used for hover hit-testing).

    Args:
        images: The images to place, filled row-major.
        ncols: Positive number of columns; defaults to ``ceil(sqrt(n))``.
        pad: Gap between cells and outer margin, in pixels.
        bg: Background color.
        titles: Optional per-image titles drawn above each cell.
        style: Style whose font is used for titles.

    Returns:
        A ``(grid_image, placements)`` pair. Each placement is an integer
        ``(x, y, width, height)`` tuple in composite-image pixels.

    Raises:
        ValueError: If ``images`` is empty.

    """
    count = len(list(images))
    cols = ncols if ncols is not None else max(1, math.ceil(math.sqrt(count)))
    return _grid(
        images, ncols=cols, pad=pad, bg=bg, titles=titles, style=style
    )


def _pad(rgba: np.ndarray, width: int, height: int, fill: Color) -> np.ndarray:
    """Pad an RGBA array to ``width`` x ``height``, anchored top-left."""
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :] = fill.rgba
    out[: rgba.shape[0], : rgba.shape[1]] = rgba
    return out


def _wrap_titles(
    titles: Sequence[str] | None, count: int, cell_w: float, style: Style
) -> list[list[str]]:
    """Wrap each title to the cell width so long titles never overflow the cell."""
    if not titles:
        return [[] for _ in range(count)]
    max_w = max(1.0, cell_w - 2 * _TITLE_PAD)
    wrapped: list[list[str]] = []
    for i in range(count):
        title = titles[i] if i < len(titles) else ""
        wrapped.append(
            _MEASURE.wrap_text(
                title,
                style.font_size,
                max_width=max_w,
                weight=style.font_weight,
            )
            if title
            else []
        )
    return wrapped


def _grid(
    images: Sequence[Image],
    *,
    ncols: int,
    pad: int,
    bg: ColorLike,
    titles: Sequence[str] | None,
    style: Style,
) -> tuple[Image, list[tuple[int, int, int, int]]]:
    """Shared tiling used by `grid`, `hstack`, and `vstack`.

    Returns the composed image and, for each input image (in order), the
    ``(x, y, w, h)`` rectangle its raster occupies in the composite.
    """
    rasters = [img.render() for img in images]
    if not rasters:
        raise ValueError("cannot compose an empty sequence of images")

    cols = min(ncols, len(rasters))
    rows = math.ceil(len(rasters) / cols)
    cell_w = max(r.shape[1] for r in rasters)
    cell_h = max(r.shape[0] for r in rasters)

    line_h = _MEASURE.measure_text(
        "Ag", style.font_size, weight=style.font_weight
    ).height
    wrapped = _wrap_titles(titles, len(rasters), cell_w, style)
    max_lines = max((len(lines) for lines in wrapped), default=0)
    title_h = max_lines * line_h + 2 * _TITLE_PAD if max_lines else 0.0

    width = pad + cols * (cell_w + pad)
    height = round(pad + rows * (title_h + cell_h + pad))

    background = Color.parse(bg)
    canvas = Canvas.blank(width, height)
    canvas.rounded_rect(Rect(0, 0, width, height), 0.0, fill=background)
    text_color = background.readable_text_color()

    placements: list[tuple[int, int, int, int]] = []
    for i, raster in enumerate(rasters):
        row, col = divmod(i, cols)
        cell_x = pad + col * (cell_w + pad)
        cell_y = round(pad + row * (title_h + cell_h + pad))
        if wrapped[i]:
            _draw_title(
                canvas,
                wrapped[i],
                cell_x + cell_w / 2,
                cell_y,
                style,
                text_color,
                line_h,
            )
        h, w = raster.shape[:2]
        x = cell_x + (cell_w - w) // 2
        y = round(cell_y + title_h) + (cell_h - h) // 2
        canvas.blit(raster, x, y)
        placements.append((x, y, w, h))

    return Image(canvas.to_rgba()), placements


def _draw_title(
    canvas: Canvas,
    lines: Sequence[str],
    center_x: float,
    top: float,
    style: Style,
    color: Color,
    line_h: float,
) -> None:
    """Draw centered, wrapped title lines above a cell."""
    y = top + _TITLE_PAD
    for line in lines:
        metrics = canvas.measure_text(
            line, style.font_size, weight=style.font_weight
        )
        canvas.text(
            (center_x - metrics.width / 2, y + metrics.ascent),
            line,
            size=style.font_size,
            color=color,
            weight=style.font_weight,
        )
        y += line_h
