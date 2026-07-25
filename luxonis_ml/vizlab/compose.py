"""Combine several images: blend (mixup), stack, and grid.

Every function here renders its inputs at native resolution (so text and edges
stay crisp) and returns a brand-new `Image` wrapping the
composited raster. The inputs are never mutated — combining multiple images makes
purity unambiguous, unlike the in-place ``Image.add``.
"""

import math
from collections.abc import Sequence

import numpy as np

from .canvas import Canvas
from .color import Color, ColorLike
from .geometry import Rect
from .image import Image
from .style import DEFAULT_STYLE, Style

_DEFAULT_BG = Color(24, 24, 28, 255)
"""Neutral dark background painted behind stacked/gridded cells and pad gaps."""

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

    Mismatched sizes are padded to the larger size (top-left anchored, so pixel
    coordinates stay valid for both images' annotations).

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
    if first.shape != second.shape:
        height = max(first.shape[0], second.shape[0])
        width = max(first.shape[1], second.shape[1])
        fill = Color.parse(bg)
        first = _pad(first, width, height, fill)
        second = _pad(second, width, height, fill)
    mixed = (1.0 - alpha) * first.astype(np.float32) + alpha * second.astype(
        np.float32
    )
    result = Image(np.clip(mixed, 0, 255).astype(np.uint8), theme=base.theme)
    for annotation in (*base.annotations, *other.annotations):
        result.add(annotation)
    return result


def hstack(
    images: Sequence[Image],
    *,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
) -> Image:
    """Lay images out in a single row, left to right.

    Args:
        images: The images to place.
        pad: Gap between cells and outer margin, in pixels.
        bg: Background color.
        titles: Optional per-image titles drawn above each cell.
        style: Style whose font is used for titles.

    Returns:
        A new `Image` of the row.

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
    )


def vstack(
    images: Sequence[Image],
    *,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
) -> Image:
    """Lay images out in a single column, top to bottom.

    Args:
        images: The images to place.
        pad: Gap between cells and outer margin, in pixels.
        bg: Background color.
        titles: Optional per-image titles drawn above each cell.
        style: Style whose font is used for titles.

    Returns:
        A new `Image` of the column.

    """
    return _grid(images, ncols=1, pad=pad, bg=bg, titles=titles, style=style)


def grid(
    images: Sequence[Image],
    *,
    ncols: int | None = None,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
) -> Image:
    """Lay images out in a grid of uniform cells.

    Args:
        images: The images to place, filled row-major.
        ncols: Number of columns; defaults to ``ceil(sqrt(n))``.
        pad: Gap between cells and outer margin, in pixels.
        bg: Background color.
        titles: Optional per-image titles drawn above each cell.
        style: Style whose font is used for titles.

    Returns:
        A new `Image` of the grid.

    Examples:
        >>> from luxonis_ml.vizlab.image import Image
        >>> import numpy as np
        >>> grid([Image(np.zeros((10, 10, 3), np.uint8))] * 4, ncols=2, pad=4).render().shape
        (32, 32, 4)

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


def _title_height(titles: Sequence[str] | None, style: Style) -> float:
    """Height reserved above each cell for a title, or ``0`` when there are none."""
    if not titles:
        return 0.0
    metrics = _MEASURE.measure_text(
        "Ag", style.font_size, weight=style.font_weight
    )
    return metrics.height + 2 * _TITLE_PAD


def _grid(
    images: Sequence[Image],
    *,
    ncols: int,
    pad: int,
    bg: ColorLike,
    titles: Sequence[str] | None,
    style: Style,
) -> Image:
    """Shared tiling used by `grid`, `hstack`, and `vstack`."""
    rasters = [img.render() for img in images]
    if not rasters:
        raise ValueError("cannot compose an empty sequence of images")

    cols = min(ncols, len(rasters))
    rows = math.ceil(len(rasters) / cols)
    cell_w = max(r.shape[1] for r in rasters)
    cell_h = max(r.shape[0] for r in rasters)
    title_h = _title_height(titles, style)

    width = pad + cols * (cell_w + pad)
    height = round(pad + rows * (title_h + cell_h + pad))

    background = Color.parse(bg)
    canvas = Canvas.blank(width, height)
    canvas.rounded_rect(Rect(0, 0, width, height), 0.0, fill=background)
    text_color = background.readable_text_color()

    for i, raster in enumerate(rasters):
        row, col = divmod(i, cols)
        cell_x = pad + col * (cell_w + pad)
        cell_y = round(pad + row * (title_h + cell_h + pad))
        if titles and i < len(titles) and titles[i]:
            _draw_title(
                canvas,
                titles[i],
                cell_x + cell_w / 2,
                cell_y,
                style,
                text_color,
            )
        h, w = raster.shape[:2]
        x = cell_x + (cell_w - w) // 2
        y = round(cell_y + title_h) + (cell_h - h) // 2
        canvas.blit(raster, x, y)

    return Image(canvas.to_rgba())


def _draw_title(
    canvas: Canvas,
    text: str,
    center_x: float,
    top: float,
    style: Style,
    color: Color,
) -> None:
    """Draw a centered title string above a cell."""
    metrics = canvas.measure_text(
        text, style.font_size, weight=style.font_weight
    )
    baseline_y = top + _TITLE_PAD + metrics.ascent
    canvas.text(
        (center_x - metrics.width / 2, baseline_y),
        text,
        size=style.font_size,
        color=color,
        weight=style.font_weight,
    )
