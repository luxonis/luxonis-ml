"""Combine several images: blend (mixup), stack, and grid.

`blend` mixes two base rasters and returns a new `Image`. The stacking helpers
return a `Composite` that places its inputs without flattening them: each tile
draws itself through `Renderable._draw_onto` at composition time, so a grid
rendered to SVG keeps every stroke and label chip as vector. The inputs are never
mutated — combining multiple images makes purity unambiguous, unlike the in-place
``Image.add``.
"""

import math
import re
from collections.abc import Mapping, Sequence
from typing import NamedTuple, TypeVar, overload

import numpy as np

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab._util import is_sequence
from luxonis_ml.vizlab.annotations import (
    Annotation,
    BBox,
    Keypoints,
    Mask,
    SemanticMask,
)
from luxonis_ml.vizlab.annotations.mask import resize_mask
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render import RenderEnvironment
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.render.capture import InteractionCapture
from luxonis_ml.vizlab.render.markup import Span, parse
from luxonis_ml.vizlab.scene.image import (
    Composite,
    Image,
    Renderable,
)
from luxonis_ml.vizlab.style import (
    DARK_THEME,
    DEFAULT_STYLE,
    Style,
    style_scale,
)

_DEFAULT_BG = DARK_THEME.background
"""Brand dark background painted behind stacked/gridded cells and pad gaps
(the default theme's background, so composites match single-image renders)."""

_TITLE_PAD = 6.0
#: Cell titles render a step larger and bold, so they read as headings above
#: their tiles rather than as body text.
_TITLE_SCALE = 1.3
_TITLE_WEIGHT = 700
_MEASURE = Canvas.blank(1, 1)

_T = TypeVar("_T")


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
        >>> from luxonis_ml.vizlab.scene.image import Image
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
    result = Image(
        np.clip(mixed, 0, 255).astype(np.uint8), options=base.options
    )
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

        source_mask = resize_mask(
            clone.to_numpy(), source_width, source_height
        )
        padded = np.zeros((height, width), dtype=source_mask.dtype)
        padded[:source_height, :source_width] = source_mask
        rle = SegmentationAnnotation._numpy_to_rle(padded)
        clone.height = rle["height"]
        clone.width = rle["width"]
        clone.counts = rle["counts"].encode("utf-8")
        clone._dense_cache = None
    elif isinstance(clone, SemanticMask) and clone.labels is not None:
        labels = resize_mask(
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
    images: Sequence[Renderable],
    *,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
) -> Renderable:
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
        A `Composite` of the row.

    Raises:
        ValueError: If ``images`` is empty.

    Examples:
        >>> from luxonis_ml.vizlab.scene.image import Image
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
    images: Sequence[Renderable],
    *,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
) -> Renderable:
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
        A `Composite` of the column.

    Raises:
        ValueError: If ``images`` is empty.

    """
    return _grid(images, ncols=1, pad=pad, bg=bg, titles=titles, style=style)[
        0
    ]


def grid(
    images: Sequence[Renderable],
    *,
    ncols: int | None = None,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
    emphasize_titles: bool = True,
) -> Renderable:
    """Render images into a new row-major grid of uniform cells.

    Cell width and height come from the largest rendered input. Smaller images
    are centered, optional titles occupy a shared title band, and unused cells in
    the final row are omitted.

    .. image:: TODO-HOST/compose.png
       :alt: Blend, stack, and grid composition of rendered images.

    Args:
        images: The images to place, filled row-major.
        ncols: Positive number of columns; defaults to ``ceil(sqrt(n))``.
        pad: Gap between cells and outer margin, in pixels.
        bg: Background color.
        titles: Optional per-image titles drawn above each cell.
        style: Style whose font is used for titles.
        emphasize_titles: When ``True`` (the default), titles render a step larger
            and bold so they read as headings; set ``False`` for a nested sub-grid
            whose labels should stay subordinate to the outer titles.

    Returns:
        A `Composite` of the grid.

    Raises:
        ValueError: If ``images`` is empty.

    Examples:
        >>> from luxonis_ml.vizlab.scene.image import Image
        >>> import numpy as np
        >>> grid(
        ...     [Image(np.zeros((10, 10, 3), np.uint8))] * 4, ncols=2, pad=4
        ... ).render().shape
        (32, 32, 4)

    """
    return grid_placed(
        images,
        ncols=ncols,
        pad=pad,
        bg=bg,
        titles=titles,
        style=style,
        emphasize_titles=emphasize_titles,
    )[0]


def grid_placed(
    images: Sequence[Renderable],
    *,
    ncols: int | None = None,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
    emphasize_titles: bool = True,
) -> tuple[Renderable, list[tuple[int, int, int, int]]]:
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
        emphasize_titles: When ``True`` (the default), titles render a step larger
            and bold; set ``False`` to keep a nested sub-grid's labels subordinate.

    Returns:
        A ``(composite, placements)`` pair. Each placement is an integer
        ``(x, y, width, height)`` tuple in composite pixels.

    Raises:
        ValueError: If ``images`` is empty.

    """
    cols = ncols if ncols is not None else _default_cols(len(images))
    composite, placements = _grid(
        images,
        ncols=cols,
        pad=pad,
        bg=bg,
        titles=titles,
        style=style,
        emphasize_titles=emphasize_titles,
    )
    return composite, placements


def fit_grid(
    images: Sequence[Renderable],
    *,
    target: tuple[int, int],
    ncols: int | None = None,
    reserve: float = 0.0,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    titles: Sequence[str] | None = None,
    style: Style = DEFAULT_STYLE,
    allow_upscale: bool = False,
) -> Renderable:
    """Grid ``images`` scaled so the whole composite fits within ``target``.

    Owns the column/chrome/scale arithmetic a caller would otherwise duplicate:
    it derives one tile scale from the tiles' native sizes, the grid padding, an
    optional per-row title band, and a horizontal ``reserve`` (e.g. for a side
    panel), then renders each tile crisply at the scaled size (via
    `Image.render_at`) before tiling — so labels stay sharp instead of being
    downscaled afterwards. Inputs are copied, never mutated.

    Args:
        images: The tiles to place, filled row-major.
        target: The ``(width, height)`` pixel budget the composite must fit in.
        ncols: Column count; defaults to `_smart_cols`.
        reserve: Horizontal pixels to keep free (subtracted from the width
            budget), e.g. for a metadata panel drawn beside the grid.
        pad: Gap between cells and the outer margin, in pixels.
        bg: Background color.
        titles: Optional per-image titles.
        style: Style whose font is used for titles.
        allow_upscale: When ``True``, tiles smaller than the budget are scaled
            *up* to fill it (rendered crisply at the larger size), so a small
            image is not shown tiny on a big screen. Defaults to ``False``, which
            only ever shrinks (the composite never exceeds its native size).

    Returns:
        A `Composite` of the grid, scaled to fit ``target``.

    Raises:
        ValueError: If ``images`` is empty.

    """
    images = list(images)
    if not images:
        raise ValueError("cannot compose an empty sequence of images")
    count = len(images)
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    title_h = _fit_title_h(cell_w, cell_h, titles, style)
    if ncols is None:
        cols = fitting_cols(
            count,
            (cell_w, cell_h),
            target=target,
            reserve=reserve,
            pad=pad,
            title_h=title_h,
        )
    else:
        cols = ncols
    cols = max(1, min(cols, count))

    scale = _fit_scale(
        count,
        (cell_w, cell_h),
        cols=cols,
        target=target,
        reserve=reserve,
        pad=pad,
        title_h=title_h,
    )
    if not allow_upscale:
        scale = min(scale, 1.0)

    scaled = [
        img.copy().render_at(
            (
                max(1, round(img.width * scale)),
                max(1, round(img.height * scale)),
            )
        )
        for img in images
    ]
    return grid(scaled, ncols=cols, pad=pad, bg=bg, titles=titles, style=style)


def _fit_scale(
    count: int,
    cell: tuple[int, int],
    *,
    cols: int,
    target: tuple[int, int],
    reserve: float,
    pad: int,
    title_h: float,
) -> float:
    """How much ``count`` cells shrink to fit ``target`` at ``cols`` columns."""
    cell_w, cell_h = cell
    rows = math.ceil(count / cols)
    avail_w = max(1.0, target[0] - pad * (cols + 1) - reserve)
    avail_h = max(1.0, target[1] - pad * (rows + 1) - rows * title_h)
    return min(avail_w / (cols * cell_w), avail_h / (rows * cell_h))


def fitting_cols(
    count: int,
    cell: tuple[int, int],
    *,
    target: tuple[int, int],
    reserve: float = 0.0,
    pad: int = 10,
    title_h: float = 0.0,
) -> int:
    """Choose the column count that shows the tiles largest inside ``target``.

    The layout question only has a right answer once you know the shape of the
    hole it has to fill. `_smart_cols` aims for a square *figure*, which is the
    wrong target for a 16:9 screen: three 16:9 tiles laid in one row there come
    out 1.5x smaller than the same three in a 2x2. So rather than guess from the
    tiles' aspect alone, this tries every arrangement and keeps the one that
    renders them biggest.

    Ties go to the arrangement with fewer rows, which puts a stereo pair side by
    side rather than stacked — the reading a viewer expects when two tiles are
    the same size and either would fit.

    Args:
        count: How many tiles are being placed.
        cell: The ``(width, height)`` of one cell, at native size.
        target: The ``(width, height)`` pixel budget to fit inside.
        reserve: Horizontal pixels kept free, e.g. for a side panel.
        pad: Gap between cells and the outer margin.
        title_h: Height of the per-row title band, if any.

    Returns:
        The best column count, between ``1`` and ``count``.

    Examples:
        >>> from luxonis_ml.vizlab.layout.compose import fitting_cols
        >>> # Three 16:9 tiles on a 16:9 screen want a 2x2, not one row.
        >>> fitting_cols(3, (960, 540), target=(1728, 972), pad=0)
        2
        >>> # A stereo pair goes side by side.
        >>> fitting_cols(2, (960, 540), target=(1728, 972), pad=0)
        2
        >>> # Tall tiles spread out instead of stacking.
        >>> fitting_cols(2, (540, 960), target=(1728, 972), pad=0)
        2

    """
    if count <= 1:
        return 1
    best_cols, best_key = 1, (-1.0, 0)
    for cols in range(1, count + 1):
        scale = _fit_scale(
            count,
            cell,
            cols=cols,
            target=target,
            reserve=reserve,
            pad=pad,
            title_h=title_h,
        )
        # Rounded so float noise cannot decide a tie that the row count should:
        # two arrangements showing the tiles the same size are settled by
        # preferring the shorter one.
        key = (round(scale, 9), -math.ceil(count / cols))
        if key > best_key:
            best_cols, best_key = cols, key
    return best_cols


def _fit_title_h(
    cell_w: int, cell_h: int, titles: Sequence[str] | None, style: Style
) -> float:
    """Reserve for a single (emphasized) title line above each cell, or ``0``.

    A slight over-estimate is intentional: it uses the native (unscaled) cell
    size, so the composite errs toward fitting the target with a little room to
    spare rather than overflowing it.
    """
    if not titles:
        return 0.0
    scale = style_scale(cell_w, cell_h)
    scaled = style.scaled(scale)
    title_style = scaled.merge(
        font_size=scaled.font_size * _TITLE_SCALE, font_weight=_TITLE_WEIGHT
    )
    line_h = _MEASURE.measure_text(
        "Ag", title_style.font_size, weight=title_style.font_weight
    ).height
    return line_h + 2 * _TITLE_PAD * scale


#: A single image, a flat group of images, or a titled group of images.
CombineGroup = (
    Renderable
    | Sequence[Renderable]
    | Mapping[str, "Renderable | Sequence[Renderable]"]
)


@overload
def combine(
    group: Image,
    /,
    *,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    style: Style = DEFAULT_STYLE,
    target: tuple[int, int] | None = None,
    reserve: float = 0.0,
    allow_upscale: bool = False,
) -> Image: ...


@overload
def combine(
    *groups: CombineGroup,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    style: Style = DEFAULT_STYLE,
    target: tuple[int, int] | None = None,
    reserve: float = 0.0,
    allow_upscale: bool = False,
) -> Renderable: ...


def combine(
    *groups: CombineGroup,
    pad: int = 10,
    bg: ColorLike = _DEFAULT_BG,
    style: Style = DEFAULT_STYLE,
    target: tuple[int, int] | None = None,
    reserve: float = 0.0,
    allow_upscale: bool = False,
) -> Renderable:
    """Lay out a mixed set of images (and grouped images) into one figure.

    A smart, layout-free counterpart to `hstack`/`vstack`/`grid`: pass any mix
    of finished scenes and scene groups, and ``combine`` arranges them
    sensibly without the caller choosing rows or columns. Anything renderable
    works, so an already-composed `grid`/`with_panel` result nests as one cell.
    Each positional ``group`` is one of:

    - a single `Renderable` (an `Image` or a composite) — placed as-is;
    - a sequence of them — gathered into their own sub-grid;
    - a mapping ``{title: scene-or-sequence}`` — each entry becomes a titled
      sub-block (its key drawn as a heading), the blocks gathered into a
      sub-grid.

    The resolved groups are then arranged with an automatically chosen column
    count (see `_smart_cols`). Inputs are rendered at native resolution and
    never mutated; a brand-new scene is returned.

    Args:
        groups: The images / image-groups to compose.
        pad: Gap between cells and the outer margin, in pixels.
        bg: Background color painted behind cells and gaps.
        style: Style whose font is used for titles.
        target: A ``(width, height)`` budget the whole figure must fit inside.
            Given one, the column count is chosen to show the tiles *largest*
            within it (see `fitting_cols`) rather than to square up the figure,
            and the tiles are re-rendered crisply at that size. ``None`` lays
            everything out at native size.
        reserve: Horizontal pixels of ``target`` to keep free, e.g. for a side
            panel drawn beside the figure.
        allow_upscale: Whether tiles smaller than ``target`` may be scaled *up*
            to fill it. Ignored without a ``target``.

    Returns:
        A `Composite` of the composed figure — or, when a single image is
        given, a copy of it (scaled, if a ``target`` was set).

    Raises:
        ValueError: If no groups are given, or a group is empty.
        TypeError: If a group is not a `Renderable`, a sequence, or a mapping.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import Image, combine
        >>> gt = Image(np.zeros((40, 40, 3), np.uint8))
        >>> pred = Image(np.zeros((40, 40, 3), np.uint8))
        >>> heat = [Image(np.zeros((20, 20, 3), np.uint8)) for _ in range(3)]
        >>> out = combine({"GT": gt, "Pred": pred}, heat).render()
        >>> out.shape[2], out.shape[0] > 40, out.shape[1] > 40
        (4, True, True)

    """
    if not groups:
        raise ValueError("cannot combine an empty set of groups")
    fit = _Fit(target, reserve, allow_upscale)

    # A lone mapping is the common case — one titled block per source — and is
    # laid out directly rather than nested inside a one-cell grid.
    if len(groups) == 1 and isinstance(groups[0], Mapping):
        group = order_by_position(groups[0])
        if not group:
            raise ValueError("cannot combine an empty group")
        tiles = [
            _resolve_member(value, pad=pad, bg=bg, style=style)
            for value in group.values()
        ]
        return _arrange(
            tiles,
            titles=list(group.keys()),
            fit=fit,
            pad=pad,
            bg=bg,
            style=style,
        )

    # A single group still returns a *copy* (see `_arrange`): never hand back the
    # caller's own object, so a later ``.add`` cannot mutate the input's scene.
    resolved = [_resolve_group(g, pad=pad, bg=bg, style=style) for g in groups]
    return _arrange(
        resolved, titles=None, fit=fit, pad=pad, bg=bg, style=style
    )


class _Fit(NamedTuple):
    """The pixel budget a composition has to live within, if any."""

    target: "tuple[int, int] | None"
    reserve: float
    allow_upscale: bool


def _arrange(
    tiles: list[Renderable],
    *,
    titles: "list[str] | None",
    fit: _Fit,
    pad: int,
    bg: ColorLike,
    style: Style,
) -> Renderable:
    """Place resolved tiles, sizing them to the budget when there is one."""
    if len(tiles) == 1:
        # One tile needs neither a title band nor grid chrome around it.
        only = tiles[0]
        if fit.target is None:
            return only.copy()
        scale = _fit_scale(
            1,
            (only.width, only.height),
            cols=1,
            target=fit.target,
            reserve=fit.reserve,
            pad=0,
            title_h=0.0,
        )
        if not fit.allow_upscale:
            scale = min(scale, 1.0)
        return only.copy().render_at(
            (
                max(1, round(only.width * scale)),
                max(1, round(only.height * scale)),
            )
        )
    if fit.target is None:
        return grid(
            tiles,
            ncols=_smart_cols(tiles),
            pad=pad,
            bg=bg,
            titles=titles,
            style=style,
        )
    return fit_grid(
        tiles,
        target=fit.target,
        reserve=fit.reserve,
        pad=pad,
        bg=bg,
        titles=titles,
        style=style,
        allow_upscale=fit.allow_upscale,
    )


def _as_images(member: "Sequence[object]") -> list[Renderable]:
    """Validate a group's sequence member into a non-empty list of scenes."""
    images: list[Renderable] = []
    for item in member:
        if not isinstance(item, Renderable):
            raise TypeError(f"unsupported group member type: {type(item)!r}")
        images.append(item)
    if not images:
        raise ValueError("cannot combine an empty group")
    return images


def _resolve_member(
    member: "Renderable | Sequence[object]",
    *,
    pad: int,
    bg: ColorLike,
    style: Style,
) -> Renderable:
    """Resolve a single mapping value / list to one image (nested sub-grid)."""
    if isinstance(member, Renderable):
        return member
    if is_sequence(member):
        items = _as_images(member)
        if len(items) == 1:
            return items[0]
        return grid(
            items,
            ncols=_smart_cols(items),
            pad=pad,
            bg=bg,
            style=style,
            emphasize_titles=False,
        )
    raise TypeError(f"unsupported group member type: {type(member)!r}")


def _resolve_group(
    group: CombineGroup, *, pad: int, bg: ColorLike, style: Style
) -> Renderable:
    if isinstance(group, Renderable):
        return group
    if isinstance(group, Mapping):
        if not group:
            raise ValueError("cannot combine an empty group")
        # Keys name the blocks, so a positional one also says where it goes.
        group = order_by_position(group)
        values = [
            _resolve_member(v, pad=pad, bg=bg, style=style)
            for v in group.values()
        ]
        return grid(
            values,
            ncols=_smart_cols(values),
            pad=pad,
            bg=bg,
            titles=list(group.keys()),
            style=style,
            emphasize_titles=True,
        )
    if is_sequence(group):
        return _resolve_member(group, pad=pad, bg=bg, style=style)
    raise TypeError(f"unsupported group type: {type(group)!r}")


#: Position words a multi-camera rig puts in its source names, ranked left to
#: right. Synonyms share a rank, so ``centre`` and ``middle`` sort alike.
_POSITION_RANK = {
    "left": 0,
    "l": 0,
    "center": 1,
    "centre": 1,
    "middle": 1,
    "mid": 1,
    "right": 2,
    "r": 2,
}


def _position_rank(name: str) -> int | None:
    """Rank a name by the position word it contains, if exactly one does.

    Matched on whole words rather than substrings, so ``rectified_left`` ranks
    but ``cleft_palate`` does not, and a name mentioning two positions (``left``
    *and* ``right``) is left unranked rather than guessed at.
    """
    tokens = {
        token for token in re.split(r"[^a-z0-9]+", name.lower()) if token
    }
    ranks = {_POSITION_RANK[t] for t in tokens if t in _POSITION_RANK}
    return next(iter(ranks)) if len(ranks) == 1 else None


def order_by_position(sources: "Mapping[str, _T]") -> "dict[str, _T]":
    """Reorder named scenes so positional names land where they describe.

    A stereo pair stored ``{"right": …, "left": …}`` would otherwise be drawn
    right-then-left, because nothing but insertion order says where a source
    belongs. Names carrying a position word (``left``, ``center``, ``right``,
    and their usual synonyms) are sorted into reading order.

    The reordering is deliberately minimal: only the *ranked* entries move, and
    only among the slots they already occupied. An unrecognised name keeps its
    place, so adding a ``thermal`` source cannot shuffle it to the end, and a
    mapping with nothing positional in it comes back untouched.

    Args:
        sources: Named scenes, or anything else keyed by source name.

    Returns:
        A new dict in position order.

    Examples:
        >>> from luxonis_ml.vizlab.layout.compose import order_by_position
        >>> list(order_by_position({"cam_right": 1, "cam_left": 2}))
        ['cam_left', 'cam_right']
        >>> list(order_by_position({"rgb": 1, "depth": 2}))  # nothing to order
        ['rgb', 'depth']
        >>> list(order_by_position({"right": 1, "thermal": 2, "left": 3}))
        ['left', 'thermal', 'right']

    """
    names = list(sources)
    slots = [
        i for i, name in enumerate(names) if _position_rank(name) is not None
    ]
    if len(slots) < 2:
        return dict(sources)
    ordered = sorted(
        (names[i] for i in slots),
        key=lambda name: _position_rank(name) or 0,
    )
    for slot, name in zip(slots, ordered, strict=True):
        names[slot] = name
    return {name: sources[name] for name in names}


#: The surface a layout is aimed at when the caller names no target. Screens,
#: slides, and figure panes are all roughly this, and picking *some* shape is
#: what lets an un-targeted composition agree with a targeted one.
_NOTIONAL_ASPECT = (16, 9)


def _smart_cols(images: Sequence[Renderable]) -> int:
    """Choose a column count for a composition with no stated size budget.

    Deliberately the same rule as `fitting_cols`, aimed at a notional 16:9
    surface. Sharing the rule is the point: a sample laid out for the screen and
    the same sample written to a file should not disagree about where its tiles
    go, which is exactly what happens when the two paths reason differently.

    Only the target's *aspect* matters here — with no padding or title band the
    scale is a common factor across every candidate, so the arrangement it picks
    is independent of how large the notional surface is.
    """
    count = len(images)
    if count <= 1:
        return 1
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    span = max(cell_w, cell_h) * count
    return fitting_cols(
        count,
        (cell_w, cell_h),
        target=(_NOTIONAL_ASPECT[0] * span, _NOTIONAL_ASPECT[1] * span),
        pad=0,
    )


def _default_cols(count: int) -> int:
    """Default plain-grid column count: roughly square, at least one."""
    return max(1, math.ceil(math.sqrt(count)))


def _pad(rgba: np.ndarray, width: int, height: int, fill: Color) -> np.ndarray:
    """Pad an RGBA array to ``width`` x ``height``, anchored top-left."""
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :] = fill.rgba
    out[: rgba.shape[0], : rgba.shape[1]] = rgba
    return out


def _wrap_titles(
    titles: Sequence[str] | None,
    count: int,
    cell_w: float,
    style: Style,
    pad: float,
) -> list[list[list[Span]]]:
    """Parse each title as markup and wrap it to the cell width.

    Titles support inline markup (see `luxonis_ml.vizlab.render.markup`) and
    newline breaks,
    so a caller can mix bold/mono runs and stack a heading over a subtitle. Each
    title becomes a list of lines, each a list of styled spans; long lines wrap so
    they never overflow the cell.
    """
    if not titles:
        return [[] for _ in range(count)]
    max_w = max(1.0, cell_w - 2 * pad)
    wrapped: list[list[list[Span]]] = []
    for i in range(count):
        title = titles[i] if i < len(titles) else ""
        spans = parse(title, weight=style.font_weight)
        wrapped.append(
            _MEASURE.wrap_spans(spans, style.font_size, max_width=max_w)
            if title
            else []
        )
    return wrapped


def _grid(
    images: Sequence[Renderable],
    *,
    ncols: int,
    pad: int,
    bg: ColorLike,
    titles: Sequence[str] | None,
    style: Style,
    emphasize_titles: bool = True,
) -> tuple[Renderable, list[tuple[int, int, int, int]]]:
    """Shared tiling used by `grid`, `hstack`, and `vstack`.

    Returns a `Composite` that draws the tiles (each via its own
    `Renderable._draw_onto`, so their annotations stay vector in an SVG) with
    titles as vector text, plus each tile's ``(x, y, w, h)`` placement.
    Nested scenes emit interactions through the same transformed draw path.
    """
    if not images:
        raise ValueError("cannot compose an empty sequence of images")
    # Tiles are placed at their display (render_at) size.
    sizes = [img._resolved_size(None) for img in images]

    cols = min(ncols, len(images))
    rows = math.ceil(len(images) / cols)
    cell_w = max(w for w, _ in sizes)
    cell_h = max(h for _, h in sizes)

    # Scale titles to the tile size so they don't shrink against large images.
    # Emphasized titles (the default) render a step larger and bold so they read
    # as headings; nested sub-grids can opt out to keep their labels subordinate.
    base_title = style.scaled(style_scale(cell_w, cell_h))
    title_style = (
        base_title.merge(
            font_size=base_title.font_size * _TITLE_SCALE,
            font_weight=_TITLE_WEIGHT,
        )
        if emphasize_titles
        else base_title
    )
    title_pad = _TITLE_PAD * style_scale(cell_w, cell_h)
    wrapped = _wrap_titles(titles, len(images), cell_w, title_style, title_pad)
    # Row height from the tallest actual line (mono runs have a taller ink box),
    # falling back to a plain measurement when there are no titles.
    line_h = max(
        (
            _MEASURE.measure_spans(line, title_style.font_size).height
            for lines in wrapped
            for line in lines
        ),
        default=_MEASURE.measure_text(
            "Ag", title_style.font_size, weight=title_style.font_weight
        ).height,
    )
    max_lines = max((len(lines) for lines in wrapped), default=0)
    title_h = max_lines * line_h + 2 * title_pad if max_lines else 0.0

    width = pad + cols * (cell_w + pad)
    height = round(pad + rows * (title_h + cell_h + pad))
    background = Color.parse(bg)
    # Titles use the on-brand heading color for the background (deep purple on a
    # light grid, near-white on a dark one) rather than plain black/white.
    text_color = brand.chrome_for(background).card_title

    cells: list[tuple[int, int]] = []  # (cell_x, cell_y) per tile
    placements: list[tuple[int, int, int, int]] = []
    for i, (w, h) in enumerate(sizes):
        row, col = divmod(i, cols)
        # A short final row is centered rather than left-aligned: three tiles
        # over two columns reads as a deliberate arrangement that way, where a
        # lone bottom-left tile reads as something missing from the corner.
        in_row = min(cols, len(sizes) - row * cols)
        row_offset = ((cols - in_row) * (cell_w + pad)) // 2
        cell_x = pad + row_offset + col * (cell_w + pad)
        cell_y = round(pad + row * (title_h + cell_h + pad))
        cells.append((cell_x, cell_y))
        x = cell_x + (cell_w - w) // 2
        y = round(cell_y + title_h) + (cell_h - h) // 2
        placements.append((x, y, w, h))

    def paint(
        canvas: Canvas,
        environment: RenderEnvironment,
        capture: InteractionCapture | None,
    ) -> None:
        canvas.rounded_rect(Rect(0, 0, width, height), 0.0, fill=background)
        for i, img in enumerate(images):
            cell_x, cell_y = cells[i]
            if wrapped[i]:
                _draw_title(
                    canvas,
                    wrapped[i],
                    cell_x + cell_w / 2,
                    cell_y,
                    title_style,
                    text_color,
                    line_h,
                    title_pad,
                )
            x, y, w, h = placements[i]
            img._draw_onto(
                canvas,
                x,
                y,
                (w, h),
                environment=environment,
                capture=capture,
            )

    return Composite((width, height), paint), placements


def _draw_title(
    canvas: Canvas,
    lines: Sequence[list[Span]],
    center_x: float,
    top: float,
    style: Style,
    color: Color,
    line_h: float,
    pad: float,
) -> None:
    """Draw centered, wrapped title lines (styled spans) above a cell."""
    y = top + pad
    for line in lines:
        metrics = canvas.measure_spans(line, style.font_size)
        canvas.draw_spans(
            (center_x - metrics.width / 2, y + metrics.ascent),
            line,
            size=style.font_size,
            color=color,
        )
        y += line_h
