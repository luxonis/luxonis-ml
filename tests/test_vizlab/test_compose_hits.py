"""Tests for hit-map threading through compose (`grid_hits`/`combine_hits`)."""

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    BBox,
    Frame,
    Image,
    Tooltip,
    combine_hits,
    fit_grid,
    grid_hits,
)
from luxonis_ml.vizlab.compose import grid_placed
from luxonis_ml.vizlab.hitmap import HitMap


def _split(frame: Frame) -> tuple[Image, HitMap]:
    """Unpack a `Frame` into its typed ``(image, hitmap)`` parts."""
    return frame.image, frame.hitmap


def _tile(title: str, h: int = 60, w: int = 100) -> tuple[Image, Tooltip]:
    tip = Tooltip(title=title)
    image = Image(np.zeros((h, w, 3), np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.8, h=0.8, tooltip=tip)
    )
    return image, tip


def _titles(hits: HitMap) -> set[str | None]:
    return {tooltip.title for _, tooltip in hits.items}


def _within(hits: HitMap, image: Image) -> bool:
    return all(
        r.left >= 0
        and r.right <= image.width
        and r.top >= 0
        and r.bottom <= image.height
        for r, _ in hits.items
    )


def test_grid_hits_maps_each_tile_to_composite_pixels() -> None:
    a, tip_a = _tile("A")
    b, tip_b = _tile("B")
    # Uniform cells (tile == cell), so tiles land at exact, un-centered offsets:
    # tile0 at (10, 10), tile1 at (120, 10) on a 230x80 composite.
    composite, hits = _split(grid_hits([a, b], ncols=2, pad=10))
    assert (composite.width, composite.height) == (230, 80)
    assert len(hits.items) == 2
    assert hits.hit(10 + 50, 10 + 30) is tip_a
    assert hits.hit(120 + 50, 10 + 30) is tip_b


def test_combine_hits_threads_through_nesting() -> None:
    a, _ = _tile("A")
    b, _ = _tile("B")
    c, _ = _tile("C")
    composite, hits = _split(combine_hits(a, [b, c]))
    assert _titles(hits) == {"A", "B", "C"}
    assert len(hits.items) == 3
    assert _within(hits, composite)


def test_combine_hits_titled_mapping() -> None:
    a, _ = _tile("A")
    b, _ = _tile("B")
    composite, hits = _split(combine_hits({"left": a, "right": b}))
    assert _titles(hits) == {"A", "B"}
    assert _within(hits, composite)


def test_combine_hits_single_group_returns_copy() -> None:
    a, _ = _tile("A")
    composite, hits = _split(combine_hits(a))
    assert composite is not a
    assert _titles(hits) == {"A"}


def test_fit_grid_matches_grid_hits_when_unscaled() -> None:
    a, _ = _tile("A")
    b, _ = _tile("B")
    fit_img, fit_hits = _split(
        fit_grid([a, b], target=(10_000, 10_000), ncols=2)
    )
    grid_img, _ = _split(grid_hits([a, b], ncols=2))
    assert (fit_img.width, fit_img.height) == (grid_img.width, grid_img.height)
    assert hits_title(fit_hits, 10 + 50, 10 + 30) == "A"


def test_fit_grid_scales_within_target() -> None:
    a, _ = _tile("A")
    b, _ = _tile("B")
    composite, hits = _split(fit_grid([a, b], target=(150, 100), ncols=2))
    assert composite.width <= 150
    assert composite.height <= 100
    assert len(hits.items) == 2
    assert _within(hits, composite)


def hits_title(hits: HitMap, x: float, y: float) -> str | None:
    tooltip = hits.hit(x, y)
    return tooltip.title if tooltip is not None else None


@pytest.mark.parametrize("seed", range(30))
def test_grid_hits_round_trips_random_layouts(seed: int) -> None:
    """For any random grid, hovering a tile's box returns that tile's tooltip.

    `grid_placed` and `grid_hits` share ``_grid`` with identical arguments, so
    each tile's raster lands at the same placement in both; the invariant is that
    the composed hit map, queried at a placement's center, resolves to the box
    drawn in that tile — proving the per-tile offset is exact.
    """
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, 6))
    tiles: list[Image] = []
    tips: list[Tooltip] = []
    for i in range(count):
        height = int(rng.integers(30, 120))
        width = int(rng.integers(30, 120))
        tip = Tooltip(title=f"tile-{i}")
        tips.append(tip)
        tiles.append(
            Image(np.zeros((height, width, 3), np.uint8)).add(
                # A centered box, so the tile's center pixel is inside its region.
                BBox(x=0.25, y=0.25, w=0.5, h=0.5, tooltip=tip)
            )
        )
    ncols = int(rng.integers(1, count + 1))
    _, hits = _split(grid_hits(tiles, ncols=ncols))
    _, placements = grid_placed(tiles, ncols=ncols)

    assert len(hits.items) == count
    for (x, y, w, h), tip in zip(placements, tips, strict=True):
        assert hits.hit(x + w / 2, y + h / 2) is tip
