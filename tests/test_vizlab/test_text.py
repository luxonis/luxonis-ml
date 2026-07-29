"""Coverage for Caption and Legend overlays."""

from collections.abc import Sequence

import numpy as np
import pytest

from luxonis_ml.vizlab import BBox, Caption, Corner, Image, Legend
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.annotations.overlay import Cell
from luxonis_ml.vizlab.canvas import Canvas
from luxonis_ml.vizlab.markup import Span


def _img() -> Image:
    return Image(np.full((120, 200, 3), 20, np.uint8))


def _legend_cell(legend: Legend, width: int, height: int) -> Cell:
    """Build ``legend``'s single card cell on a ``width`` x ``height`` canvas."""
    ctx = RenderContext(canvas=Canvas.blank(width, height))
    (cell,) = legend._cells(ctx, legend.resolve_style(ctx))
    return cell


def test_caption_renders() -> None:
    base = _img()
    out = (
        base.copy()
        .add(Caption(text="frame 1", corner=Corner.BOTTOM_LEFT))
        .render()
    )
    assert not np.array_equal(out, base.copy().render())


def test_caption_title_and_empty() -> None:
    base = _img()
    plain = base.copy().render()
    titled = base.copy().add(Caption(text="Title", title=True)).render()
    assert not np.array_equal(titled, plain)
    assert np.array_equal(base.copy().add(Caption(text="")).render(), plain)


def test_caption_with_explicit_background_renders() -> None:
    base = _img()
    plain = base.copy().render()
    captioned = (
        base.copy().add(Caption(text="custom", background="#f5d142")).render()
    )
    assert not np.array_equal(captioned, plain)


def test_legend_palette_and_explicit_entries() -> None:
    base = _img()
    legend = Legend(entries=["car", ("road", "#3355aa")], title="classes")
    out = base.copy().add(legend).render()
    assert not np.array_equal(out, base.copy().render())


def test_legend_title_only_and_empty() -> None:
    base = _img()
    plain = base.copy().render()
    assert not np.array_equal(
        base.copy().add(Legend(entries=[], title="only")).render(), plain
    )
    assert np.array_equal(base.copy().add(Legend(entries=[])).render(), plain)


def test_legend_stays_single_column_when_it_fits() -> None:
    few = Legend(entries=["car", "person", "bus"], title="classes")
    narrow = _legend_cell(few, 800, 600)
    # A few classes fit in one column: the card is narrow (one swatch+name wide).
    assert narrow.width < 300


def test_many_class_legend_flows_into_columns_and_fits_the_canvas() -> None:
    names = [f"class_{i:02d}" for i in range(40)]
    legend = Legend(entries=names, title="classes")
    tall = _legend_cell(legend, 760, 520)
    # It never runs off the image, in either dimension...
    assert tall.height <= 520 - 2 * legend.margin + 1
    assert tall.width <= 760 - 2 * legend.margin + 1
    # ...and it used more than one column (wider than a single-column card would).
    single = _legend_cell(Legend(entries=["class_00"]), 760, 520)
    assert tall.width > single.width * 2


def test_many_class_legend_draws_column_swatches() -> None:
    names = [f"class_{index:02d}" for index in range(60)]
    image = Image(np.full((200, 760, 3), 20, np.uint8))
    rendered = image.add(Legend(entries=names, title="classes")).render()
    assert rendered.shape == (200, 760, 4)


def test_caption_skips_when_wrapping_produces_no_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_lines(
        self: Canvas,
        spans: Sequence[Span],
        size: float,
        *,
        max_width: float,
    ) -> list[list[Span]]:
        return []

    monkeypatch.setattr(Canvas, "wrap_spans", no_lines)
    base = _img()
    assert np.array_equal(
        base.copy().add(Caption(text="hidden")).render(),
        base.copy().render(),
    )


def test_legend_caps_with_overflow_when_even_columns_cannot_fit() -> None:
    names = [f"class_{i:02d}" for i in range(60)]
    legend = Legend(entries=names, title="classes")
    # A small canvas cannot hold 60 rows even in columns: it must still fit.
    cell = _legend_cell(legend, 320, 200)
    assert cell.height <= 200 - 2 * legend.margin + 1
    assert cell.width <= 320 - 2 * legend.margin + 1


def test_overlays_render_on_top_regardless_of_add_order() -> None:
    base = np.full((120, 200, 3), 20, np.uint8)

    def box() -> BBox:
        return BBox(x=0.02, y=0.03, w=0.96, h=0.94, color="#4c8dff")

    def cap() -> Caption:
        return Caption(text="hello", corner=Corner.TOP_LEFT)

    assert np.array_equal(
        Image(base).add(cap()).add(box()).render(),
        Image(base).add(box()).add(cap()).render(),
    )
