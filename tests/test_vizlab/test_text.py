"""Coverage for Caption and Legend overlays."""

import numpy as np

from luxonis_ml.vizlab import BBox, Caption, Corner, Image, Legend


def _img() -> Image:
    return Image(np.full((120, 200, 3), 20, np.uint8))


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
