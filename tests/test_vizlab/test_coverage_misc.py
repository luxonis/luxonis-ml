"""Targeted tests for the remaining uncovered branches across small modules."""

from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    BBox,
    Caption,
    Classification,
    Color,
    Corner,
    Image,
    Palette,
    Tooltip,
    vstack,
)
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.annotations.layout import LabelLayout, label_candidates
from luxonis_ml.vizlab.geometry import Rect, bounding_rect
from luxonis_ml.vizlab.interaction.maps import InteractionCapture
from luxonis_ml.vizlab.layout.compose import grid
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.style import LabelPlacement


def test_color_saturate_both_directions() -> None:
    base = Color(100, 150, 200)
    assert base.saturate(0.5).hls[2] > base.hls[2]
    assert base.saturate(-0.5).hls[2] < base.hls[2]


def test_color_tuple_and_hex_errors() -> None:
    assert Color.parse((1, 2, 3, 4)) == Color(1, 2, 3, 4)
    with pytest.raises(ValueError, match="3 or 4 elements"):
        Color.parse((1, 2))
    # A bare int is a grayscale color; a bad hex still reports as hex.
    assert Color.parse(123) == Color(123, 123, 123)
    with pytest.raises(ValueError, match="invalid hex color"):
        Color.parse("#gghhii")  # right length, non-hex digits


def test_bounding_rect_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        bounding_rect([])


def test_rect_union_contains_both_rectangles() -> None:
    assert Rect(1.0, 4.0, 5.0, 8.0).union(Rect(-2.0, 6.0, 3.0, 10.0)) == Rect(
        -2.0, 4.0, 5.0, 10.0
    )


def test_palette_at_is_deterministic() -> None:
    assert Palette().at(3) == Palette().at(3)


def test_bbox_extent_is_none_for_normalized() -> None:
    # LDF boxes are always normalized, so the pixel extent is unknown here.
    assert BBox(x=0.1, y=0.1, w=0.5, h=0.5).extent() is None


def test_label_candidates_inside_first() -> None:
    cands = label_candidates(
        Rect(0.0, 0.0, 20.0, 20.0), 5.0, 5.0, LabelPlacement.INSIDE
    )
    assert cands[0] == (0.0, 0.0)


def test_base_reserve_is_noop_for_spatial() -> None:
    ctx = RenderContext(
        canvas=Canvas.blank(10, 10), layout=LabelLayout(10, 10)
    )
    assert BBox(x=0.1, y=0.1, w=0.4, h=0.4).reserve(ctx) is None
    assert ctx.layout is not None
    assert ctx.layout.placed == []


def test_render_context_hit_capture() -> None:
    capture = InteractionCapture()
    tooltip = Tooltip(title="car")
    ctx = RenderContext(canvas=Canvas.blank(10, 10), capture=capture)
    region = Rect(1.0, 2.0, 3.0, 4.0)

    ctx.emit_hit(region, None)
    ctx.emit_hit(region, tooltip)

    assert capture.hover == [(region, tooltip)]


def test_classification_edge_cases() -> None:
    assert Classification(tags=[]).extent() is None
    # No layout -> reserve is a no-op; empty/blank tags produce no chips.
    ctx = RenderContext(canvas=Canvas.blank(40, 40))
    Classification(tags=[""]).reserve(ctx)  # layout is None branch
    base = Image(np.zeros((40, 40, 3), np.uint8))
    plain = base.copy().render()
    assert np.array_equal(
        base.copy().add(Classification(tags=[])).render(), plain
    )
    # A blank tag string is skipped (no chip drawn).
    assert np.array_equal(
        base.copy().add(Classification(tags=[""])).render(), plain
    )


def test_compose_vstack_and_empty_grid() -> None:
    cells = [Image(np.zeros((6, 8, 3), np.uint8)) for _ in range(2)]
    assert vstack(cells, pad=3).render().shape[1] == 8 + 2 * 3
    with pytest.raises(ValueError, match="empty sequence"):
        grid([])


def test_image_output_methods(tmp_path: Path) -> None:
    img = Image(np.full((4, 6, 3), 50, np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.8, h=0.6, color="#4c8dff")
    )
    assert img.to_numpy("bgr").shape == (4, 6, 3)
    assert img.to_pil().mode == "RGBA"
    assert img.base_rgba().shape == (4, 6, 4)
    out = tmp_path / "out.png"
    assert img.save(out) is img
    assert out.exists()
    assert "Image(size=6x4" in repr(img)


def test_image_copy_is_independent() -> None:
    original = Image(np.zeros((4, 4, 3), np.uint8)).add(
        BBox(x=0.0, y=0.0, w=1.0, h=1.0)
    )
    clone = original.copy()
    clone.add(BBox(x=0.1, y=0.1, w=0.3, h=0.3))
    assert len(original.annotations) == 1
    assert len(clone.annotations) == 2


def test_image_show_uses_pil(monkeypatch: pytest.MonkeyPatch) -> None:
    shown = {}

    class _Dummy:
        def show(self) -> None:
            shown["called"] = True

    monkeypatch.setattr(Image, "to_pil", lambda self: _Dummy())
    Image(np.zeros((2, 2, 3), np.uint8)).show()
    assert shown["called"]


def test_label_layout_grows_and_clamps() -> None:
    layout = LabelLayout(20, 10)
    for index in range(9):
        layout._record(Rect(float(index), 0.0, float(index + 1), 1.0))

    assert len(layout.placed) == 9
    assert layout._clamp(-5.0, 20.0, 4.0, 3.0) == Rect(0.0, 7.0, 4.0, 10.0)


def test_corner_stack_empty_size_and_nonoverlapping_reservation() -> None:
    caption = Caption(text="", corner=Corner.TOP_LEFT)
    assert caption.content_size() == (0.0, 0.0)

    populated = Caption(text="hello", corner=Corner.TOP_LEFT)
    ctx = RenderContext(canvas=Canvas.blank(200, 100))
    positioned = populated._positioned(ctx, populated.resolve_style(ctx))
    reserved = [Rect(180.0, 0.0, 200.0, 20.0)]
    assert populated._avoid_reserved(positioned, reserved) == positioned
