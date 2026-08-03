"""Coverage for `Polyline`: geometry, per-vertex look, labels, and targets."""

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    BBox,
    Image,
    Palette,
    Polyline,
    Rect,
    Style,
    Tooltip,
    grid,
    with_panel,
)
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.annotations.polyline import _at_distance
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.render.context import layer_scope
from luxonis_ml.vizlab.scene.html import draws_anything

LANE = [(0.1, 0.8), (0.5, 0.5), (0.9, 0.2)]


def _canvas(w: int = 200, h: int = 120) -> Image:
    return Image(np.full((h, w, 3), 30, np.uint8))


def _ctx() -> RenderContext:
    return RenderContext(canvas=Canvas.blank(2, 2))


def _drawn(image: Image, size: tuple[int, int] | None = None) -> int:
    """Count pixels the annotations changed against the bare base."""
    base = Image(image.base_rgba()).render(size)
    return int(
        (np.abs(image.render(size).astype(int) - base).sum(-1) > 8).sum()
    )


# --- geometry ---------------------------------------------------------------


def test_points_resolve_against_the_canvas_not_the_source() -> None:
    line = Polyline(points=LANE)
    assert line._pixels(200, 100)[0] == (20.0, 80.0)
    # The same normalized annotation on a canvas of another size.
    assert line._pixels(400, 200)[0] == (40.0, 160.0)


def test_region_at_bounds_the_vertices() -> None:
    assert Polyline(points=LANE).region_at(100, 100) == Rect(
        10.0, 20.0, 90.0, 80.0
    )
    assert Polyline(points=[]).region_at(100, 100) is None
    # Normalized annotations have no extent until a canvas size is known.
    assert Polyline(points=LANE).extent() is None


def test_closed_run_returns_to_the_first_vertex() -> None:
    ring = Polyline(points=LANE, closed=True)
    run = ring._run(ring._pixels(100, 100))
    assert len(run) == 4
    assert run[-1] == run[0]
    # Two points cannot enclose anything, so closing them is left alone.
    pair = Polyline(points=LANE[:2], closed=True)
    assert len(pair._run(pair._pixels(100, 100))) == 2


def test_at_distance_walks_the_arc_length() -> None:
    run = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    lengths = [0.0, 10.0, 20.0]
    assert _at_distance(run, lengths, 5.0) == ((5.0, 0.0), (10.0, 0.0))
    assert _at_distance(run, lengths, 15.0) == ((10.0, 5.0), (0.0, 10.0))
    # The far end lands exactly on the last vertex.
    assert _at_distance(run, lengths, 20.0)[0] == (10.0, 10.0)


# --- drawing ----------------------------------------------------------------


def test_polyline_draws_and_degenerate_ones_do_not() -> None:
    assert _drawn(_canvas().add(Polyline(points=LANE))) > 0
    assert _drawn(_canvas().add(Polyline(points=[(0.5, 0.5)]))) == 0
    assert _drawn(_canvas().add(Polyline(points=[]))) == 0


def test_fill_only_applies_to_a_closed_ring() -> None:
    square = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
    interior = (60, 100)  # well inside the ring, away from its stroke

    def center(**kwargs: object) -> np.ndarray:
        image = _canvas().add(Polyline(points=square, **kwargs))  # type: ignore[arg-type]
        return image.render()[interior]

    bare = _canvas().render()[interior]
    assert np.array_equal(center(closed=True), bare)  # unfilled by default
    assert not np.array_equal(center(closed=True, fill=True), bare)
    # An open run has no interior to fill, so the flag changes nothing.
    assert np.array_equal(center(fill=True), bare)


def test_uniform_stroke_is_one_path_and_per_vertex_widths_are_not() -> None:
    calls: list[dict] = []
    real = Canvas.polygon

    def record(self: Canvas, points: list, **kwargs: object) -> None:
        calls.append({"points": len(points), **kwargs})
        return real(self, points, **kwargs)  # type: ignore[arg-type]

    def strokes(**kwargs: object) -> list[dict]:
        calls.clear()
        _canvas().add(Polyline(points=LANE, **kwargs)).render()  # type: ignore[arg-type]
        return [c for c in calls if c.get("stroke") is not None]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Canvas, "polygon", record)
        assert len(strokes()) == 1  # one path for the whole run
        tapered = strokes(widths=[0.5, 1.0, 2.0])
    assert len(tapered) == 2  # one per segment
    # The segment widths are the means of their endpoints' multipliers.
    base = Style().stroke_width
    assert [round(c["stroke_width"] / base, 3) for c in tapered] == [0.75, 1.5]


def test_values_color_each_segment_through_the_gradient() -> None:
    seen: list[tuple[Color, Color]] = []

    def record(
        self: Canvas,
        p1: object,
        p2: object,
        c1: Color,
        c2: Color,
        **kw: object,
    ) -> None:
        seen.append((c1, c2))

    line = Polyline(points=LANE, values=[0.0, 0.5, 1.0], gradient="viridis")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Canvas, "gradient_line", record)
        _canvas().add(line).render()
    assert len(seen) == 2  # one gradient stroke per segment
    assert seen[0][1] == seen[1][0]  # the shared vertex has one color
    assert seen[0][0] != seen[1][1]  # the ends differ
    # The palette is not consulted at all while values are colouring the run.
    assert seen[0][0] != Palette().color_for("lane")


def test_pinned_value_range_puts_two_lines_on_one_scale() -> None:
    def head_color(**kwargs: object) -> Color:
        line = Polyline(points=LANE, gradient="viridis", **kwargs)  # type: ignore[arg-type]
        colors = line._vertex_colors(_ctx(), 3)
        assert colors is not None
        return colors[0]

    # Unpinned, each line normalizes over its own range: both start cold.
    assert head_color(values=[0.0, 1.0, 2.0]) == head_color(
        values=[10.0, 11.0, 12.0]
    )
    # Pinned, the second line's values sit at the top of the shared range.
    assert head_color(
        values=[0.0, 1.0, 2.0], vmin=0.0, vmax=12.0
    ) != head_color(values=[10.0, 11.0, 12.0], vmin=0.0, vmax=12.0)


def test_arrows_draw_one_chevron_each() -> None:
    heads: list[int] = []
    real = Canvas.polygon

    def record(self: Canvas, points: list, **kwargs: object) -> None:
        if kwargs.get("fill") is not None and len(points) == 3:
            heads.append(len(points))
        return real(self, points, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Canvas, "polygon", record)
        for count in (0, 1, 3):
            heads.clear()
            _canvas().add(Polyline(points=LANE, arrows=count)).render()
            assert len(heads) == count


def test_a_run_of_zero_length_draws_no_chevron() -> None:
    # All vertices coincide, so there is no direction to point in. The stroke
    # still leaves its round cap; only the chevrons are skipped.
    coincident = [(0.5, 0.5)] * 3
    assert _drawn(_canvas().add(Polyline(points=coincident, arrows=2))) == (
        _drawn(_canvas().add(Polyline(points=coincident)))
    )


# --- labels, hover, and style resolution ------------------------------------


def test_label_chip_is_placed_by_the_shared_layout() -> None:
    line = Polyline(points=LANE, label="lane")
    plain = _drawn(_canvas().add(Polyline(points=LANE)))
    assert _drawn(_canvas().add(line)) > plain  # the chip adds pixels
    # An open run anchors its chip at the first vertex, not at its bounds.
    anchor = line._anchor_region(200, 120, Style())
    assert anchor.center == (20.0, 96.0)
    # A ring is labeled like a box: against the whole shape.
    ring = Polyline(points=LANE, closed=True, label="zone")
    assert ring._anchor_region(200, 120, Style()) == Rect(
        20.0, 24.0, 180.0, 96.0
    )


def test_markup_in_the_label_is_parsed_not_drawn() -> None:
    def render(label: str) -> np.ndarray:
        return _canvas().add(Polyline(points=LANE, label=label)).render()

    plain, tagged = render("lane"), render("<b>lane</b>")

    def spread(image: np.ndarray) -> int:
        return int((np.abs(image.astype(int) - plain).sum(-1) > 8).sum())

    assert not np.array_equal(plain, tagged)  # the tag took effect
    # A recognized tag is consumed; eleven literal characters are drawn.
    assert spread(render("<zz>lane</zz>")) > spread(tagged)


def test_tooltip_region_covers_the_run() -> None:
    line = Polyline(points=LANE, tooltip=Tooltip(title="lane"))
    _, hits = _canvas().add(line).render_hits()
    assert len(hits.items) == 1
    region, _ = hits.items[0]
    # The vertex bounds, padded by the stroke so a hover near the line counts.
    assert region.left < 20.0
    assert region.right > 180.0
    assert hits.hit(100.0, 60.0) is not None
    assert hits.hit(199.0, 119.0) is None


def test_color_resolution_matches_a_box() -> None:
    palette = Palette()
    line = Polyline(points=LANE, label="lane", palette=palette)
    box = BBox(x=0, y=0, w=1, h=1, label="lane", palette=palette)
    ctx = _ctx()
    assert line.resolve_color(ctx) == box.resolve_color(ctx)  # type: ignore[arg-type]
    # An explicit override still wins.
    pinned = Polyline(points=LANE, label="lane", color="#ff0000")
    assert pinned.resolve_color(ctx) == Color(255, 0, 0)  # type: ignore[arg-type]


def test_nested_polyline_derives_the_parent_look() -> None:
    parent = BBox(x=0.1, y=0.1, w=0.8, h=0.8, label="road")
    child = Polyline(points=LANE)
    parent.add(child)
    image = _canvas().add(parent)
    assert _drawn(image) > 0
    ctx = _ctx()
    style = parent.resolve_style(ctx)  # type: ignore[arg-type]
    color = parent.resolve_color(ctx)  # type: ignore[arg-type]
    nested = ctx.descend(color, style)  # type: ignore[attr-defined]
    # An unlabeled child reads as part of its parent: derived color, dashed.
    assert child.resolve_color(nested) != color
    assert child.resolve_style(nested).dash is not None


def test_styled_overrides_reach_the_stroke() -> None:
    thin = _drawn(
        _canvas().add(Polyline(points=LANE).styled(stroke_width=1.0))
    )
    thick = _drawn(
        _canvas().add(Polyline(points=LANE).styled(stroke_width=9.0))
    )
    assert thick > thin


# --- render targets and composition -----------------------------------------


def test_polyline_is_vector_in_svg_and_lives_on_its_own_layer() -> None:
    image = _canvas().add(Polyline(points=LANE, label="lane"))
    with layer_scope({"polyline"}, chrome=False, labels=False):
        marks = image.render_svg()
    assert draws_anything(marks)
    # Real vector geometry, not a rasterized stand-in.
    assert b"<image" not in marks
    with layer_scope({"box"}, chrome=False, labels=False):
        assert not draws_anything(image.render_svg())


def test_polyline_survives_composition() -> None:
    image = _canvas().add(
        Polyline(points=LANE, label="lane", tooltip=Tooltip(title="lane"))
    )
    stacked = grid([image, image.copy()], ncols=2)
    _, hits = stacked.render_hits()
    assert len(hits.items) == 2  # one per tile, in composite coordinates
    assert stacked.render().shape[1] > image.render().shape[1]
    panelled = with_panel(image, {"lanes": 1})
    assert panelled.render().shape[2] == 4
    assert len(panelled.render_svg()) > 0


def test_render_size_scales_the_run_with_the_image() -> None:
    image = _canvas().add(Polyline(points=LANE))
    assert _drawn(image, (400, 240)) > _drawn(image)
