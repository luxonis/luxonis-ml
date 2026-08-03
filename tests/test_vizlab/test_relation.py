"""Coverage for `Arrow`: endpoint resolution, curvature, heads, and targets."""

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    Arrow,
    BBox,
    Image,
    Keypoints,
    Mask,
    Palette,
    Polyline,
    Rect,
    Tooltip,
    grid,
)
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.render.context import layer_scope
from luxonis_ml.vizlab.scene.html import draws_anything


def _canvas(w: int = 200, h: int = 120) -> Image:
    return Image(np.full((h, w, 3), 30, np.uint8))


def _ctx() -> RenderContext:
    return RenderContext(canvas=Canvas.blank(2, 2))


def _drawn(image: Image) -> int:
    base = Image(image.base_rgba()).render()
    return int((np.abs(image.render().astype(int) - base).sum(-1) > 8).sum())


def _boxes() -> tuple[BBox, BBox]:
    return (
        BBox(x=0.0, y=0.4, w=0.2, h=0.2, label="car"),
        BBox(x=0.7, y=0.4, w=0.2, h=0.2, label="plate"),
    )


# --- endpoint resolution ----------------------------------------------------


def test_annotation_endpoints_land_on_the_facing_edges() -> None:
    left, right = _boxes()
    arrow = Arrow(start=left, end=right)
    # The boxes' inner edges (x = 20 and x = 70), plus the default 4px gap.
    assert arrow._endpoints(100, 100) == ((24.0, 50.0), (66.0, 50.0))
    # The same annotation on a canvas of another size scales with it.
    assert arrow._endpoints(200, 200) == ((44.0, 100.0), (136.0, 100.0))


def test_endpoints_resolve_late_so_a_moved_box_moves_the_arrow() -> None:
    left, right = _boxes()
    arrow = Arrow(start=left, end=right)
    image = _canvas().add(left).add(right).add(arrow)
    before = image.render()
    right.x = 0.4
    moved = arrow._endpoints(100, 100)
    assert moved is not None
    assert moved[1] == (36.0, 50.0)
    # The cached render tracks the scene graph, references included.
    assert not np.array_equal(before, image.render())


def test_point_endpoints_are_taken_literally() -> None:
    arrow = Arrow(start=(0.25, 0.5), end=(0.75, 0.5), gap=20.0)
    # An empty rectangle has no edge to stand off from, so no gap is applied.
    assert arrow._endpoints(100, 100) == ((25.0, 50.0), (75.0, 50.0))


def test_mixed_endpoints_only_stand_off_from_the_shape() -> None:
    left, _ = _boxes()
    arrow = Arrow(start=left, end=(0.9, 0.5))
    ends = arrow._endpoints(100, 100)
    assert ends is not None
    assert ends[0] == (24.0, 50.0)
    assert ends[1] == (90.0, 50.0)


@pytest.mark.parametrize(
    "target",
    [
        Mask(mask=np.zeros((8, 8), np.uint8)),  # type: ignore[call-arg]
        Keypoints(keypoints=[(0.1, 0.1, 0)]),  # every joint invisible
        Polyline(points=[]),
    ],
)
def test_an_endpoint_with_no_bounds_draws_nothing(target: object) -> None:
    arrow = Arrow(start=target, end=(0.9, 0.9))  # type: ignore[arg-type]
    assert arrow._endpoints(100, 100) is None
    assert _drawn(_canvas().add(arrow)) == 0


def test_coincident_endpoints_draw_nothing() -> None:
    arrow = Arrow(start=(0.5, 0.5), end=(0.5, 0.5))
    assert arrow._endpoints(100, 100) is None
    assert _drawn(_canvas().add(arrow)) == 0


def test_any_annotation_can_be_an_anchor() -> None:
    line = Polyline(points=[(0.1, 0.1), (0.3, 0.3)])
    keypoints = Keypoints(keypoints=[(0.7, 0.7, 2), (0.9, 0.9, 2)])
    arrow = Arrow(start=line, end=keypoints)
    ends = arrow._endpoints(100, 100)
    assert ends is not None
    assert line.region_at(100, 100) == Rect(10.0, 10.0, 30.0, 30.0)
    assert keypoints.region_at(100, 100) == Rect(70.0, 70.0, 90.0, 90.0)


# --- shape ------------------------------------------------------------------


def test_curvature_bows_the_path_and_its_sign_picks_the_side() -> None:
    left, right = _boxes()
    straight = Arrow(start=left, end=right)._path(100, 100)
    assert straight == [(24.0, 50.0), (66.0, 50.0)]  # two points, no sampling

    def middle(curvature: float) -> tuple[float, float]:
        path = Arrow(start=left, end=right, curvature=curvature)._path(
            100, 100
        )
        assert path is not None
        return path[len(path) // 2]

    assert middle(0.3)[1] > 50.0
    assert middle(-0.3)[1] < 50.0
    # Two relations between the same pair stay apart instead of overlapping.
    assert middle(0.3) != middle(-0.3)


def test_a_curved_arrow_leaves_along_its_own_tangent() -> None:
    left, right = _boxes()
    bowed = Arrow(start=left, end=right, curvature=0.4)._endpoints(100, 100)
    assert bowed is not None
    # The straight arrow leaves the box's right edge; the bowed one leaves
    # lower down, aimed at the control point rather than along the chord.
    assert bowed[0][1] > 50.0


@pytest.mark.parametrize(
    ("heads", "count"),
    [("end", 1), ("start", 1), ("both", 2), ("none", 0)],
)
def test_head_modes_draw_the_right_number_of_heads(
    heads: str, count: int
) -> None:
    drawn: list[int] = []
    real = Canvas.polygon

    def record(self: Canvas, points: list, **kwargs: object) -> None:
        if kwargs.get("fill") is not None and len(points) == 3:
            drawn.append(1)
        return real(self, points, **kwargs)  # type: ignore[arg-type]

    left, right = _boxes()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Canvas, "polygon", record)
        _canvas().add(Arrow(start=left, end=right, heads=heads)).render()  # type: ignore[arg-type]
    assert len(drawn) == count


def test_gap_pushes_the_tip_clear_of_the_shape() -> None:
    left, right = _boxes()
    tight = Arrow(start=left, end=right, gap=0.0)._endpoints(100, 100)
    wide = Arrow(start=left, end=right, gap=10.0)._endpoints(100, 100)
    assert tight is not None
    assert wide is not None
    assert tight[0] == (20.0, 50.0)  # exactly on the edge
    assert wide[0] == (30.0, 50.0)


# --- label, hover, and style ------------------------------------------------


def test_the_mid_label_is_anchored_halfway_along_the_arrow() -> None:
    # Regression: a straight arrow is two sampled points, so taking the middle
    # index put its "mid" label on the arrow's tip instead of between the ends.
    left, right = _boxes()
    anchors: list[Rect] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "luxonis_ml.vizlab.annotations.relation.place_label",
            lambda _ctx, region, *_rest: anchors.append(region),
        )
        for curvature in (0.0, 0.3):
            image = _canvas(100, 100)
            image.add(Arrow(start=left, end=right, curvature=curvature))
            image.render()
    straight, bowed = anchors
    assert straight.center == (45.0, 50.0)  # halfway between 24 and 66
    assert 24.0 < bowed.center[0] < 66.0  # and the curve's halfway point
    assert bowed.center[1] > 50.0  # which is off the chord it bows from


def test_mid_label_is_placed_by_the_shared_layout() -> None:
    left, right = _boxes()
    plain = _drawn(_canvas().add(Arrow(start=left, end=right)))
    labelled = _drawn(
        _canvas().add(Arrow(start=left, end=right, label="tows"))
    )
    assert labelled > plain
    # Markup in the label is parsed rather than drawn.
    tagged = _canvas().add(Arrow(start=left, end=right, label="<b>tows</b>"))
    assert not np.array_equal(
        tagged.render(),
        _canvas().add(Arrow(start=left, end=right, label="tows")).render(),
    )


def test_tooltip_region_follows_the_arrow() -> None:
    left, right = _boxes()
    arrow = Arrow(start=left, end=right, tooltip=Tooltip(title="rel"))
    _, hits = _canvas().add(arrow).render_hits()
    assert len(hits.items) == 1
    assert hits.hit(100.0, 60.0) is not None  # on the run
    assert hits.hit(5.0, 5.0) is None


def test_color_and_style_resolve_like_any_annotation() -> None:
    ctx = _ctx()
    palette = Palette()
    labelled = Arrow(start=(0, 0), end=(1, 1), label="tows", palette=palette)
    assert labelled.resolve_color(ctx) == palette.color_for("tows")
    assert Arrow(start=(0, 0), end=(1, 1), color="#ff0000").resolve_color(
        ctx
    ) == Color(255, 0, 0)
    assert (
        Arrow(start=(0, 0), end=(1, 1))
        .styled(stroke_width=7.0)
        .resolve_style(ctx)
        .stroke_width
        == 7.0
    )
    left, right = _boxes()
    thin = _drawn(
        _canvas().add(Arrow(start=left, end=right).styled(stroke_width=1.0))
    )
    thick = _drawn(
        _canvas().add(Arrow(start=left, end=right).styled(stroke_width=8.0))
    )
    assert thick > thin


# --- render targets and composition -----------------------------------------


def test_arrow_is_vector_in_svg_and_lives_on_its_own_layer() -> None:
    left, right = _boxes()
    image = _canvas().add(left).add(right).add(Arrow(start=left, end=right))
    with layer_scope({"relation"}, chrome=False, labels=False):
        marks = image.render_svg()
    assert draws_anything(marks)
    assert b"<image" not in marks  # real geometry, not a rasterized stand-in
    with layer_scope({"keypoint"}, chrome=False, labels=False):
        assert not draws_anything(image.render_svg())


def test_arrow_survives_composition() -> None:
    left, right = _boxes()
    image = _canvas().add(left).add(right)
    image.add(Arrow(start=left, end=right, tooltip=Tooltip(title="rel")))
    stacked = grid([image, image.copy()], ncols=2)
    _, hits = stacked.render_hits()
    assert len(hits.items) == 2
    assert len(stacked.render_svg()) > 0
