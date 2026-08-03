"""Coverage for the metadata sidebar panel."""

import numpy as np

from luxonis_ml.vizlab import BBox, Image, with_panel
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.layout.panel import (
    _MARGIN,
    _PAD,
    _PANEL_SIZE,
    _build_ops,
    _format_scalar,
    _format_sections,
    _format_tree,
    _metrics,
)
from luxonis_ml.vizlab.render import text_layout
from luxonis_ml.vizlab.style import Style


def test_value_ops_are_monospace_keys_are_not() -> None:
    """Values render in JetBrains Mono; keys/bullets stay in Inter (sans)."""
    m = _metrics(1.0, Color(24, 24, 28))
    lines = _format_tree({"speed": 12.4})
    ops, _ = _build_ops(lines, content_w=200.0, m=m)
    # An op carries styled spans, so the family lives on the runs themselves.
    by_text = {
        "".join(span.text for span in spans): all(span.mono for span in spans)
        for _y, _x, spans, _color in ops
    }
    assert by_text["speed: "] is False  # the key is sans
    assert by_text["12.4"] is True  # the value is mono


def _img(w: int = 120, h: int = 80) -> Image:
    return Image(np.full((h, w, 3), 30, np.uint8)).add(
        BBox(x=0.05, y=0.05, w=0.9, h=0.9, label="obj")
    )


def test_format_tree_nested_dict_list_scalar() -> None:
    lines = _format_tree({"a": 1, "b": [2, 3], "c": {"d": True}})
    assert lines == [
        (0, "a: ", True, "1"),
        (0, "b:", True, ""),
        (1, "• ", False, "2"),
        (1, "• ", False, "3"),
        (0, "c:", True, ""),
        (1, "d: ", True, "true"),
    ]


def test_format_tree_scalars_and_types() -> None:
    assert _format_tree(None) == [(0, "", False, "null")]
    assert _format_tree(3.14159)[0][3] == "3.142"
    assert _format_tree([{"x": 1}]) == [
        (0, "•", False, ""),
        (1, "x: ", True, "1"),
    ]


def test_format_scalar_unwraps_block() -> None:
    from luxonis_ml.vizlab import Block

    assert _format_scalar(Block("frame.jpg")) == "frame.jpg"


def test_format_sections_accepts_a_scalar_root() -> None:
    sections = _format_sections("ready")
    assert len(sections) == 1
    assert sections[0].heading is None
    assert sections[0].lines == [(0, "", False, "ready")]


def test_format_tree_truncates_long_string() -> None:
    body = _format_tree("z" * 800)[0][3]
    assert body.endswith("…")
    assert len(body) == 500


def test_wrap_splits_and_handles_empty() -> None:
    assert text_layout.wrap("", 100.0, 12.0) == [""]
    wrapped = text_layout.wrap("word " * 40, 80.0, 13.5)
    assert len(wrapped) > 1


def test_with_panel_right_widens_output() -> None:
    img = _img(120, 80)
    out = with_panel(img, {"source": "a.jpg", "split": "train"}, title="meta")
    rendered = out.render()
    assert rendered.shape[1] > 120  # widened by the panel
    assert rendered.shape[0] >= 80


def test_metrics_scale_proportionally() -> None:
    from luxonis_ml.vizlab.color import Color

    m = _metrics(2.0, Color(24, 24, 28))
    assert m.size == _PANEL_SIZE * 2.0
    assert m.pad == _PAD * 2.0


def test_metrics_colors_adapt_to_background() -> None:
    """Panel text is a light purple on dark and a deeper purple on light."""
    from luxonis_ml.vizlab.color import Color

    dark = _metrics(1.0, Color(24, 24, 28))
    light = _metrics(1.0, Color(240, 240, 244))
    # Both on-brand blue-purple (not plain white/black), lighter on the dark bg.
    assert dark.value.b > dark.value.r
    assert light.value.b > light.value.r
    assert sum(dark.value.rgb) > sum(light.value.rgb)


def test_panel_type_scales_with_image_size() -> None:
    """A larger image gets a larger (not shrinking) panel, so type stays legible."""
    data = {
        "source": "frame.jpg",
        "note": "a longer value that may wrap around",
    }
    small = with_panel(_img(200, 200), data).render()
    large = with_panel(_img(1600, 1600), data).render()
    small_panel_w = small.shape[1] - 200
    large_panel_w = large.shape[1] - 1600
    assert large_panel_w > small_panel_w


def test_with_panel_sides() -> None:
    img = _img(100, 60)
    data = {"aug": ["flip", "blur"], "tags": {"hard": True}}
    right = with_panel(img, data, side="right").render()
    left = with_panel(img, data, side="left").render()
    bottom = with_panel(img, data, side="bottom").render()
    assert right.shape[1] > 100
    assert left.shape[1] == right.shape[1]
    assert bottom.shape[0] > 60


def test_bottom_panel_places_image_above_the_panel() -> None:
    img = _img(100, 60)
    source = img.render()
    out = with_panel(img, {"aug": "flip"}, side="bottom").render()
    assert out.shape[0] > source.shape[0]  # panel added below the image
    # The image is placed unscaled at the top, offset by the outer margin
    # (style scale is 1.0 at this size, so the margin is exactly _MARGIN); an
    # interior pixel (clear of the rounded corners and border) survives intact.
    mgn = int(_MARGIN)
    assert np.array_equal(out[mgn + 30, mgn + 50], source[30, 50])


def test_with_panel_explicit_width_widens_by_that_width() -> None:
    img = _img(100, 60)
    narrow = img.with_panel({"k": "v"}, width=180.0).render().shape[1]
    wide = img.with_panel({"k": "v"}, width=240.0).render().shape[1]
    # The explicit width is respected: +60 px of panel -> +60 px of output,
    # independent of the (constant) outer margin and gap around it.
    assert wide - narrow == 60


def test_format_sections_groups_scalars_and_heads_containers() -> None:
    # Consecutive scalars share one heading-less section; each nested container
    # is its own section, headed by its key, so rules and labels mark groups.
    sections = _format_sections(
        {"a": 1, "b": 2, "arrays": {"x": 1}, "aug": ["flip"]}
    )
    assert [(s.heading, len(s.lines), s.block) for s in sections] == [
        (None, 2, False),
        ("arrays", 1, False),
        ("aug", 1, False),
    ]


def test_format_sections_makes_a_block_field_its_own_headed_section() -> None:
    from luxonis_ml.vizlab import Block

    sections = _format_sections({"id": 1, "path": Block("/a/b/img.jpg")})
    assert [(s.heading, s.block) for s in sections] == [
        (None, False),  # the scalar id
        ("path", True),  # the block field, headed and on its own line
    ]
    # The block's single body line is the bare value (no inline "key: " prefix).
    assert sections[1].lines == [(0, "", False, "/a/b/img.jpg")]


def test_nested_block_folds_its_key_and_value_onto_separate_lines() -> None:
    # A Block nested inside a container (e.g. a filename inside a batched
    # "sample N") folds like the top level: the label on one line, the value on
    # the next, rather than an inline "filename: a.jpg".
    from luxonis_ml.vizlab import Block

    lines = _format_tree({"filename": Block("a.jpg"), "id": 1})
    assert lines == [
        (0, "filename", True, ""),  # the label...
        (0, "", False, "a.jpg"),  # ...and the value, on its own line
        (0, "id: ", True, "1"),  # a plain scalar stays inline
    ]


def test_format_sections_handles_swatches_and_controls() -> None:
    from luxonis_ml.vizlab import Controls, Swatches

    sections = _format_sections(
        {
            "controls": Controls((("m", "masks", "on", True),)),
            "classes": Swatches((("#ff0000", "car"),)),
        }
    )
    assert sections[0].heading == "controls"
    assert sections[0].controls == (("m", "masks", "on", True),)
    assert sections[1].heading == "classes"
    # The swatch color is resolved to a Color, with an enabled flag, for drawing.
    assert sections[1].swatches == ((Color(255, 0, 0), "car", True),)


def test_format_sections_marks_disabled_swatches() -> None:
    from luxonis_ml.vizlab import Swatches

    sections = _format_sections(
        {
            "classes": Swatches(
                (("#f00", "car"), ("#0f0", "person")), frozenset({"car"})
            )
        }
    )
    swatches = sections[0].swatches
    assert swatches is not None
    # car is disabled (enabled=False); person stays enabled.
    assert [(label, enabled) for _c, label, enabled in swatches] == [
        ("car", False),
        ("person", True),
    ]


def test_disabled_swatches_render_hollow_and_struck_through() -> None:
    from luxonis_ml.vizlab import Swatches

    rendered = with_panel(
        _img(200, 140),
        {
            "classes": Swatches(
                ((Color(255, 0, 0), "car"),),
                frozenset({"car"}),
            )
        },
    ).render()
    assert rendered.shape[2] == 4


def test_empty_swatches_take_no_rows() -> None:
    from luxonis_ml.vizlab.layout.panel import _BodyLayout

    m = _metrics(1.0, Color(24, 24, 28))
    body = _BodyLayout(None, 0.0, 0.0, 200.0, m)
    assert body._swatches((), 17.0) == 17.0


def test_swatches_reserve_keeps_the_panel_width_stable() -> None:
    from luxonis_ml.vizlab import Swatches
    from luxonis_ml.vizlab.layout.panel import _auto_width

    m = _metrics(1.0, Color(24, 24, 28))

    def width(labels: list[str]) -> float:
        sections = _format_sections(
            {
                "classes": Swatches(
                    tuple((Color(1, 2, 3), lbl) for lbl in labels),
                    reserve="a_fairly_long_class_name",
                )
            }
        )
        return _auto_width(sections, None, m)

    # Different present classes, same reserved longest name -> identical width,
    # so the panel does not jump sample to sample.
    assert width(["car"]) == width(["bicycle", "person", "traffic light"])


def test_swatches_grid_packs_more_columns_into_a_wider_panel() -> None:
    from luxonis_ml.vizlab.layout.panel import _BodyLayout

    m = _metrics(1.0, Color(24, 24, 28))
    items = tuple((Color(200, 50, 50), "ab", True) for _ in range(6))
    narrow_body = _BodyLayout(None, 0.0, 0.0, 40.0, m)
    wide = _BodyLayout(None, 0.0, 0.0, 800.0, m)._swatches(items, 0.0)
    narrow = narrow_body._swatches(items, 0.0)
    # A wide panel fits several aligned columns (fewer rows -> shorter); a narrow
    # one falls to a single column (six rows).
    assert wide < narrow
    assert narrow == 6 * narrow_body.row_h


def test_swatches_legend_is_pinned_to_the_panel_bottom() -> None:
    from luxonis_ml.vizlab import Swatches

    # A tall image makes the panel card stretch well past its own content.
    img = Image(np.zeros((600, 220, 3), np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.3, h=0.3, label="obj")
    )
    out = with_panel(
        img, {"id": 1, "classes": Swatches(((Color(255, 0, 0), "car"),))}
    ).render()
    red_rows = np.where(
        np.all(out[..., :3] == (255, 0, 0), axis=2).any(axis=1)
    )[0]
    assert red_rows.size > 0  # the swatch drew
    # ...and it sits in the bottom of the composite, not up near the metadata.
    assert red_rows.min() > out.shape[0] * 0.6


def test_frame_with_panel_builds_a_clickmap_of_controls_and_swatches() -> None:
    from luxonis_ml.vizlab import Controls, Swatches
    from luxonis_ml.vizlab.interaction.frame import Frame

    framed = Frame(_img(200, 140)).with_panel(
        {
            "controls": Controls((("m", "masks", "on", True),)),
            "classes": Swatches(((Color(255, 0, 0), "car"),)),
        }
    )
    actions = {action for _rect, action in framed.clickmap.items}
    assert "key:m" in actions  # a control row is clickable
    assert "class:car" in actions  # a legend swatch is clickable
    assert "classes:toggle" in actions  # the legend's master on/off switch


def test_informational_swatches_do_not_register_class_actions() -> None:
    from luxonis_ml.vizlab import Swatches
    from luxonis_ml.vizlab.interaction.frame import Frame

    framed = Frame(_img(200, 140)).with_panel(
        {
            "tasks": Swatches(
                ((Color(255, 0, 0), "objects"),),
                interactive=False,
            )
        }
    )
    assert framed.clickmap.items == []


def test_panel_scene_frame_captures_child_hover_and_clicks() -> None:
    from luxonis_ml.vizlab import Controls, Tooltip

    image = Image(np.zeros((100, 160, 3), np.uint8)).add(
        BBox(
            x=0.1,
            y=0.1,
            w=0.5,
            h=0.5,
            tooltip=Tooltip(title="object"),
        )
    )
    frame = with_panel(
        image, {"controls": Controls((("m", "masks", "on", True),))}
    ).frame()

    assert frame.hitmap.items
    assert {action for _rect, action in frame.clickmap.items} == {"key:m"}


def test_frame_with_panel_preserves_existing_clickmap() -> None:
    from luxonis_ml.vizlab import Controls
    from luxonis_ml.vizlab.geometry import Rect
    from luxonis_ml.vizlab.interaction.frame import Frame
    from luxonis_ml.vizlab.render.capture import ClickMap

    frame = Frame(
        _img(200, 140),
        clickmap=ClickMap([(Rect(1, 2, 3, 4), "existing")]),
    ).with_panel({"controls": Controls((("m", "masks", "on", True),))})

    assert {action for _rect, action in frame.clickmap.items} == {
        "existing",
        "key:m",
    }


def test_with_panel_renders_controls_and_swatches() -> None:
    from luxonis_ml.vizlab import Controls, Swatches

    out = with_panel(
        _img(120, 80),
        {
            "controls": Controls(
                (("m", "masks", "on", True), ("b", "boxes", "off", False))
            ),
            "classes": Swatches(
                ((Color(200, 50, 50), "car"), (Color(50, 200, 50), "person"))
            ),
            "id": 1,
        },
    )
    assert out.render().shape[2] == 4  # renders without error, valid RGBA


def test_block_value_is_middle_ellipsized_when_too_long() -> None:
    from luxonis_ml.vizlab.layout.panel import _metrics

    m = _metrics(1.0, Color(24, 24, 28))
    long = "/datasets/coco/images/train2017/000000123456_aug.jpg"
    out = text_layout.middle_ellipsize(long, 120.0, m.size)
    assert out != long
    assert "…" in out
    assert out.startswith("/data")  # keeps the start ...
    assert out.endswith(".jpg")  # ... and the end (the extension)


def test_block_panel_renders_and_extreme_ellipsize_falls_back_to_mark() -> (
    None
):
    from luxonis_ml.vizlab import Block

    m = _metrics(1.0, Color(24, 24, 28))
    assert text_layout.middle_ellipsize("abcdef", 0.0, m.size) == "…"
    rendered = with_panel(
        _img(120, 80),
        {"path": Block("/datasets/coco/images/frame.jpg")},
    ).render()
    assert rendered.shape[2] == 4


def test_panel_frames_the_image_with_an_outer_margin() -> None:
    img = _img(120, 80)
    source = img.render()
    out = with_panel(img, {"a": 1}).render()
    # The image floats inside an outer margin now, so its top-left content is
    # pushed off the composite corner (which shows the background instead).
    assert not np.array_equal(out[0, 0], source[0, 0])
    mgn = int(_MARGIN)
    assert np.array_equal(out[mgn + 40, mgn + 60], source[40, 60])


def test_frame_with_panel_offsets_the_hitmap_to_match() -> None:
    from luxonis_ml.vizlab.geometry import Rect
    from luxonis_ml.vizlab.interaction.frame import Frame
    from luxonis_ml.vizlab.render.capture import HitMap
    from luxonis_ml.vizlab.tooltip import Tooltip

    img = _img(120, 80)
    hit = (Rect(10.0, 20.0, 30.0, 40.0), Tooltip(title="obj", rows=()))
    framed = Frame(img, HitMap([hit])).with_panel({"a": 1})
    ((rect, _tooltip),) = framed.hitmap.items
    # Shifted by the outer margin so hover still lands on the box (scale 1.0).
    assert (rect.left, rect.top) == (10.0 + _MARGIN, 20.0 + _MARGIN)


def test_hints_rows_are_hoverable_and_carry_their_detail() -> None:
    """A `Hints` row draws only its label and hovers to reveal its data."""
    from luxonis_ml.vizlab import Hints
    from luxonis_ml.vizlab.interaction.frame import Frame

    framed = Frame(_img(200, 140)).with_panel(
        {
            "augmentations": Hints(
                (
                    ("HorizontalFlip", {"p": 0.5}),
                    ("Affine", {"rotate": -12.345678}),
                )
            )
        }
    )
    assert [tip.title for _rect, tip in framed.hitmap.items] == [
        "HorizontalFlip",
        "Affine",
    ]
    rect, tooltip = framed.hitmap.items[0]
    # The region sits in the panel, right of the image, and hit-tests there.
    assert rect.left > 200
    mid_y = (rect.top + rect.bottom) / 2
    assert framed.hitmap.hit(rect.left + 1.0, mid_y) is tooltip
    assert framed.hitmap.hit(rect.left - 1.0, rect.top - 1.0) is None


def test_hints_detail_costs_the_panel_no_rows() -> None:
    """The detail is hover-only: the rows are the labels, nothing more."""
    from luxonis_ml.vizlab import Hints

    (section,) = _format_sections(
        {"augmentations": Hints((("Flip", {"p": 0.5}),))}
    )
    assert section.heading == "augmentations"
    assert section.hints == (("Flip", {"p": 0.5}),)
    assert section.lines == []  # the detail never becomes a row


def test_with_panel_wraps_long_value() -> None:
    img = _img(100, 60)
    out = with_panel(img, {"note": "lorem ipsum dolor sit amet " * 6})
    # Renders without error and stays a valid RGBA image.
    assert out.render().shape[2] == 4


def test_both_cards_cast_a_shadow() -> None:
    """The image and the panel float above the page rather than sitting flat."""
    image = Image(np.zeros((60, 80, 3), np.uint8))
    rendered = with_panel(image, {"frame": 3}).render()

    metrics = _metrics(1.0, Color(0, 0, 0))
    page = metrics.page
    # Sample the page a little below each card, where a drop shadow lands. The
    # margin is where the page shows through, so anything darker than the bare
    # page there is the shadow.
    height = rendered.shape[0]
    strip = rendered[height - 3, :, :3].astype(int)
    page_rgb = np.array([page.r, page.g, page.b], int)
    assert (strip.sum(axis=1) < page_rgb.sum()).any()


def test_the_shadow_scales_with_the_render() -> None:
    # Metrics are resolved per render size, so a large composite must not get a
    # hairline shadow sized for a thumbnail.
    small, large = _metrics(1.0, Color(0, 0, 0)), _metrics(3.0, Color(0, 0, 0))
    assert large.shadow.blur > small.shadow.blur
    assert large.shadow.dy > small.shadow.dy


def test_a_styles_font_size_scales_the_panel() -> None:
    image = _img()
    data = {"speed": 12.4}
    default = with_panel(image, data).render()
    explicit = with_panel(image, data, style=Style()).render()
    big = with_panel(image, data, style=Style(font_size=32.0)).render()
    assert explicit.shape == default.shape  # the default style is a no-op
    assert big.shape[0] > default.shape[0]
    assert big.shape[1] > default.shape[1]
