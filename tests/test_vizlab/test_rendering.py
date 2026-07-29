"""Tests that are awkward to express as doctests.

Doctests cover the pure logic (color/geometry math, palette semantics). This
module covers the things that need numpy arrays or many-sample sweeps: that
rendering is deterministic and actually draws, and that the palette keeps colors
visibly apart even for a large number of classes.
"""

import math
from pathlib import Path

import numpy as np

from luxonis_ml.vizlab import (
    DARK_THEME,
    LIGHT_THEME,
    BBox,
    Classification,
    Color,
    Corner,
    Image,
    Keypoints,
    Mask,
    Rect,
    RenderOptions,
    SemanticMask,
    default_options,
    get_default_theme,
    grid,
    hstack,
    set_default_theme,
)
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.annotations.layout import (
    LabelLayout,
    Placement,
    _overlap_area,
    label_candidates,
)
from luxonis_ml.vizlab.canvas import Canvas
from luxonis_ml.vizlab.scene.image import _freeze_render_state
from luxonis_ml.vizlab.style import (
    DEFAULT_STYLE,
    LabelPlacement,
    Palette,
    Style,
    derive_child_color,
    derive_child_style,
)


def _rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.dist(a, b)


def _closest_pair_distance(colors: list[Color]) -> float:
    rgbs = [(c.r, c.g, c.b) for c in colors]
    return min(
        _rgb_distance(rgbs[i], rgbs[j])
        for i in range(len(rgbs))
        for j in range(i + 1, len(rgbs))
    )


def test_palette_colors_stay_far_apart() -> None:
    palette = Palette()
    colors = [palette.color_for(f"class_{i}") for i in range(12)]
    assert len({(c.r, c.g, c.b) for c in colors}) == len(colors)
    assert _closest_pair_distance(colors) > 30.0


def test_palette_avoids_near_identical_at_scale() -> None:
    palette = Palette()
    colors = [palette.color_for(f"class_{i}") for i in range(80)]
    assert len({(c.r, c.g, c.b) for c in colors}) == len(colors)
    assert _closest_pair_distance(colors) > 12.0


def test_palette_color_is_stable_within_instance() -> None:
    palette = Palette()
    first = palette.color_for("car")
    for other in ("truck", "bus", "person"):
        palette.color_for(other)
    assert palette.color_for("car") == first


def test_render_is_deterministic_and_draws() -> None:
    base = np.full((80, 120, 3), 40, np.uint8)
    first = (
        Image(base)
        .add(BBox(x=0.1, y=0.1, w=0.7, h=0.6, color="#4c8dff"))
        .render()
    )
    second = (
        Image(base)
        .add(BBox(x=0.1, y=0.1, w=0.7, h=0.6, color="#4c8dff"))
        .render()
    )
    assert first.shape == (80, 120, 4)
    assert np.array_equal(first, second)
    assert not np.array_equal(first[..., :3], base)


def test_render_is_cached_until_add() -> None:
    img = Image(np.full((40, 40, 3), 10, np.uint8))
    before = img.render()
    img.add(BBox(x=0.1, y=0.1, w=0.7, h=0.7, color="#ff6b6b"))
    after = img.render()
    assert not np.array_equal(before, after)


def test_nested_child_derives_color_and_style() -> None:
    parent_color = Color(50, 110, 190)
    ctx = RenderContext(
        canvas=None,  # type: ignore[arg-type]
        depth=1,
        parent_color=parent_color,
        parent_style=Style(),
    )
    # An unlabeled child (e.g. keypoints on a box) derives the parent's color and
    # style, so it reads as part of the parent.
    child = BBox(x=0.1, y=0.1, w=0.2, h=0.2)
    assert child.resolve_color(ctx) == derive_child_color(parent_color)
    assert child.resolve_style(ctx) == derive_child_style(Style())

    # A labeled child is its own class: it fills and captions in its own palette
    # color, not a shade of the parent, but still derives the dashed sub-label
    # style and outlines in the parent's color so it stays visibly nested.
    palette = Palette()
    labeled = BBox(x=0.1, y=0.1, w=0.2, h=0.2, label="car", palette=palette)
    own = labeled.resolve_color(ctx)
    assert own == palette.color_for("car")
    assert own != derive_child_color(parent_color)
    assert labeled.resolve_style(ctx) == derive_child_style(Style())
    assert labeled.outline_color(ctx, own) == parent_color


def test_top_level_uses_palette_and_explicit_color_wins() -> None:
    ctx = RenderContext(canvas=None)  # type: ignore[arg-type]
    palette = Palette()
    labeled = BBox(x=0.0, y=0.0, w=1.0, h=1.0, label="car", palette=palette)
    assert labeled.resolve_color(ctx) == palette.color_for("car")
    assert labeled.resolve_style(ctx) is DEFAULT_STYLE
    assert BBox(x=0.0, y=0.0, w=1.0, h=1.0, color=(255, 0, 0)).resolve_color(
        ctx
    ) == Color(255, 0, 0)


def test_fluent_helpers_set_fields() -> None:
    style = Style(stroke_width=6.0)
    box = (
        BBox(x=0.0, y=0.0, w=1.0, h=1.0)
        .tag("car", score=0.9)
        .caption("plate-42")
        .styled(style)
    )
    assert (box.label, box.score, box.payload, box.style) == (
        "car",
        0.9,
        "plate-42",
        style,
    )


def test_render_cache_tracks_annotation_mutations() -> None:
    base = np.full((160, 200, 3), 20, np.uint8)
    box = BBox(x=0.2, y=0.2, w=0.5, h=0.5)
    image = Image(base).add(box)

    previous = image.render()
    box.tag("car")
    tagged = image.render()
    assert not np.array_equal(previous, tagged)

    box.caption("plate-42")
    captioned = image.render()
    assert not np.array_equal(tagged, captioned)

    box.styled(Style(stroke_width=9.0))
    styled = image.render()
    assert not np.array_equal(captioned, styled)

    box.add(BBox(x=0.25, y=0.25, w=0.2, h=0.2, label="child"))
    nested = image.render()
    assert not np.array_equal(styled, nested)

    image.annotations.append(Classification(tags=["scene"]))
    appended = image.render()
    assert not np.array_equal(nested, appended)


def test_render_state_freezes_sets_deterministically() -> None:
    assert _freeze_render_state({3, 1, 2}) == _freeze_render_state({2, 3, 1})


def test_render_cache_tracks_scoped_style_state() -> None:
    base = np.full((80, 120, 3), 20, np.uint8)
    image = Image(base).add(BBox(x=0.13, y=0.17, w=0.61, h=0.53))
    regular = image.render()

    with Style.override(fill_alpha=0.8, stroke_width=8.0, shadow=False):
        overridden = image.render()
    with Style(fill_alpha=0.0, stroke_width=1.0).as_default():
        replaced = image.render()

    assert not np.array_equal(regular, overridden)
    assert not np.array_equal(regular, replaced)


def test_render_cache_tracks_scoped_antialias_option() -> None:
    base = np.full((80, 120, 3), 20, np.uint8)
    image = Image(base).add(BBox(x=0.133, y=0.177, w=0.611, h=0.533))
    antialiased = image.render()

    with default_options(RenderOptions(antialias=False)):
        hard_edges = image.render()

    assert not np.array_equal(antialiased, hard_edges)


def test_semantic_segmentation_renders_beneath_other_masks() -> None:
    """A `SemanticMask` is a background layer, drawn under every other spatial
    annotation regardless of the order it was added.
    """
    assert SemanticMask.BACKGROUND is True
    assert Mask.BACKGROUND is False

    instance = Mask.model_validate(
        {
            "mask": np.ones((20, 20), np.uint8),
            "color": "#ff0000",
            "fill_alpha": 1.0,
            "contour": False,
        }
    )
    semantic = SemanticMask(
        labels=np.ones((20, 20), np.int32),
        color_map={1: "#0000ff"},
        fill_alpha=1.0,
    )
    # Semantic segmentation is added *after* the instance mask but must still
    # render below it, so the center pixel stays the instance mask's red.
    rendered = (
        Image(np.zeros((20, 20, 3), np.uint8)).add(instance).add(semantic)
    )
    center = rendered.render()[10, 10]
    assert center[0] > center[2]


def test_nested_rendering_changes_pixels() -> None:
    base = np.full((120, 120, 3), 30, np.uint8)
    parent_only = (
        Image(base).add(BBox(x=0.1, y=0.1, w=0.8, h=0.8, label="car")).render()
    )
    with_child = (
        Image(base)
        .add(
            BBox(x=0.1, y=0.1, w=0.8, h=0.8, label="car").add(
                BBox(x=0.25, y=0.25, w=0.5, h=0.5, label="driver")
            )
        )
        .render()
    )
    assert not np.array_equal(parent_only, with_child)


def test_classification_renders_corner_chips() -> None:
    base = np.full((160, 200, 3), 20, np.uint8)
    tagged = (
        Image(base)
        .add(Classification(tags=[("cat", 0.9), ("indoor", 0.7)]))
        .render()
    )
    assert not np.array_equal(tagged[..., :3], base)


def test_image_tags_render_on_top_regardless_of_add_order() -> None:
    base = np.full((120, 200, 3), 15, np.uint8)

    def box() -> BBox:
        return BBox(x=0.02, y=0.03, w=0.96, h=0.94, color="#4c8dff")

    def tag() -> Classification:
        return Classification(tags=[("cat", 0.9)], corner=Corner.TOP_LEFT)

    tag_then_box = Image(base).add(tag()).add(box()).render()
    box_then_tag = Image(base).add(box()).add(tag()).render()
    assert np.array_equal(tag_then_box, box_then_tag)


def test_classification_reserves_positions_for_layout() -> None:
    canvas = Canvas.blank(200, 120)
    layout = LabelLayout(200, 120)
    ctx = RenderContext(canvas=canvas, layout=layout)
    Classification(tags=[("a", 0.9), ("b", 0.8)]).reserve(ctx)
    assert len(layout.placed) == 2


def test_classification_overlays_share_one_corner_stack() -> None:
    base = np.full((160, 200, 3), 20, np.uint8)
    separate = (
        Image(base)
        .add(Classification(tags=["cat"]))
        .add(Classification(tags=["dog"]))
        .render()
    )
    combined = Image(base).add(Classification(tags=["cat", "dog"])).render()
    assert np.array_equal(separate, combined)


def test_blend_pads_mismatched_sizes_and_is_pure() -> None:
    a = Image(np.zeros((10, 20, 3), np.uint8))
    b = Image(np.full((30, 12, 3), 200, np.uint8))
    out = a.blend(b, alpha=0.5).render()
    assert out.shape == (30, 20, 4)
    assert a.annotations == []
    assert b.annotations == []


def test_blend_transforms_annotations_for_padded_images() -> None:
    box = BBox(x=0.5, y=0.3, w=0.4, h=0.5, label="small")
    box.add(Keypoints(keypoints=[(0.75, 0.6, 2)]))
    box.add(Mask(mask=np.ones((10, 20), dtype=np.uint8)))  # type: ignore
    small = Image(np.zeros((10, 20, 3), np.uint8)).add(box)
    large = Image(np.zeros((30, 40, 3), np.uint8))

    small.render()  # populate the source mask's decoded-array cache
    merged = large.blend(small)
    transformed = merged.annotations[0]

    assert isinstance(transformed, BBox)
    assert np.allclose(
        (transformed.x, transformed.y, transformed.w, transformed.h),
        (0.25, 0.1, 0.2, 1 / 6),
    )
    keypoints = next(
        child for child in transformed.children if isinstance(child, Keypoints)
    )
    assert np.allclose(keypoints.keypoints, [(0.375, 0.2, 2)])
    mask = next(
        child for child in transformed.children if isinstance(child, Mask)
    )
    mask_array = mask._dense()
    assert mask_array.shape == (30, 40)
    assert np.all(mask_array[:10, :20] == 1)
    assert np.all(mask_array[10:, :] == 0)
    assert np.all(mask_array[:, 20:] == 0)
    assert (box.x, box.y, box.w, box.h) == (0.5, 0.3, 0.4, 0.5)


def test_blend_pads_semantic_mask_labels() -> None:
    labels = np.ones((10, 20), dtype=np.int32)
    small = Image(np.zeros((10, 20, 3), np.uint8)).add(
        SemanticMask(labels=labels)
    )
    large = Image(np.zeros((30, 40, 3), np.uint8))

    merged = large.blend(small)

    semantic = next(
        annotation
        for annotation in merged.annotations
        if isinstance(annotation, SemanticMask)
    )
    assert semantic.labels is not None
    assert semantic.labels.shape == (30, 40)
    assert np.all(semantic.labels[:10, :20] == 1)
    assert np.all(semantic.labels[10:, :] == 0)
    assert np.all(semantic.labels[:, 20:] == 0)


def test_blend_only_mixes_base_and_keeps_labels_crisp() -> None:
    a = Image(np.zeros((60, 60, 3), np.uint8)).add(
        BBox(x=0.08, y=0.08, w=0.85, h=0.85, label="cat")
    )
    b = Image(np.full((60, 60, 3), 200, np.uint8)).add(
        BBox(x=0.08, y=0.08, w=0.85, h=0.85, label="dog")
    )
    merged = a.blend(b, alpha=0.5)
    assert [ann.label for ann in merged.annotations] == ["cat", "dog"]
    assert merged.render()[..., 3].max() == 255


def test_grid_lays_out_uniform_cells() -> None:
    cells = [Image(np.zeros((10, 10, 3), np.uint8)) for _ in range(4)]
    assert grid(cells, ncols=2, pad=4).render().shape == (32, 32, 4)


def test_titles_add_height() -> None:
    cells = [Image(np.zeros((10, 10, 3), np.uint8)) for _ in range(2)]
    plain = hstack(cells, pad=4).render().shape[0]
    titled = hstack(cells, pad=4, titles=["a", "b"]).render().shape[0]
    assert titled > plain


def test_multiline_markup_title_adds_a_line() -> None:
    """A newline in a (markup) title stacks a second line, growing the band."""
    cells = [Image(np.zeros((40, 120, 3), np.uint8)) for _ in range(2)]
    one_line = grid(cells, ncols=2, pad=4, titles=["Heading", "x"])
    two_line = grid(
        cells, ncols=2, pad=4, titles=["Heading\n<code>path</code>", "x"]
    )
    assert two_line.render().shape[0] > one_line.render().shape[0]


def test_emphasized_titles_are_taller_than_plain() -> None:
    """Emphasized titles (larger + bold) occupy a taller band than plain ones."""
    cells = [Image(np.zeros((40, 200, 3), np.uint8)) for _ in range(2)]
    emphasized = grid(cells, ncols=2, pad=4, titles=["Heading", "x"])
    plain = grid(
        cells, ncols=2, pad=4, titles=["Heading", "x"], emphasize_titles=False
    )
    assert emphasized.render().shape[0] > plain.render().shape[0]


def _placed_chips(
    *boxes: BBox, width: int = 400, height: int = 300
) -> list[Rect]:
    canvas = Canvas.blank(width, height)
    layout = LabelLayout(width, height)
    ctx = RenderContext(canvas=canvas, layout=layout)
    # Shapes first, then the label chips (mirroring Image.render's two passes).
    for box in boxes:
        box.render(ctx)
    for box in boxes:
        box.render_labels(ctx)
    return layout.placed


def test_labels_are_deferred_to_a_separate_pass() -> None:
    """``render`` draws only the shape; ``render_labels`` places the chip.

    This is what keeps a later box from covering an earlier box's label.
    """
    canvas = Canvas.blank(200, 150)
    layout = LabelLayout(200, 150)
    ctx = RenderContext(canvas=canvas, layout=layout)
    box = BBox(x=0.1, y=0.2, w=0.5, h=0.5, label="car", score=0.9)

    box.render(ctx)
    assert layout.placed == []  # no chip yet — only the box outline was drawn
    box.render_labels(ctx)
    assert len(layout.placed) == 1


def test_overlapping_box_labels_do_not_collide() -> None:
    a = BBox(x=0.1, y=0.2, w=0.75, h=0.85, label="cat", score=0.9)
    b = BBox(x=0.1, y=0.2, w=0.75, h=0.85, label="dog", score=0.9)
    chips = _placed_chips(a, b)
    assert len(chips) == 2
    assert _overlap_area(chips[0], chips[1]) == 0.0


def test_many_stacked_boxes_minimize_label_overlap() -> None:
    boxes = [
        BBox(x=0.08, y=0.17, w=0.8, h=0.83, label=name)
        for name in ("a", "b", "c", "d")
    ]
    chips = _placed_chips(*boxes)
    total = sum(
        _overlap_area(chips[i], chips[j])
        for i in range(len(chips))
        for j in range(i + 1, len(chips))
    )
    single_area = chips[0].width * chips[0].height
    assert total < single_area


def _place_repeatedly(
    region: Rect, n: int, w: float = 70.0, h: float = 26.0
) -> list[Placement]:
    """Place ``n`` chips all labeling the same ``region`` and return placements."""
    layout = LabelLayout(600, 600)
    out = []
    for _ in range(n):
        cands = label_candidates(region, w, h, LabelPlacement.TOP)
        out.append(layout.place(w, h, cands, region=region))
    return out


def test_clear_box_places_chip_against_it_without_a_leader() -> None:
    region = Rect(200.0, 200.0, 320.0, 260.0)
    (placement,) = _place_repeatedly(region, 1)
    assert placement.leader is None  # nothing to escape — sits on the box


def test_dense_labels_escape_with_leaders_back_to_their_box() -> None:
    """When a box's neighborhood fills up, later chips push out on leader lines."""
    region = Rect(250.0, 250.0, 350.0, 310.0)
    placements = _place_repeatedly(region, 7)

    detached = [p for p in placements if p.leader is not None]
    assert detached, (
        "expected some chips to be pushed clear of the crowded box"
    )
    # Every leader anchors on the box it belongs to.
    for p in detached:
        assert p.leader is not None
        ax, ay = p.leader
        assert region.left <= ax <= region.right
        assert region.top <= ay <= region.bottom

    # Declutter actually worked: total pairwise overlap stays well under what a
    # naive same-corner stack (every chip on top of the last) would produce.
    rects = [p.rect for p in placements]
    total = sum(
        _overlap_area(rects[i], rects[j])
        for i in range(len(rects))
        for j in range(i + 1, len(rects))
    )
    single_area = rects[0].width * rects[0].height
    assert total < single_area


def test_small_overlap_keeps_the_chip_on_the_box_rather_than_leading_out() -> (
    None
):
    """A single mild collision is tolerated in place — no gratuitous leader."""
    layout = LabelLayout(600, 600)
    region = Rect(250.0, 250.0, 350.0, 310.0)
    first = label_candidates(region, 70.0, 26.0, LabelPlacement.TOP)
    layout.place(70.0, 26.0, first, region=region)
    # A second, slightly offset box: adjacent slots are still mostly free, so it
    # should settle against its box without a leader line.
    near = Rect(258.0, 254.0, 358.0, 314.0)
    cands = label_candidates(near, 70.0, 26.0, LabelPlacement.TOP)
    placement = layout.place(70.0, 26.0, cands, region=near)
    assert placement.leader is None


def test_label_alpha_fades_the_chip() -> None:
    base = np.full((80, 160, 3), 30, np.uint8)
    opaque = (
        Image(base)
        .add(BBox(x=0.06, y=0.37, w=0.9, h=0.5, label="car", score=0.9))
        .render()
    )
    faded = (
        Image(base)
        .add(
            BBox(
                x=0.06,
                y=0.37,
                w=0.9,
                h=0.5,
                label="car",
                score=0.9,
                style=Style(label_alpha=0.4),
            )
        )
        .render()
    )
    assert not np.array_equal(opaque, faded)


def test_theme_supplies_default_palette_and_style() -> None:
    canvas = Canvas.blank(1, 1)
    box = BBox(x=0.0, y=0.0, w=1.0, h=1.0, label="car")
    dark = box.resolve_color(RenderContext(canvas=canvas, theme=DARK_THEME))
    light = box.resolve_color(RenderContext(canvas=canvas, theme=LIGHT_THEME))
    assert dark != light
    light_ctx = RenderContext(canvas=canvas, theme=LIGHT_THEME)
    assert box.resolve_style(light_ctx) is LIGHT_THEME.style
    own = Palette()
    assert BBox(
        x=0.0, y=0.0, w=1.0, h=1.0, label="car", palette=own
    ).resolve_color(light_ctx) == own.color_for("car")


def test_default_theme_is_used_and_settable() -> None:
    base = np.full((40, 80, 3), 30, np.uint8)
    box = BBox(x=0.06, y=0.37, w=0.9, h=0.5, label="car")
    original_theme = get_default_theme()
    try:
        set_default_theme(DARK_THEME)
        image = Image(base).add(box)
        dark_render = image.render()
        set_default_theme(LIGHT_THEME)
        assert get_default_theme() is LIGHT_THEME
        light_render = image.render()
        fresh_light_render = Image(base).add(box).render()
    finally:
        set_default_theme(original_theme)
    assert not np.array_equal(dark_render, light_render)
    assert np.array_equal(light_render, fresh_light_render)


def test_render_svg_is_vector_over_an_embedded_base() -> None:
    image = Image(np.full((60, 90, 3), 40, np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.4, label="car", score=0.9)
    )
    svg = image.render_svg().decode("utf-8")
    assert svg.startswith("<?xml")
    assert 'width="90"' in svg  # native viewport
    assert 'height="60"' in svg
    assert "<path" in svg  # box stroke + glyph outlines are vectors
    assert "<image" in svg or "base64" in svg  # the photo is embedded raster


def test_render_svg_honors_the_requested_size() -> None:
    image = Image(np.full((60, 90, 3), 40, np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.4, label="car")
    )
    svg = image.render_svg((180, 120)).decode("utf-8")
    assert 'width="180"' in svg
    assert 'height="120"' in svg


def test_render_svg_keeps_text_when_not_pathified() -> None:
    image = Image(np.full((40, 60, 3), 40, np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.4, label="car")
    )
    assert "<text" in image.render_svg(text_as_paths=False).decode("utf-8")


def test_save_writes_svg_from_extension(tmp_path: Path) -> None:
    path = tmp_path / "out.svg"
    Image(np.zeros((20, 30, 3), np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.5, label="car")
    ).save(path)
    assert path.read_bytes().startswith(b"<?xml")


def test_with_panel_composite_renders_to_vector_svg() -> None:
    # Target-aware composition: a paneled composite renders to SVG with the photo
    # embedded once and everything else (boxes, panel chrome) as vectors.
    from luxonis_ml.vizlab import Composite, with_panel

    img = Image(np.full((60, 90, 3), 40, np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.4, label="car", score=0.9)
    )
    composite = with_panel(img, {"id": 1, "source": "frame.jpg"})
    assert isinstance(composite, Composite)
    svg = composite.render_svg().decode("utf-8")
    assert svg.count("<image") == 1  # only the photo is raster
    assert svg.count("<path") > 10  # boxes + panel text are vectors
    # ...and it still renders to raster identically in shape to a normal render.
    assert composite.render().shape[2] == 4


def test_composite_supports_the_shared_numpy_output_contract() -> None:
    from luxonis_ml.vizlab import grid

    composite = grid([Image(np.zeros((20, 30, 3), np.uint8))])

    assert composite.to_numpy("bgr").shape == (
        composite.height,
        composite.width,
        3,
    )


def test_grid_composite_renders_to_svg() -> None:
    # A grid is a composite too: SVG embeds each tile's photo, tiles' boxes vector.
    from luxonis_ml.vizlab import Composite, grid

    tiles = [
        Image(np.full((40, 40, 3), 30, np.uint8)).add(
            BBox(x=0.1, y=0.1, w=0.5, h=0.5, label="car")
        )
        for _ in range(4)
    ]
    composed = grid(tiles, ncols=2)
    assert isinstance(composed, Composite)
    svg = composed.render_svg().decode("utf-8")
    assert svg.count("<image") == 4  # one embedded photo per tile
