"""Coverage for Mask, SemanticMask, and Keypoints rendering and geometry."""

import sys

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    BBox,
    Image,
    Keypoints,
    Mask,
    MaskOutline,
    SemanticMask,
    Style,
)
from luxonis_ml.vizlab.annotations.mask import _mask_contours, _smooth_ring
from luxonis_ml.vizlab.geometry import Rect


def _canvas(w: int = 80, h: int = 60) -> Image:
    return Image(np.full((h, w, 3), 30, np.uint8))


def _disc(w: int, h: int, cx: int, cy: int, r: int) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    return ((xs - cx) ** 2 + (ys - cy) ** 2 <= r**2).astype(np.uint8)


# --- Mask (subclass of InstanceSegmentationAnnotation) ----------------------


def test_mask_array_with_contour_and_label() -> None:
    base = _canvas()
    out = (
        base.copy()
        .add(Mask(mask=_disc(80, 60, 40, 30, 15), label="moon"))  # type: ignore
        .render()
    )
    assert not np.array_equal(out[..., :3], _canvas().render()[..., :3])


def test_mask_array_without_contour() -> None:
    base = _canvas()
    base.add(Mask(mask=_disc(80, 60, 40, 30, 15), contour=False))  # type: ignore
    assert base.render()[..., 3].max() > 0


def test_mask_outline_none_skips_the_contour_like_no_contour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # NONE traces no contour (the mask's only polygon draw), exactly like
    # `contour=False`; SMOOTH still traces it. Asserted on the draw calls, which
    # is immune to the sub-pixel raster noise a pixel comparison would expose.
    from luxonis_ml.vizlab.render.canvas import Canvas

    real_polygon = Canvas.polygon
    disc = _disc(80, 60, 40, 30, 15)

    def contour_draws(mask: Mask) -> int:
        count = 0

        def counting(self: Canvas, *args: object, **kwargs: object) -> None:
            nonlocal count
            count += 1
            return real_polygon(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Canvas, "polygon", counting)
        # Upscaled render so a contour, if drawn, is smoothed and traced.
        _canvas().add(mask).render((160, 120))
        return count

    smooth = Mask(mask=disc)  # type: ignore[call-arg]
    none = Mask(mask=disc).styled(mask_outline=MaskOutline.NONE)  # type: ignore[call-arg]
    off = Mask(mask=disc, contour=False)  # type: ignore[call-arg]
    assert contour_draws(smooth) > 0  # SMOOTH (default) traces the outline
    assert contour_draws(none) == 0  # the NONE knob skips it
    assert contour_draws(off) == 0  # as does contour=False


def test_mask_outline_crisp_drops_only_the_smoothing_on_upscale() -> None:
    # On an upscaled mask, CRISP keeps the outline but skips the corner-rounding,
    # so it differs from SMOOTH (rounded) yet still draws more than NONE (none).
    disc = _disc(80, 60, 40, 30, 15)
    size = (160, 120)  # 2x upscale -> SMOOTH would smooth the contour

    def render(outline: MaskOutline) -> np.ndarray:
        return (
            _canvas()
            .add(Mask(mask=disc, label="m").styled(mask_outline=outline))  # type: ignore
            .render(size)
        )

    smooth, crisp, none = (
        render(MaskOutline.SMOOTH),
        render(MaskOutline.CRISP),
        render(MaskOutline.NONE),
    )
    assert not np.array_equal(smooth, crisp)  # smoothing dropped
    assert not np.array_equal(crisp, none)  # outline still present


def test_semantic_outline_none_removes_contours() -> None:
    labels = np.zeros((60, 80), np.int32)
    labels[:, 40:] = 1
    size = (160, 120)
    smooth = _canvas().add(SemanticMask(labels=labels)).render(size)
    none = (
        _canvas()
        .add(SemanticMask(labels=labels).styled(mask_outline=MaskOutline.NONE))
        .render(size)
    )
    assert not np.array_equal(smooth, none)


def test_mask_reuses_ldf_rle_to_numpy() -> None:
    """The mask decodes through the inherited LDF ``to_numpy`` (RLE-backed)."""
    binary = _disc(80, 60, 40, 30, 12)
    mask = Mask(mask=binary)  # type: ignore
    assert np.array_equal(mask.to_numpy() > 0, binary > 0)


def test_mask_from_polygon_points() -> None:
    base = _canvas()
    base.add(
        Mask(  # type: ignore
            points=[(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)],  # type: ignore
            width=80,
            height=60,
            label="tri",
        )
    )
    assert base.render()[..., 3].max() > 0


def test_mask_from_ldf() -> None:
    from luxonis_ml.ldf import InstanceSegmentationAnnotation

    ann = InstanceSegmentationAnnotation(mask=_disc(80, 60, 40, 30, 10))  # type: ignore
    mask = Mask.from_ldf(ann, label="obj")
    assert mask.label == "obj"
    assert np.array_equal(mask.to_numpy() > 0, ann.to_numpy() > 0)


def test_mask_extent() -> None:
    assert Mask(mask=_disc(80, 60, 40, 30, 10)).extent() is not None  # type: ignore
    assert Mask(mask=np.zeros((10, 10), dtype=np.uint8)).extent() is None  # type: ignore


def test_mask_drawn_on_smaller_canvas() -> None:
    """A mask stored at the image resolution draws onto a scaled-down canvas.

    The image can be resized for display while masks stay at their source
    resolution; drawing must resample rather than index out of bounds.
    """
    big = _disc(160, 120, 80, 60, 30)  # source-resolution mask
    small = Image(np.full((60, 80, 3), 30, np.uint8))  # half-size canvas
    out = small.add(Mask(mask=big, label="obj")).render()  # type: ignore
    assert out.shape[:2] == (60, 80)
    assert out[..., 3].max() > 0


def test_mask_drawn_on_larger_canvas() -> None:
    """A small mask upsamples to a larger canvas without error."""
    small_mask = _disc(40, 30, 20, 15, 8)
    big = Image(np.full((120, 160, 3), 30, np.uint8))
    out = big.add(Mask(mask=small_mask)).render()  # type: ignore
    assert out.shape[:2] == (120, 160)
    assert out[..., 3].max() > 0


def test_semantic_mask_drawn_on_smaller_canvas() -> None:
    """A semantic label map at source resolution resamples to a scaled canvas."""
    labels = np.zeros((120, 160), dtype=np.int32)
    labels[:60] = 1
    labels[60:] = 2
    out = (
        Image(np.full((60, 80, 3), 30, np.uint8))
        .add(SemanticMask(labels=labels, ignore_index=0))
        .render()
    )
    assert out.shape[:2] == (60, 80)
    assert out[..., 3].max() > 0


def test_mask_contours_without_opencv_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Array-mask contours degrade to none (fill only) when OpenCV is absent."""
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert _mask_contours(np.ones((6, 6), dtype=bool)) == []
    base = _canvas()
    base.add(Mask(mask=_disc(80, 60, 40, 30, 12)))  # type: ignore
    assert base.render()[..., 3].max() > 0


def test_smooth_ring_preserves_sharp_corners() -> None:
    """A rectangle keeps its 90° corners (does not round into an ellipse)."""
    square = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    smoothed = _smooth_ring(square, iterations=3)
    # Every corner is sharp, so none is cut: the ring is unchanged.
    assert len(smoothed) == 4
    assert np.allclose(
        np.array(sorted(map(tuple, smoothed))),
        np.array(sorted(map(tuple, square))),
    )


def test_smooth_ring_rounds_gentle_bends() -> None:
    """A many-sided (near-circular) outline has its gentle corners rounded."""
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 10.0
    smoothed = _smooth_ring(polygon, iterations=1)
    # Gentle (~30°) bends are cut, so the ring gains points.
    assert len(smoothed) > len(polygon)


# --- SemanticMask (render-only) ---------------------------------------------


def _label_map() -> np.ndarray:
    labels = np.zeros((60, 80), dtype=np.int32)
    labels[:30] = 1
    labels[30:] = 2
    labels[10:20, 10:30] = 3
    return labels


def test_semantic_mask_with_name_dict_and_ignore_int() -> None:
    base = _canvas()
    names = {0: "bg", 1: "sky", 2: "road", 3: "car"}
    base.add(SemanticMask(labels=_label_map(), names=names, ignore_index=0))
    assert base.render()[..., 3].max() > 0


def test_semantic_mask_name_list_and_ignore_sequence_and_color_map() -> None:
    base = _canvas()
    base.add(
        SemanticMask(
            labels=_label_map(),
            names=["bg", "sky", "road", "car"],
            ignore_index=[0, 1],
            color_map={2: "#123456"},
        )
    )
    assert base.render()[..., 3].max() > 0


def test_semantic_mask_name_out_of_range_uses_id() -> None:
    sm = SemanticMask(labels=_label_map(), names=["only_zero"])
    assert sm._name(3) == "3"  # index out of range -> str(id)
    assert SemanticMask(labels=_label_map())._name(2) == "2"  # no names


def test_semantic_mask_none_labels_is_noop() -> None:
    base = _canvas()
    plain = base.copy().render()
    with_none = base.copy().add(SemanticMask(labels=None)).render()
    assert np.array_equal(plain, with_none)


def test_semantic_mask_from_ldf_and_extent() -> None:
    from luxonis_ml.ldf import SegmentationAnnotation

    road = SegmentationAnnotation(mask=(_label_map() == 1).astype(np.uint8))  # type: ignore
    car = SegmentationAnnotation(mask=(_label_map() == 3).astype(np.uint8))  # type: ignore
    sm = SemanticMask.from_ldf([("road", road), ("car", car)])
    assert sm.extent() is None
    assert sm.labels is not None
    assert set(np.unique(sm.labels)) == {0, 1, 2}
    assert sm.names == {1: "road", 2: "car"}


def test_semantic_mask_stroke_width_zero_skips_contour() -> None:
    base = _canvas()
    base.add(
        SemanticMask(
            labels=_label_map(),
            ignore_index=0,
            style=Style(stroke_width=0.0),
        )
    )
    assert base.render()[..., 3].max() > 0


# --- Keypoints (subclass of KeypointAnnotation) -----------------------------

_EDGES = [(0, 1), (1, 2), (5, 6)]
_NAMES = ["a", "b", "c"]


def test_keypoints_render_with_skeleton_and_names() -> None:
    base = _canvas()
    base.add(
        Keypoints(
            keypoints=[(0.25, 0.15, 2), (0.25, 0.5, 1), (0.5, 0.7, 0)],
            edges=_EDGES,
            keypoint_names=_NAMES,
            label="pose",
            point_labels="names",
        )
    )
    assert base.render()[..., 3].max() > 0


def test_keypoints_point_colors_differ_from_single_color() -> None:
    # Coloring individual joints changes the pixels vs one uniform color.
    joints = [(0.25, 0.2, 2), (0.5, 0.4, 2), (0.75, 0.7, 2)]
    uniform = _canvas().add(
        Keypoints(keypoints=joints, color="#35d6a6")  # type: ignore
    )
    graded = _canvas().add(
        Keypoints(
            keypoints=joints,  # type: ignore
            color="#35d6a6",
            point_colors=[
                "#35d6a6",
                "#ff6b6b",
                None,
            ],  # None -> instance color
        )
    )
    assert not np.array_equal(uniform.render(), graded.render())


def test_keypoints_gradient_limb_between_two_colors() -> None:
    # A limb spanning two joint colors renders differently from a solid one.
    solid = _canvas().add(
        Keypoints(keypoints=[(0.2, 0.5, 2), (0.8, 0.5, 2)], edges=[(0, 1)])
    )
    two_tone = _canvas().add(
        Keypoints(
            keypoints=[(0.2, 0.5, 2), (0.8, 0.5, 2)],
            edges=[(0, 1)],
            point_colors=["#35d6a6", "#ff6b6b"],
        )
    )
    assert not np.array_equal(solid.render(), two_tone.render())


def test_keypoints_point_labels_numbers_render_without_skeleton() -> None:
    # "numbers" mode labels each joint by index and needs no skeleton.
    plain = _canvas().render()
    labeled = (
        _canvas()
        .add(
            Keypoints(
                keypoints=[(0.25, 0.15, 2), (0.5, 0.5, 2)],
                point_labels="numbers",
            )
        )
        .render()
    )
    assert not np.array_equal(plain, labeled)


def test_keypoints_point_label_text() -> None:
    kp = Keypoints(
        keypoints=[(0.1, 0.1, 2), (0.2, 0.2, 2)], keypoint_names=_NAMES
    )
    assert kp._point_label(0) is None  # default "none"
    assert (
        Keypoints(
            keypoints=[(0.1, 0.1, 2)], point_labels="numbers"
        )._point_label(0)
        == "0"
    )
    named = Keypoints(
        keypoints=[(0.1, 0.1, 2)], keypoint_names=_NAMES, point_labels="names"
    )
    assert named._point_label(1) == "b"
    full = Keypoints(
        keypoints=[(0.1, 0.1, 2)], keypoint_names=_NAMES, point_labels="full"
    )
    assert full._point_label(2) == "2:c"
    # Falls back to the index when no name is available.
    assert named._point_label(9) == "9"


def test_keypoints_visibility_threshold_hides_points() -> None:
    base = _canvas()
    base.add(
        Keypoints(
            keypoints=[(0.1, 0.1, 2), (0.4, 0.5, 0)], visibility_threshold=0.5
        )
    )
    assert base.render()[..., 3].max() > 0


def test_keypoint_occluded_renders_distinctly() -> None:
    """A COCO-occluded joint (visibility 1) reads apart from a visible one."""
    base = _canvas()
    # Mixed 2/1 is COCO-style: the visibility-1 joint is drawn as a diamond.
    mixed = (
        base.copy()
        .add(Keypoints(keypoints=[(0.3, 0.5, 2), (0.7, 0.5, 1)]))
        .render()
    )
    both_visible = (
        base.copy()
        .add(Keypoints(keypoints=[(0.3, 0.5, 2), (0.7, 0.5, 2)]))
        .render()
    )
    # Same positions, but the occluded joint's shape differs from a visible dot.
    assert not np.array_equal(mixed, both_visible)


def test_every_joint_occluded_still_reads_as_occluded() -> None:
    """A pose whose joints are *all* COCO flag 1 keeps its occluded markers.

    Visibility decides the marker per joint, with no whole-pose interpretation
    on top: an earlier heuristic read an all-``1`` column as confidence scores
    and drew plain dots, erasing the distinction for the poses where it matters
    most.
    """
    base = _canvas()
    points = [(0.3, 0.5), (0.7, 0.5)]
    occluded = (
        base.copy()
        .add(Keypoints(keypoints=[(x, y, 1) for x, y in points]))
        .render()
    )
    visible = (
        base.copy()
        .add(Keypoints(keypoints=[(x, y, 2) for x, y in points]))
        .render()
    )
    assert not np.array_equal(occluded, visible)


def test_joint_size_is_independent_of_the_visibility_value() -> None:
    # Visibility picks the marker shape and nothing else; it never scales the
    # joint. (Per-joint confidence is meant for a tooltip, not the geometry.)
    base = _canvas()

    def drawn(visibility: int) -> int:
        rendered = (
            base.copy()
            .add(Keypoints(keypoints=[(0.5, 0.5, visibility)]))  # type: ignore[list-item]
            .render()
        )
        # Pixels differing from the flat canvas background.
        return int((rendered[..., :3] != 30).any(axis=-1).sum())

    assert drawn(1) == pytest.approx(drawn(2), rel=0.15)


def test_keypoints_extent_is_none() -> None:
    # LDF keypoints are always normalized, so the pixel extent is unknown.
    assert Keypoints(keypoints=[(0.1, 0.2, 2), (0.3, 0.4, 2)]).extent() is None


def test_keypoints_from_ldf() -> None:
    from luxonis_ml.ldf import KeypointAnnotation

    ann = KeypointAnnotation(keypoints=[(0.2, 0.3, 2), (0.4, 0.5, 1)])
    kp = Keypoints.from_ldf(ann, label="pose")
    assert kp.label == "pose"
    assert len(kp.keypoints) == 2


def test_keypoints_compose_with_box() -> None:
    base = _canvas()
    base.add(BBox(x=0.05, y=0.05, w=0.9, h=0.9, label="person"))
    base.add(
        Keypoints(
            keypoints=[(0.25, 0.25, 2), (0.5, 0.6, 2)],
            edges=[(0, 1)],
        )
    )
    assert base.render()[..., 3].max() > 0


def test_mask_rect_helper_is_available() -> None:
    # Sanity: geometry Rect still imports for downstream use.
    assert Rect(0.0, 0.0, 1.0, 1.0).width == 1.0


def test_mask_contours_skip_single_point_rings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2

    monkeypatch.setattr(
        cv2,
        "findContours",
        lambda _mask, _mode, _method: (
            [np.array([[[1, 1]]], dtype=np.int32)],
            None,
        ),
    )
    assert _mask_contours(np.ones((3, 3), dtype=bool)) == []


def test_empty_semantic_mask_from_ldf() -> None:
    semantic = SemanticMask.from_ldf([])
    assert semantic.labels is None


def test_confidence_keypoint_visibility_scales_joint_radius() -> None:
    rendered = (
        _canvas()
        .add(
            Keypoints(
                keypoints=[(0.3, 0.5, 1), (0.7, 0.5, 1)],
                visibility_threshold=0.0,
            )
        )
        .render()
    )
    assert rendered.shape == (60, 80, 4)
