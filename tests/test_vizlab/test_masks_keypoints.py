"""Coverage for Mask, SemanticMask, and Keypoints rendering and geometry."""

import sys

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    BBox,
    Image,
    Keypoints,
    Mask,
    SemanticMask,
    Skeleton,
    Style,
)
from luxonis_ml.vizlab.annotations.mask import _mask_contours
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


def test_mask_contours_without_opencv_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Array-mask contours degrade to none (fill only) when OpenCV is absent."""
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert _mask_contours(np.ones((6, 6), dtype=bool)) == []
    base = _canvas()
    base.add(Mask(mask=_disc(80, 60, 40, 30, 12)))  # type: ignore
    assert base.render()[..., 3].max() > 0


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

_SKELETON = Skeleton(edges=((0, 1), (1, 2), (5, 6)), names=("a", "b", "c"))


def test_keypoints_render_with_skeleton_and_names() -> None:
    base = _canvas()
    base.add(
        Keypoints(
            keypoints=[(0.25, 0.15, 2), (0.25, 0.5, 1), (0.5, 0.7, 0)],
            skeleton=_SKELETON,
            label="pose",
            show_names=True,
        )
    )
    assert base.render()[..., 3].max() > 0


def test_keypoints_visibility_threshold_hides_points() -> None:
    base = _canvas()
    base.add(
        Keypoints(
            keypoints=[(0.1, 0.1, 2), (0.4, 0.5, 0)], visibility_threshold=0.5
        )
    )
    assert base.render()[..., 3].max() > 0


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
            skeleton=Skeleton(edges=((0, 1),)),
        )
    )
    assert base.render()[..., 3].max() > 0


def test_mask_rect_helper_is_available() -> None:
    # Sanity: geometry Rect still imports for downstream use.
    assert Rect(0.0, 0.0, 1.0, 1.0).width == 1.0
