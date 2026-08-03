import numpy as np
import pytest

from luxonis_ml.data.augmentations.custom.cutmix import CutMix


def test_compute_patch() -> None:
    assert CutMix._compute_patch(0.75, 8, 8, 4, 4) == (2, 2, 6, 6)
    assert CutMix._compute_patch(0.75, 8, 8, 0, 0) == (0, 0, 2, 2)


def test_cutmix_image() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    image1 = np.zeros((6, 8, 3), dtype=np.uint8)
    image2 = np.full((6, 8, 3), 255, dtype=np.uint8)

    image = cutmix.apply(
        [image1, image2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert np.all(image[1:4, 2:5] == 255)
    assert np.all(image[:1] == 0)
    assert np.all(image[4:] == 0)
    assert np.all(image[:, :2] == 0)
    assert np.all(image[:, 5:] == 0)


def test_cutmix_image_zero_area_patch() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    image1 = np.zeros((6, 8, 3), dtype=np.uint8)
    image2 = np.full((6, 8, 3), 255, dtype=np.uint8)

    image = cutmix.apply(
        [image1, image2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=2,
        y2=4,
    )

    np.testing.assert_array_equal(image, image1)


def test_cutmix_without_aspect_ratio_resize() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, keep_aspect_ratio=False)
    image1 = np.zeros((6, 8, 3), dtype=np.uint8)
    image2 = np.full((3, 4, 3), 255, dtype=np.uint8)

    image = cutmix.apply(
        [image1, image2],
        image_shapes=[(6, 8), (3, 4)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert np.all(image[1:4, 2:5] == 255)


def test_cutmix_semantic_mask() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8, 1), 2, dtype=np.uint8)

    mask = cutmix.apply_to_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert np.all(mask[1:4, 2:5] == 2)
    assert np.all(mask[:1] == 1)
    assert np.all(mask[4:] == 1)
    assert np.all(mask[:, :2] == 1)
    assert np.all(mask[:, 5:] == 1)


def test_cutmix_semantic_mask_empty_base() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask2 = np.full((6, 8), 2, dtype=np.uint8)

    mask = cutmix.apply_to_mask(
        [np.array([]), mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert mask.shape == (6, 8, 1)
    assert np.all(mask[1:4, 2:5] == 2)
    assert np.all(mask[:1] == 0)
    assert np.all(mask[4:] == 0)


def test_cutmix_semantic_mask_2d_patch_source_after_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8), 2, dtype=np.uint8)
    monkeypatch.setattr(cutmix, "_resize", lambda *_, **__: mask2)

    mask = cutmix.apply_to_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert mask.shape == (6, 8, 1)
    assert np.all(mask[1:4, 2:5] == 2)
    assert np.all(mask[:1] == 1)
    assert np.all(mask[4:] == 1)


def test_cutmix_semantic_mask_empty_inputs() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)

    mask = cutmix.apply_to_mask(
        [np.array([]), np.array([])],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert mask.size == 0


def test_cutmix_semantic_mask_zero_area_patch() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8, 1), 2, dtype=np.uint8)

    mask = cutmix.apply_to_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=2,
        y2=4,
    )

    np.testing.assert_array_equal(mask, mask1)


def test_cutmix_semantic_mask_empty_patch_source() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)

    mask = cutmix.apply_to_mask(
        [mask1, np.array([])],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert np.all(mask[1:4, 2:5] == 0)
    assert np.all(mask[:1] == 1)
    assert np.all(mask[4:] == 1)


def test_cutmix_instance_masks() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8, 1), 2, dtype=np.uint8)

    masks = cutmix.apply_to_instance_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert masks.shape == (6, 8, 2)
    assert np.all(masks[1:4, 2:5, 0] == 0)
    assert np.all(masks[:1, :, 0] == 1)
    assert np.all(masks[4:, :, 0] == 1)
    assert np.all(masks[:, :2, 0] == 1)
    assert np.all(masks[:, 5:, 0] == 1)
    assert np.all(masks[1:4, 2:5, 1] == 2)
    assert np.all(masks[:1, :, 1] == 0)
    assert np.all(masks[4:, :, 1] == 0)
    assert np.all(masks[:, :2, 1] == 0)
    assert np.all(masks[:, 5:, 1] == 0)


def test_cutmix_instance_mask_empty_base() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask2 = np.full((6, 8, 1), 2, dtype=np.uint8)

    masks = cutmix.apply_to_instance_mask(
        [np.array([]), mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert masks.shape == (6, 8, 1)
    assert np.all(masks[1:4, 2:5, 0] == 2)
    assert np.all(masks[:1, :, 0] == 0)
    assert np.all(masks[4:, :, 0] == 0)


def test_cutmix_instance_mask_2d_base_empty_patch_source() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8), dtype=np.uint8)

    masks = cutmix.apply_to_instance_mask(
        [mask1, np.array([])],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert masks.shape == (6, 8, 1)
    assert np.all(masks[1:4, 2:5, 0] == 0)
    assert np.all(masks[:1, :, 0] == 1)
    assert np.all(masks[4:, :, 0] == 1)


def test_cutmix_instance_mask_2d_patch_source() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, keep_aspect_ratio=False)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8), 2, dtype=np.uint8)

    masks = cutmix.apply_to_instance_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert masks.shape == (6, 8, 2)
    assert np.all(masks[1:4, 2:5, 1] == 2)


def test_cutmix_bboxes() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    bbox1 = np.array(
        [
            [0.0, 0.0, 0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    bbox2 = np.array([[0.2, 0.2, 0.8, 0.8, 1.0, 1.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, bbox2],
        image_shapes=[(10, 10), (10, 10)],
        x1=4,
        y1=3,
        x2=9,
        y2=7,
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.4, 0.3, 0.8, 0.7, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(bboxes, expected)


def test_cutmix_bboxes_strict_visibility() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, bbox_min_visibility=1.0)
    bbox1 = np.array(
        [
            [0.0, 0.0, 0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    bbox2 = np.array([[0.2, 0.2, 0.8, 0.8, 1.0, 1.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, bbox2],
        image_shapes=[(10, 10), (10, 10)],
        x1=4,
        y1=3,
        x2=9,
        y2=7,
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.3, 0.2, 0.0, 0.0],
            [0.4, 0.3, 0.8, 0.7, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(bboxes, expected)


def test_cutmix_bboxes_min_visibility_keeps_all() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, bbox_min_visibility=0.0)
    bbox1 = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=0,
        y1=0,
        x2=10,
        y2=10,
    )

    np.testing.assert_allclose(bboxes, bbox1)


def test_cutmix_bboxes_min_visibility_drops_mostly_covered() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, bbox_min_visibility=0.5)
    bbox1 = np.array([[0.0, 0.0, 0.4, 0.4, 0.0, 0.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=1,
        y1=1,
        x2=4,
        y2=4,
    )

    assert bboxes.shape == (0, 6)


def test_cutmix_empty_bboxes() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)

    bboxes = cutmix.apply_to_bboxes(
        [np.array([]), np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=4,
        y1=3,
        x2=9,
        y2=7,
    )

    assert bboxes.shape == (0, 6)


def test_cutmix_keypoints() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    keypoints1 = np.array(
        [[2.0, 2.0, 0.0, 0.0, 2.0], [6.0, 6.0, 0.0, 0.0, 2.0]]
    )
    keypoints2 = keypoints1.copy()

    keypoints = cutmix.apply_to_keypoints(
        [keypoints1, keypoints2],
        image_shapes=[(8, 8), (8, 8)],
        x1=1,
        y1=1,
        x2=4,
        y2=4,
    )

    np.testing.assert_array_equal(keypoints[:, -1], [0.0, 2.0, 2.0, 0.0])


def test_cutmix_keypoints_zero_area_patch() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    keypoints1 = np.array([[2.0, 2.0, 0.0, 0.0, 2.0]])
    keypoints2 = np.array([[6.0, 6.0, 0.0, 0.0, 2.0]])

    keypoints = cutmix.apply_to_keypoints(
        [keypoints1, keypoints2],
        image_shapes=[(8, 8), (8, 8)],
        x1=2,
        y1=1,
        x2=2,
        y2=7,
    )

    np.testing.assert_array_equal(keypoints, keypoints1)


def test_cutmix_non_spatial_labels_zero_area_patch() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    array1 = np.array([[1.0, 2.0]])
    metadata1 = np.array([[3.0]])
    classification1 = np.array([1.0, 0.0])

    params = {"x1": 2, "y1": 1, "x2": 2, "y2": 7}

    np.testing.assert_array_equal(
        cutmix.apply_to_array([array1, np.array([[4.0, 5.0]])], **params),
        array1,
    )
    np.testing.assert_array_equal(
        cutmix.apply_to_metadata([metadata1, np.array([[6.0]])], **params),
        metadata1,
    )
    np.testing.assert_array_equal(
        cutmix.apply_to_classification(
            [classification1, np.array([0.0, 1.0])], **params
        ),
        classification1,
    )


def test_cutmix_empty_keypoints() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)

    keypoints = cutmix.apply_to_keypoints(
        [np.array([]), np.array([])],
        image_shapes=[(8, 8), (8, 8)],
        x1=1,
        y1=1,
        x2=4,
        y2=4,
    )

    assert keypoints.shape == (0, 5)


def test_resize_invalid_target_type() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)

    with pytest.raises(ValueError, match="Unsupported target type"):
        cutmix._resize(
            np.zeros((1, 1, 1), dtype=np.uint8),
            [(1, 1), (1, 1)],
            "invalid",  # type: ignore[arg-type]
        )


def test_bbox_patch_helpers_empty_results() -> None:
    bboxes = np.array([[0.0, 0.0, 0.2, 0.2, 0.0, 0.0]])

    clipped = CutMix._clip_bboxes_to_patch(
        bboxes, height=10, width=10, x1=4, y1=4, x2=8, y2=8
    )
    assert clipped.shape == (0, 6)

    zero_area = CutMix._clip_bboxes_to_patch(
        bboxes, height=10, width=10, x1=4, y1=4, x2=4, y2=8
    )
    assert zero_area.shape == (0, 6)

    unchanged = CutMix._filter_bboxes_by_visibility(
        bboxes,
        height=10,
        width=10,
        x1=4,
        y1=4,
        x2=4,
        y2=8,
        min_visibility=0.5,
    )
    np.testing.assert_array_equal(unchanged, bboxes)


def test_keypoint_and_dimension_helpers_empty_or_dimensional() -> None:
    keypoints = np.array([])
    assert (
        CutMix._mark_keypoints_in_patch(
            keypoints, x1=1, y1=1, x2=4, y2=4, visible_inside=True
        ).size
        == 0
    )

    squeezed = CutMix._match_dimensions(
        np.ones((2, 3, 1), dtype=np.uint8),
        np.ones((2, 3), dtype=np.uint8),
    )
    assert squeezed.shape == (2, 3)

    expanded = CutMix._match_dimensions(
        np.ones((2, 3), dtype=np.uint8),
        np.ones((2, 3, 1), dtype=np.uint8),
    )
    assert expanded.shape == (2, 3, 1)


def test_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        CutMix(alpha=0)


def test_invalid_bbox_min_visibility() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CutMix(bbox_min_visibility=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CutMix(bbox_min_visibility=-0.1)
