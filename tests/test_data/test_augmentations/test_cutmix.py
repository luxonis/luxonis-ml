import numpy as np
import pytest

from luxonis_ml.data.augmentations.custom.cutmix import CutMix


def test_compute_patch() -> None:
    assert CutMix._compute_patch(0.75, 8, 8, 4, 4) == (
        2,
        2,
        6,
        6,
        0.75,
    )

    x1, y1, x2, y2, lambda_value = CutMix._compute_patch(0.75, 8, 8, 0, 0)
    assert (x1, y1, x2, y2) == (0, 0, 2, 2)
    assert lambda_value == pytest.approx(0.9375)


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
            [0.4, 0.3, 0.8, 0.7, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(bboxes, expected)


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


def test_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        CutMix(alpha=0)
