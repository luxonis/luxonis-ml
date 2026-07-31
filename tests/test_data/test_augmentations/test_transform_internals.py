import numpy as np
import pytest

from luxonis_ml.data.augmentations import MixUp, Mosaic4
from luxonis_ml.data.augmentations.custom import (
    HorizontalSymmetricKeypointsFlip,
    TransposeSymmetricKeypoints,
    VerticalSymmetricKeypointsFlip,
)

from .helpers import KeepFirstSample

SHAPES = [(16, 16), (16, 16)]

FLIPS = [
    HorizontalSymmetricKeypointsFlip,
    VerticalSymmetricKeypointsFlip,
    TransposeSymmetricKeypoints,
]


def test_mixup_can_resize_without_keeping_aspect_ratio() -> None:
    mixup = MixUp(keep_aspect_ratio=False, p=1.0)

    output = mixup.apply(
        [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.full((8, 32, 3), 255, dtype=np.uint8),
        ],
        image_shapes=[(16, 16), (8, 32)],
        alpha=0.5,
    )

    assert output.shape == (16, 16, 3)


@pytest.mark.parametrize("empty_index", [0, 1])
def test_mixup_semantic_mask_tolerates_a_missing_mask(
    empty_index: int,
) -> None:
    masks: list[np.ndarray] = [
        np.ones((16, 16, 1), dtype=np.uint8),
        np.ones((16, 16, 1), dtype=np.uint8),
    ]
    masks[empty_index] = np.array([])

    output = MixUp(p=1.0).apply_to_mask(masks, image_shapes=SHAPES, alpha=0.5)

    assert output.shape == (16, 16, 1)


@pytest.mark.parametrize("empty_index", [0, 1])
def test_mixup_instance_mask_tolerates_a_missing_mask(
    empty_index: int,
) -> None:
    masks: list[np.ndarray] = [
        np.ones((16, 16, 2), dtype=np.uint8),
        np.ones((16, 16, 3), dtype=np.uint8),
    ]
    expected = masks[1 - empty_index].shape
    masks[empty_index] = np.array([])

    output = MixUp(p=1.0).apply_to_instance_mask(masks, image_shapes=SHAPES)

    assert output.shape == expected


def test_mixup_masks_regain_a_channel_axis_after_resize() -> None:
    """Only the plain-resize path drops it; `LetterboxResize` keeps one."""
    masks = [
        np.ones((16, 16, 1), dtype=np.uint8),
        np.ones((8, 8), dtype=np.uint8),
    ]

    output = MixUp(keep_aspect_ratio=False, p=1.0).apply_to_mask(
        masks, image_shapes=[(16, 16), (8, 8)], alpha=0.5
    )

    assert output.ndim == 3


def test_mixup_instance_masks_regain_a_channel_axis_after_resize() -> None:
    masks = [
        np.ones((16, 16, 1), dtype=np.uint8),
        np.ones((8, 8), dtype=np.uint8),
    ]

    output = MixUp(keep_aspect_ratio=False, p=1.0).apply_to_instance_mask(
        masks, image_shapes=[(16, 16), (8, 8)]
    )

    assert output.shape == (16, 16, 2)


@pytest.mark.parametrize("ndim", [2, 3])
def test_mosaic_semantic_mask_fills_in_missing_masks(ndim: int) -> None:
    shape = (16, 16) if ndim == 2 else (16, 16, 2)
    # A sample missing the task arrives with the spatial dims collapsed,
    # which is how `_preprocess_batch` represents "no mask here".
    empty_shape = (0, 0) if ndim == 2 else (0, 0, 2)
    masks = [
        np.ones(shape, dtype=np.uint8),
        np.empty(empty_shape, dtype=np.uint8),
        np.ones(shape, dtype=np.uint8),
        np.empty(empty_shape, dtype=np.uint8),
    ]

    output = Mosaic4(height=16, width=16, p=1.0).apply_to_mask(
        masks, x_crop=4, y_crop=4, out_height=16, out_width=16
    )

    assert output.shape[:2] == (16, 16)


def test_mosaic_instance_masks_with_nothing_to_place() -> None:
    output = Mosaic4._apply_mosaic4_to_instance_masks(
        [np.array([]) for _ in range(4)],
        out_height=16,
        out_width=16,
        x_crop=0,
        y_crop=0,
    )

    assert output.shape == (16, 16, 0)


def test_mosaic_composes_images_without_a_channel_axis() -> None:
    images = [np.full((16, 16), i, dtype=np.uint8) for i in range(4)]

    output = Mosaic4._apply_mosaic4_to_images(
        images, out_height=16, out_width=16, x_crop=8, y_crop=8
    )

    assert output.shape == (16, 16)


def test_classification_defaults_to_a_single_absent_class() -> None:
    """Samples missing a classification task must not break the OR."""
    output = KeepFirstSample().apply_to_classification(
        [np.array([]), np.array([1.0])]
    )

    assert output.tolist() == [1.0]


@pytest.mark.parametrize("flip", FLIPS)
def test_flips_pass_empty_bboxes_through(flip: type) -> None:
    transform = flip(keypoint_pairs=[(0, 1)], p=1.0)

    output = transform.apply_to_bboxes(
        np.zeros((0, 4)), orig_width=8, orig_height=8
    )

    assert output.shape == (0, 4)


@pytest.mark.parametrize("flip", FLIPS)
def test_flips_pass_empty_keypoints_through(flip: type) -> None:
    transform = flip(keypoint_pairs=[(0, 1)], p=1.0)

    output = transform.apply_to_keypoints(
        np.zeros((0, 3)), orig_width=8, orig_height=8
    )

    assert output.shape == (0, 3)


@pytest.mark.parametrize("flip", FLIPS)
def test_flips_reject_a_partial_final_instance(flip: type) -> None:
    transform = flip(keypoint_pairs=[(0, 1)], p=1.0)
    keypoints = np.zeros((3, 3))

    with pytest.raises(ValueError, match="not a multiple of n_keypoints"):
        transform.apply_to_keypoints(keypoints, orig_width=8, orig_height=8)
