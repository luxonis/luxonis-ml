from typing import Any

import albumentations as A
import cv2
import numpy as np
import pytest

from luxonis_ml.data.augmentations.custom.symetric_keypoints_flip import (
    HorizontalSymetricKeypointsFlip,
    TransposeSymmetricKeypoints,
    VerticalSymetricKeypointsFlip,
)


@pytest.fixture
def img() -> np.ndarray:
    return np.arange(6, dtype=np.uint8).reshape(2, 3, 1)


@pytest.fixture
def mask(img: np.ndarray) -> np.ndarray:
    return img.copy()


@pytest.fixture
def bboxes() -> np.ndarray:
    return np.array([[0.2, 0.3, 0.6, 0.8]], dtype=float)


@pytest.fixture
def keypoints_single() -> np.ndarray:
    return np.array([[1.0, 2.0]], dtype=float)


@pytest.fixture
def keypoints_pair() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )


def get_params(
    transform: A.DualTransform,
    img_shape: tuple[int, ...],
) -> dict[str, Any]:
    return transform.get_params_dependent_on_data({"shape": img_shape}, {})


def test_horizontal_flip_keypoints_single(
    keypoints_single: np.ndarray,
    img: np.ndarray,
) -> None:
    t = HorizontalSymetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, img.shape)
    out = t.apply_to_keypoints(keypoints_single, **params)
    orig_width = params["orig_width"]
    expected = np.array([[orig_width - 1.0, 2.0]])
    assert np.allclose(out, expected)


def test_vertical_flip_keypoints_single(
    keypoints_single: np.ndarray,
    img: np.ndarray,
) -> None:
    t = VerticalSymetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, img.shape)
    out = t.apply_to_keypoints(keypoints_single, **params)
    orig_height = params["orig_height"]
    expected = np.array([[1.0, orig_height - 2.0]])
    assert np.allclose(out, expected)


def test_transpose_keypoints_single(
    keypoints_single: np.ndarray,
    img: np.ndarray,
) -> None:
    t = TransposeSymmetricKeypoints(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, img.shape)
    out = t.apply_to_keypoints(keypoints_single, **params)
    r, c = keypoints_single[0]
    expected = np.array([[c, r]])
    assert np.allclose(out, expected)


@pytest.mark.parametrize(
    ("Transform", "flip_axis"),
    [
        (HorizontalSymetricKeypointsFlip, "horizontal"),
        (VerticalSymetricKeypointsFlip, "vertical"),
        (TransposeSymmetricKeypoints, "transpose"),
    ],
)
def test_flip_and_swap_keypoints_pair(
    Transform: type[A.DualTransform],
    flip_axis: str,
    keypoints_pair: np.ndarray,
    img: np.ndarray,
) -> None:
    t = Transform(keypoint_pairs=[(0, 1)], p=1.0)  # type: ignore
    params = get_params(t, img.shape)

    out = t.apply_to_keypoints(keypoints_pair, **params)

    flipped = keypoints_pair.copy()
    if flip_axis == "horizontal":
        orig_width = params["orig_width"]
        flipped[:, 0] = orig_width - flipped[:, 0]
    elif flip_axis == "vertical":
        orig_height = params["orig_height"]
        flipped[:, 1] = orig_height - flipped[:, 1]
    else:
        flipped = flipped[:, [1, 0]]

    expected = flipped.copy()
    expected[[0, 1]] = expected[[1, 0]]

    assert np.allclose(out, expected), (
        f"{Transform.__name__} did not correctly flip-and-swap:\n"
        f"expected\n{expected}\n but got\n{out}"
    )


def test_horizontal_flip_image_and_mask(
    img: np.ndarray,
    mask: np.ndarray,
) -> None:
    t = HorizontalSymetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, img.shape)
    assert np.array_equal(t.apply(img, **params), cv2.flip(img, 1))
    assert np.array_equal(t.apply_to_mask(mask, **params), cv2.flip(img, 1))


def test_horizontal_flip_bboxes(bboxes: np.ndarray) -> None:
    t = HorizontalSymetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, (2, 3, 1))
    out = t.apply_to_bboxes(bboxes, **params)
    expected = np.array([[1 - 0.6, 0.3, 1 - 0.2, 0.8]])
    assert np.allclose(out, expected)


def test_vertical_flip_image_and_mask(
    img: np.ndarray,
    mask: np.ndarray,
) -> None:
    t = VerticalSymetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, img.shape)
    assert np.array_equal(t.apply(img, **params), cv2.flip(img, 0))
    assert np.array_equal(t.apply_to_mask(mask, **params), cv2.flip(img, 0))


def test_vertical_flip_bboxes(bboxes: np.ndarray) -> None:
    t = VerticalSymetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, (2, 3, 1))
    out = t.apply_to_bboxes(bboxes, **params)
    expected = np.array([[0.2, 1 - 0.8, 0.6, 1 - 0.3]])
    assert np.allclose(out, expected)


def test_transpose_image_and_mask(
    img: np.ndarray,
    mask: np.ndarray,
) -> None:
    t = TransposeSymmetricKeypoints(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, img.shape)
    assert np.array_equal(
        t.apply(img, **params), img.transpose((1, 0, *range(2, img.ndim)))
    )
    assert np.array_equal(
        t.apply_to_mask(mask, **params),
        mask.transpose((1, 0, *range(2, mask.ndim))),
    )


def test_transpose_bboxes(bboxes: np.ndarray) -> None:
    t = TransposeSymmetricKeypoints(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, (2, 3, 1))
    out = t.apply_to_bboxes(bboxes, **params)
    x_min, y_min, x_max, y_max = bboxes[0]
    expected = np.array([[y_min, x_min, y_max, x_max]])
    assert np.allclose(out, expected)
