from typing import Any

import albumentations as A
import cv2
import numpy as np
import pytest

from luxonis_ml.data.augmentations import AlbumentationsEngine
from luxonis_ml.data.augmentations.custom.symmetric_keypoints_flip import (
    HorizontalSymmetricKeypointsFlip,
    TransposeSymmetricKeypoints,
    VerticalSymmetricKeypointsFlip,
)
from luxonis_ml.ldf import KeypointMetadata


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
    t = HorizontalSymmetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, img.shape)
    out = t.apply_to_keypoints(keypoints_single, **params)
    orig_width = params["orig_width"]
    expected = np.array([[orig_width - 1.0, 2.0]])
    assert np.allclose(out, expected)


def test_vertical_flip_keypoints_single(
    keypoints_single: np.ndarray,
    img: np.ndarray,
) -> None:
    t = VerticalSymmetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
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
        (HorizontalSymmetricKeypointsFlip, "horizontal"),
        (VerticalSymmetricKeypointsFlip, "vertical"),
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
    t = HorizontalSymmetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, img.shape)
    assert np.array_equal(t.apply(img, **params), cv2.flip(img, 1))
    assert np.array_equal(t.apply_to_mask(mask, **params), cv2.flip(mask, 1))


def test_horizontal_flip_bboxes(bboxes: np.ndarray) -> None:
    t = HorizontalSymmetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, (2, 3, 1))
    out = t.apply_to_bboxes(bboxes, **params)
    expected = np.array([[1 - 0.6, 0.3, 1 - 0.2, 0.8]])
    assert np.allclose(out, expected)


def test_vertical_flip_image_and_mask(
    img: np.ndarray,
    mask: np.ndarray,
) -> None:
    t = VerticalSymmetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
    params = get_params(t, img.shape)
    assert np.array_equal(t.apply(img, **params), cv2.flip(img, 0))
    assert np.array_equal(t.apply_to_mask(mask, **params), cv2.flip(img, 0))


def test_vertical_flip_bboxes(bboxes: np.ndarray) -> None:
    t = VerticalSymmetricKeypointsFlip(keypoint_pairs=[(0, 0)], p=1.0)
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


def test_an_unpaired_keypoint_keeps_the_instances_apart() -> None:
    # The COCO pairs leave out the nose, so they name 16 of 17 keypoints.
    # Each instance here is a nose, a left and a right keypoint.
    t = HorizontalSymmetricKeypointsFlip(keypoint_pairs=[(1, 2)], p=1.0)
    first = [[5.0, 0.0], [2.0, 0.0], [8.0, 0.0]]
    second = [[5.0, 9.0], [2.0, 9.0], [8.0, 9.0]]

    out = t.apply_to_keypoints(np.array(first + second), orig_width=10)

    # The flip moves left to x=8, and the swap gives the name back to x=2.
    assert out[:, 0].tolist() == [5.0, 2.0, 8.0, 5.0, 2.0, 8.0]
    assert out[:, 1].tolist() == [0.0, 0.0, 0.0, 9.0, 9.0, 9.0]


def test_vertical_flip_mirrors_keypoints_across_the_image_height() -> None:
    compose = A.Compose(
        [VerticalSymmetricKeypointsFlip(p=1.0)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    out = compose(
        image=np.zeros((10, 40, 3), np.uint8),
        keypoints=np.array([[5.0, 2.0]]),
    )

    assert np.allclose(out["keypoints"], [[5.0, 8.0]])


FLIP = {"name": "HorizontalSymmetricKeypointsFlip", "params": {"p": 1.0}}
ONE_OF_FLIP = {"name": "OneOf", "params": {"p": 1.0, "transforms": [FLIP]}}
# A hand with four keypoints whose outer two are a pair.
HANDS = KeypointMetadata(
    labels=["a", "b", "c", "d"], flip_pairs={"horizontal": [(0, 3)]}
)


def augment_people_and_hands(
    augmentation: dict[str, Any],
    people: KeypointMetadata,
    people_keypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a person and a hand, and return their keypoint x and y."""
    targets = {
        "people/boundingbox": "boundingbox",
        "people/keypoints": "keypoints",
        "hands/boundingbox": "boundingbox",
        "hands/keypoints": "keypoints",
    }
    engine = AlbumentationsEngine(
        20,
        20,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        [augmentation],
        keypoint_metadata={"people": people, "hands": HANDS},
    )
    box = np.array([[0, 0.1, 0.1, 0.8, 0.8]])
    labels = {
        "people/boundingbox": box,
        "people/keypoints": people_keypoints,
        "hands/boundingbox": box,
        "hands/keypoints": np.array(
            [[0.1, 0.5, 2, 0.3, 0.5, 2, 0.6, 0.5, 2, 0.9, 0.6, 2]]
        ),
    }

    _, out = engine.apply([({"image": np.zeros((20, 20, 3))}, labels)])

    return (
        out["people/keypoints"].reshape(-1, 3)[:, :2],
        out["hands/keypoints"].reshape(-1, 3)[:, :2],
    )


@pytest.mark.parametrize(
    "augmentation",
    [
        pytest.param(FLIP, id="top-level"),
        pytest.param(ONE_OF_FLIP, id="OneOf"),
        pytest.param(
            {
                "name": "Sequential",
                "params": {"p": 1.0, "transforms": [ONE_OF_FLIP]},
            },
            id="Sequential-OneOf",
        ),
    ],
)
def test_each_keypoint_task_swaps_its_own_stored_pairs(
    augmentation: dict[str, Any],
) -> None:
    # A person with a nose and two eyes. The flip has no pairs of its own.
    people, hands = augment_people_and_hands(
        augmentation,
        KeypointMetadata(
            labels=["nose", "left", "right"],
            flip_pairs={"horizontal": [(1, 2)]},
        ),
        np.array([[0.5, 0.2, 2, 0.2, 0.3, 2, 0.8, 0.4, 2]]),
    )

    assert np.allclose(people, [[0.5, 0.2], [0.2, 0.4], [0.8, 0.3]])
    assert np.allclose(hands, [[0.1, 0.6], [0.7, 0.5], [0.4, 0.5], [0.9, 0.5]])


def test_configured_pairs_replace_the_stored_pairs_they_fit() -> None:
    # The configured pair swaps only the ears of the person. The hands have
    # fewer keypoints than the configured pair indexes, so they swap their
    # own stored pair.
    people, hands = augment_people_and_hands(
        {
            "name": "HorizontalSymmetricKeypointsFlip",
            "params": {"p": 1.0, "keypoint_pairs": [(3, 4)]},
        },
        KeypointMetadata(
            labels=["nose", "l_eye", "r_eye", "l_ear", "r_ear"],
            flip_pairs={"horizontal": [(1, 2), (3, 4)]},
        ),
        np.array(
            [
                [0.5, 0.2, 2],
                [0.4, 0.3, 2],
                [0.6, 0.3, 2],
                [0.3, 0.25, 2],
                [0.7, 0.25, 2],
            ]
        ).reshape(1, -1),
    )

    assert np.allclose(
        people, [[0.5, 0.2], [0.6, 0.3], [0.4, 0.3], [0.3, 0.25], [0.7, 0.25]]
    )
    assert np.allclose(hands, [[0.1, 0.6], [0.7, 0.5], [0.4, 0.5], [0.9, 0.5]])


# The corners of a plate are named after the sides of the image, so each
# mirror swaps other corners.
PLATE = KeypointMetadata.model_validate(
    {
        "labels": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "flip_pairs": {
            "horizontal": [
                ("top_left", "top_right"),
                ("bottom_left", "bottom_right"),
            ],
            "vertical": [
                ("top_left", "bottom_left"),
                ("top_right", "bottom_right"),
            ],
            "transpose": [("top_right", "bottom_left")],
        },
    }
)


@pytest.mark.parametrize(
    "name",
    [
        "HorizontalSymmetricKeypointsFlip",
        "VerticalSymmetricKeypointsFlip",
        "TransposeSymmetricKeypoints",
    ],
)
def test_each_flip_swaps_the_stored_pairs_of_its_mirror(name: str) -> None:
    corners, _ = augment_people_and_hands(
        {"name": name, "params": {"p": 1.0}},
        PLATE,
        np.array([0.2, 0.3, 2, 0.8, 0.3, 2, 0.8, 0.7, 2, 0.2, 0.7, 2]).reshape(
            1, -1
        ),
    )

    # Each corner is still in the corner of the image that names it.
    x, y = corners.T
    assert (x < 0.5).tolist() == [True, False, False, True]
    assert (y < 0.5).tolist() == [True, True, False, False]


def test_a_mirror_without_stored_pairs_swaps_no_keypoint() -> None:
    # Both tasks store only horizontal pairs, so the vertical flip moves
    # the keypoints and keeps their order.
    people, hands = augment_people_and_hands(
        {"name": "VerticalSymmetricKeypointsFlip", "params": {"p": 1.0}},
        KeypointMetadata(
            labels=["nose", "left", "right"],
            flip_pairs={"horizontal": [(1, 2)]},
        ),
        np.array([[0.5, 0.2, 2, 0.2, 0.3, 2, 0.8, 0.4, 2]]),
    )

    assert np.allclose(people, [[0.5, 0.8], [0.2, 0.7], [0.8, 0.6]])
    assert np.allclose(hands, [[0.1, 0.5], [0.3, 0.5], [0.6, 0.5], [0.9, 0.4]])
