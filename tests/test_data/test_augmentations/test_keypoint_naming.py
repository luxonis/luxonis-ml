"""The engine names the augmented keypoints from what each name means.

Each check reads the picture, not the engine. A mirror partner keeps the
turn of the triangle it forms with a keypoint on the midline, because a
correct name puts the left eye of a person where a left eye can be. A
keypoint with a place in the picture must be the corner at that place.
Random transforms run with several seeds: the checks hold for any outcome.
"""

from typing import Any

import cv2
import numpy as np
import pytest

from luxonis_ml.data.augmentations import AlbumentationsEngine
from luxonis_ml.ldf import KeypointMetadata

SIZE = 64

# A person who faces the camera: the left side is on the right of the image.
PEOPLE = KeypointMetadata(
    labels=["nose", "left_eye", "right_eye", "left_hand", "right_hand"],
    flip_pairs=[(1, 2), (3, 4)],
)
PERSON = [0.50, 0.45, 2, 0.56, 0.40, 2, 0.44, 0.40, 2, 0.68, 0.60, 2]
PERSON += [0.32, 0.60, 2]

# A floor tile has no front, so its corners are named after the picture.
TILES = KeypointMetadata(
    labels=["a", "b", "c", "d"],
    position=["top_left", "top_right", "bottom_right", "bottom_left"],
)
TILE = [0.38, 0.38, 2, 0.62, 0.38, 2, 0.62, 0.62, 2, 0.38, 0.62, 2]

# Keypoints without a meaning keep their columns.
HANDS = KeypointMetadata(labels=["wrist", "tip", "thumb"])
HAND = [0.50, 0.70, 2, 0.50, 0.55, 2, 0.58, 0.62, 2]

TASKS = {"people": PEOPLE, "tiles": TILES, "hands": HANDS}
ROWS = {"people": PERSON, "tiles": TILE, "hands": HAND}
BOX = np.array([[0, 0.2, 0.2, 0.6, 0.6]])


def augment(
    config: list[dict[str, Any]],
    seed: int = 0,
    batch: int = 1,
    tasks: dict[str, KeypointMetadata] = TASKS,
) -> dict[str, np.ndarray]:
    """Augment the scene, and return the keypoints of each task."""
    targets = {}
    for task in tasks:
        targets[f"{task}/boundingbox"] = "boundingbox"
        targets[f"{task}/keypoints"] = "keypoints"
    engine = AlbumentationsEngine(
        SIZE,
        SIZE,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        config,
        keypoint_metadata=tasks,
        seed=seed,
    )
    labels = {}
    for task, row in ROWS.items():
        labels[f"{task}/boundingbox"] = BOX
        labels[f"{task}/keypoints"] = np.array([row], dtype=float)
    sample = ({"image": np.zeros((SIZE, SIZE, 3))}, labels)
    _, out = engine.apply([sample] * batch)
    return {
        task: out[f"{task}/keypoints"].reshape(-1, len(meta.labels), 3)
        for task, meta in tasks.items()
    }


def turn(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the sign of the turn from ``a`` over ``b`` to ``c``."""
    return float(
        np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    )


def assert_people_named(people: np.ndarray) -> None:
    original = np.array(PERSON).reshape(-1, 3)
    for person in people:
        if (person[:, 2] == 0).any():
            continue
        for left, right in PEOPLE.flip_pairs:
            assert turn(person[left], person[right], person[0]) == turn(
                original[left], original[right], original[0]
            )


def assert_tiles_named(tiles: np.ndarray) -> None:
    for tile in tiles:
        if (tile[:, 2] == 0).any():
            continue
        x, y = tile[:, 0], tile[:, 1]
        # a: top left, b: top right, c: bottom right, d: bottom left.
        assert np.argmin(x + y) == 0
        assert np.argmin(y - x) == 1
        assert np.argmax(x + y) == 2
        assert np.argmax(y - x) == 3


FLIP = {"name": "HorizontalFlip", "params": {"p": 1.0}}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "config",
    [
        pytest.param([FLIP], id="HorizontalFlip"),
        pytest.param([{"name": "VerticalFlip", "params": {"p": 1}}], id="v"),
        pytest.param([{"name": "Transpose", "params": {"p": 1}}], id="t"),
        pytest.param(
            [{"name": "RandomRotate90", "params": {"p": 1}}], id="rot90"
        ),
        pytest.param([{"name": "D4", "params": {"p": 1}}], id="D4"),
        pytest.param(
            [FLIP, {"name": "VerticalFlip", "params": {"p": 1}}],
            id="two-flips",
        ),
        pytest.param(
            [{"name": "Affine", "params": {"rotate": (-30, 30), "p": 1}}],
            id="Affine",
        ),
        pytest.param(
            [{"name": "Perspective", "params": {"scale": 0.08, "p": 1}}],
            id="Perspective",
        ),
        pytest.param(
            [
                {
                    "name": "OneOf",
                    "params": {"p": 1, "transforms": [FLIP]},
                }
            ],
            id="OneOf",
        ),
    ],
)
def test_names_follow_their_meaning(
    config: list[dict[str, Any]], seed: int
) -> None:
    out = augment(config, seed)

    assert_people_named(out["people"])
    assert_tiles_named(out["tiles"])


@pytest.mark.parametrize("seed", [1, 2])
def test_names_follow_their_meaning_in_a_mosaic(seed: int) -> None:
    """The probes travel with each instance through the batch transform.

    These seeds keep at least one whole person and one whole tile in the
    crop of the mosaic.
    """
    mosaic = {
        "name": "Mosaic4",
        "params": {"out_width": 2 * SIZE, "out_height": 2 * SIZE, "p": 1.0},
    }

    out = augment([mosaic, FLIP], seed, batch=4)

    assert any((person[:, 2] > 0).all() for person in out["people"])
    assert any((tile[:, 2] > 0).all() for tile in out["tiles"])
    assert_people_named(out["people"])
    assert_tiles_named(out["tiles"])


def test_a_task_without_meaning_keeps_its_columns() -> None:
    out = augment([FLIP])

    hand = np.array(HAND).reshape(-1, 3)
    assert np.allclose(out["hands"][0, :, 0], 1 - hand[:, 0], atol=0.02)
    assert np.allclose(out["hands"][0, :, 1], hand[:, 1], atol=0.02)


def test_a_rotation_keeps_the_mirror_partners() -> None:
    """Two flips make a rotation by 180°, which no side survives as a swap."""
    out = augment([FLIP, {"name": "VerticalFlip", "params": {"p": 1}}])

    person = out["people"][0]
    # Upside down and facing the camera: the left eye is on the image left.
    assert person[1, 0] < person[2, 0]


def test_keypoints_that_a_distortion_drops_come_back_unlabeled() -> None:
    """``ElasticTransform`` drops keypoints outside the image.

    Before, the rows of an instance then no longer split evenly, and the
    loader failed with ``cannot reshape array``.
    """
    shift = {
        "name": "Affine",
        "params": {"translate_percent": {"x": 0.4, "y": 0.0}, "p": 1},
    }
    elastic = {"name": "ElasticTransform", "params": {"p": 1}}

    out = augment([shift, elastic])

    assert out["people"].shape == (1, 5, 3)
    # The left hand ends up outside the image.
    assert out["people"][0, 3].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("seed", range(6))
def test_names_survive_a_heavy_pipeline(seed: int) -> None:
    config = [
        {"name": "D4", "params": {"p": 0.5}},
        {"name": "Affine", "params": {"rotate": (-25, 25), "p": 0.7}},
        {"name": "Perspective", "params": {"scale": 0.06, "p": 0.5}},
        {"name": "ElasticTransform", "params": {"p": 0.5}},
        {"name": "GridDistortion", "params": {"p": 0.5}},
        FLIP,
    ]

    out = augment(config, seed)

    assert_people_named(out["people"])
    assert_tiles_named(out["tiles"])


def test_the_deprecated_flips_give_partners_to_a_task_without_them() -> None:
    config = [
        {
            "name": "HorizontalSymmetricKeypointsFlip",
            "params": {"p": 1.0, "keypoint_pairs": [[1, 2], [3, 4]]},
        }
    ]
    unpaired = {**TASKS, "people": KeypointMetadata(labels=PEOPLE.labels)}

    out = augment(config, tasks=unpaired)

    assert_people_named(out["people"])


def test_identity_pairs_of_a_deprecated_flip_change_nothing() -> None:
    """An old config gave ``[[0, 0]]`` to keep the names of a vertical flip.

    The stored partners still swap: every mirror swaps the sides of a body.
    """
    config = [
        {
            "name": "VerticalSymmetricKeypointsFlip",
            "params": {"p": 1.0, "keypoint_pairs": [[0, 0]]},
        }
    ]

    out = augment(config)

    assert_people_named(out["people"])


def test_reflect_padding_keeps_the_original_keypoints() -> None:
    """Reflect padding adds reflected copies of each keypoint.

    The copies have the id of their keypoint. Before, the last copy won,
    so a plain shift moved the keypoints outside the image.
    """

    def shift(border: int) -> dict[str, Any]:
        return {
            "name": "Affine",
            "params": {
                "translate_percent": {"x": -0.3, "y": 0.0},
                "border_mode": border,
                "p": 1.0,
            },
        }

    constant = augment([shift(cv2.BORDER_CONSTANT)])
    reflect = augment([shift(cv2.BORDER_REFLECT_101)])

    for task, keypoints in constant.items():
        assert np.allclose(reflect[task], keypoints)
