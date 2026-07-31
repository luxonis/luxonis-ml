import numpy as np
import pytest
from pytest_subtests.plugin import SubTests

from luxonis_ml.data.augmentations.custom.mixup import MixUp


@pytest.mark.parametrize("augmentation_data", [2], indirect=True)
@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_mixup(
    height: int,
    width: int,
    augmentation_data: dict[str, list[np.ndarray]],
    subtests: SubTests,
    alpha: float,
):
    mixup = MixUp(p=1.0, alpha=alpha)
    transformed = mixup(**augmentation_data)  # type: ignore
    assert set(transformed.keys()) == set(augmentation_data.keys())
    for value in transformed.values():
        assert isinstance(value, np.ndarray)

    with subtests.test("image"):
        assert transformed["image"].shape == (height, width, 3)
        assert np.allclose(
            (
                augmentation_data["image"][0] * alpha
                + augmentation_data["image"][1] * (1 - alpha)
            ),
            transformed["image"],
        )

    with subtests.test("bboxes"):
        assert transformed["bboxes"].shape == (2, 6)
        assert np.allclose(
            np.concatenate(augmentation_data["bboxes"]), transformed["bboxes"]
        )

    with subtests.test("keypoints"):
        assert transformed["keypoints"].shape == (2, 5)
        assert np.allclose(
            np.concatenate(augmentation_data["keypoints"]),
            transformed["keypoints"],
        )

    with subtests.test("mask"):
        assert transformed["mask"].shape == (height, width, 1)
        assert np.allclose(
            augmentation_data["mask"][0].astype(bool)
            | augmentation_data["mask"][1].astype(bool),
            transformed["mask"].astype(bool),
        )


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        # A bare `np.array([])` has no width of its own and takes the
        # populated sample's.
        ([np.array([]), np.zeros((2, 7), np.float32)], (2, 7)),
        # With nothing populated the width still has to come from the empty
        # rows, not from the fallback.
        (
            [np.zeros((0, 7), np.float32), np.zeros((0, 7), np.float32)],
            (0, 7),
        ),
    ],
)
def test_mixup_pads_empty_samples_to_the_batch_width(
    batch: list[np.ndarray],
    expected: tuple[int, int],
    subtests: SubTests,
) -> None:
    """A sample with no boxes must not narrow the merged array."""
    mixup = MixUp(p=1.0)
    shapes = [(16, 16), (16, 16)]

    with subtests.test("bboxes"):
        assert mixup.apply_to_bboxes(list(batch), shapes).shape == expected

    with subtests.test("keypoints"):
        assert mixup.apply_to_keypoints(list(batch), shapes).shape == expected


def test_mixup_leaves_the_callers_batch_alone() -> None:
    mixup = MixUp(p=1.0)
    batch = [np.array([]), np.zeros((2, 7), dtype=np.float32)]

    mixup.apply_to_bboxes(list(batch), [(16, 16), (16, 16)])

    assert batch[0].shape == (0,)


def test_invalid():
    with pytest.raises(ValueError, match="must be in range"):
        MixUp(alpha=(-1, 1))
    with pytest.raises(ValueError, match="must be in range"):
        MixUp(alpha=(1, -1))
    with pytest.raises(ValueError, match="ascending order"):
        MixUp(alpha=(0.8, 0.2))
