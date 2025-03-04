from typing import Dict, List

import numpy as np
import pytest
from pytest_subtests.plugin import SubTests

from luxonis_ml.data.augmentations.custom.mixup import MixUp


@pytest.mark.parametrize("augmentation_data", [2], indirect=True)
@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_mixup(
    height: int,
    width: int,
    augmentation_data: Dict[str, List[np.ndarray]],
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


def test_invalid():
    with pytest.raises(ValueError, match="must be in range"):
        MixUp(alpha=(-1, 1))
    with pytest.raises(ValueError, match="must be in range"):
        MixUp(alpha=(1, -1))
    with pytest.raises(ValueError, match="ascending order"):
        MixUp(alpha=(0.8, 0.2))
