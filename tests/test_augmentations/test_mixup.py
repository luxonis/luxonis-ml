from typing import Final

import numpy as np
import pytest
from pytest_subtests.plugin import SubTests

from luxonis_ml.data.augmentations.custom.mixup import MixUp

WIDTH: Final[int] = 640
HEIGHT: Final[int] = 480


def test_mixup(subtests: SubTests):
    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    mixup = MixUp(p=1.0)
    m = mixup.apply(
        image_batch=[img, img], image_shapes=[(HEIGHT, WIDTH), (HEIGHT, WIDTH)]
    )
    assert m.shape == (HEIGHT, WIDTH, 3)
    with subtests.test("image"):
        result = mixup(image=[img, img])
        assert result["image"].shape == (HEIGHT, WIDTH, 3)

    with subtests.test("bboxes"):
        result = mixup(
            image=[img, img],
            bboxes=[
                np.array(
                    [
                        [100, 100, 150, 175, 0, 0],
                    ]
                ),
                np.array(
                    [
                        [0, 0, 10, 10, 1, 1],
                    ]
                ),
            ],
        )
        assert result["image"].shape == (HEIGHT, WIDTH, 3)
        assert result["bboxes"].shape == (2, 6)


def test_invalid():
    with pytest.raises(ValueError):
        MixUp(alpha=(-1, 1))
    with pytest.raises(ValueError):
        MixUp(alpha=(1, -1))
    with pytest.raises(ValueError):
        MixUp(alpha=(0.8, 0.2))
