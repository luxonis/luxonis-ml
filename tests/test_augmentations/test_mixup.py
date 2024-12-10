from typing import Final

import numpy as np

from luxonis_ml.data.augmentations.custom.mixup import MixUp

WIDTH: Final[int] = 640
HEIGHT: Final[int] = 480


def test_mixup():
    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    mixup = MixUp(p=1.0)
    m = mixup.apply(
        image_batch=[img, img], image_shapes=[(HEIGHT, WIDTH), (HEIGHT, WIDTH)]
    )
    assert m.shape == (HEIGHT, WIDTH, 3)
    result = mixup(image=[img, img], labels={})
    assert result["image"].shape == (HEIGHT, WIDTH, 3)
