from typing import Final

import numpy as np

from luxonis_ml.data.augmentations.custom.mixup import MixUp

WIDTH: Final[int] = 640
HEIGHT: Final[int] = 480


def test_mixup():
    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    mixup = MixUp(p=1.0, always_apply=True)
    m = mixup.apply_to_image_batch(
        image_batch=[img, img], image_shapes=[(HEIGHT, WIDTH), (HEIGHT, WIDTH)]
    )
    assert m[0].shape == (HEIGHT, WIDTH, 3)
    result = mixup(image_batch=[img, img], labels={})
    assert result["image_batch"][0].shape == (HEIGHT, WIDTH, 3)
