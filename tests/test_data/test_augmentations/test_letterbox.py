from typing import Final

import numpy as np

from luxonis_ml.data.augmentations.custom.letterbox_resize import (
    LetterboxResize,
)

WIDTH: Final[int] = 640
HEIGHT: Final[int] = 480


def test_letterbox():
    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    letterbox = LetterboxResize(HEIGHT, WIDTH, p=1.0)
    x = letterbox(image=img, labels={})
    assert x["image"].shape == (HEIGHT, WIDTH, 3)
