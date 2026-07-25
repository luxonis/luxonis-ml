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


def test_letterbox_fill_values_resolve_to_rgb():
    # Fill colors resolve through the shared Color to proper 0-255 RGB.
    assert LetterboxResize(HEIGHT, WIDTH)._image_fill_value == (0, 0, 0)
    assert LetterboxResize(
        HEIGHT, WIDTH, image_fill_value="white"
    )._image_fill_value == (255, 255, 255)
    assert LetterboxResize(
        HEIGHT, WIDTH, image_fill_value=(10, 20, 30)
    )._image_fill_value == (10, 20, 30)
