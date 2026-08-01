from typing import Final

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "fill", [300, -1, (300, 20, 30), (10, 20, -5), (0, 0, 256)]
)
def test_letterbox_rejects_out_of_range_fill_values(
    fill: int | tuple[int, int, int],
) -> None:
    """A fill value outside [0, 255] is a config error, not something to clamp.

    The shared `Color.parse` clamps, which would silently pad every letterboxed
    image with a different color than the config asked for.
    """
    with pytest.raises(ValueError, match=r"out of range \[0, 255\]"):
        LetterboxResize(HEIGHT, WIDTH, image_fill_value=fill)


def test_letterbox_rejects_out_of_range_mask_fill_value() -> None:
    with pytest.raises(ValueError, match="mask_fill_value"):
        LetterboxResize(HEIGHT, WIDTH, mask_fill_value=999)


def test_letterbox_accepts_the_full_valid_range() -> None:
    assert LetterboxResize(
        HEIGHT, WIDTH, image_fill_value=(0, 128, 255)
    )._image_fill_value == (0, 128, 255)
    assert LetterboxResize(
        HEIGHT, WIDTH, image_fill_value=255
    )._image_fill_value == (255, 255, 255)
