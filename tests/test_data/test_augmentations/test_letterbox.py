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


@pytest.mark.parametrize(
    ("fill", "expected"),
    [
        ("white", (255, 255, 255)),
        ("red", (255, 0, 0)),
        ("#0000ff", (0, 0, 255)),
        (128, (128, 128, 128)),
        (255, (255, 255, 255)),  # the top of the valid range
        ((0, 128, 255), (0, 128, 255)),
    ],
)
def test_letterbox_pads_with_the_requested_color(
    fill: str | int | tuple[int, int, int],
    expected: tuple[int, int, int],
):
    """Fill colors used to resolve to 0-1 floats, padding with near-black."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    letterbox = LetterboxResize(HEIGHT, WIDTH, image_fill_value=fill)
    padded = letterbox(image=img, labels={})["image"]
    # A square image in a 4:3 frame is padded on the left and right only.
    assert (padded[:, 0] == expected).all()
    assert (padded[:, -1] == expected).all()


def test_letterbox_pads_with_black_by_default():
    """The default fill is the one most configs use, so it gets its own check.

    The source image is deliberately *not* black: padding black onto a black
    image would pass whether or not the default resolved correctly.
    """
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    padded = LetterboxResize(HEIGHT, WIDTH)(image=img, labels={})["image"]
    assert (padded[:, 0] == (0, 0, 0)).all()
    assert (padded[:, -1] == (0, 0, 0)).all()


@pytest.mark.parametrize(
    "fill", [300, -1, (300, 20, 30), (10, 20, -5), (0, 0, 256)]
)
def test_letterbox_rejects_out_of_range_fill_values(
    fill: int | tuple[int, int, int],
):
    """A fill value outside [0, 255] is a config error, not something to clamp.

    `Color.parse` clamps, which would silently pad every letterboxed image with
    a different color than the config asked for.
    """
    with pytest.raises(ValueError, match=r"out of range \[0, 255\]"):
        LetterboxResize(HEIGHT, WIDTH, image_fill_value=fill)


def test_letterbox_rejects_out_of_range_mask_fill_value():
    with pytest.raises(ValueError, match="mask_fill_value"):
        LetterboxResize(HEIGHT, WIDTH, mask_fill_value=999)
