from typing import Final

import numpy as np
import pytest

from luxonis_ml.data.augmentations.custom.mosaic import (
    bbox_mosaic4,
    keypoint_mosaic4,
    mosaic4,
)

WIDTH: Final[int] = 640
HEIGHT: Final[int] = 480


def test_mosaic4():
    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    mosaic = mosaic4([img, img, img, img], HEIGHT, WIDTH)
    assert mosaic.shape == (HEIGHT, WIDTH, 3)


def test_bbox_mosaic4():
    bbox = (0, 0, WIDTH, HEIGHT)
    for i in range(4):
        mosaic_bbox = bbox_mosaic4(bbox, HEIGHT // 2, WIDTH // 2, i, HEIGHT, WIDTH)
        assert pytest.approx(mosaic_bbox, abs=0.5) == (0, 0, WIDTH // 2, HEIGHT // 2)


def test_keypoint_mosaic4():
    keypoint = (WIDTH // 2, HEIGHT // 2, 0, 0)
    for i, (w, h) in enumerate(
        [
            (WIDTH // 2, HEIGHT // 2),
            (WIDTH, HEIGHT // 2),
            (WIDTH // 2, HEIGHT),
            (WIDTH, HEIGHT),
        ]
    ):
        mosaic_keypoint = keypoint_mosaic4(
            keypoint, HEIGHT // 2, WIDTH // 2, i, HEIGHT, WIDTH
        )
        assert pytest.approx(mosaic_keypoint, abs=0.25) == (w, h, 0, 0)
