from typing import Final

import numpy as np
import pytest

from luxonis_ml.data.augmentations.custom.mosaic import (
    Mosaic4,
    bbox_mosaic4,
    keypoint_mosaic4,
    mosaic4,
)

WIDTH: Final[int] = 640
HEIGHT: Final[int] = 480


def test_mosaic4():
    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    mosaic = mosaic4([img, img, img, img], HEIGHT, WIDTH, 0, 0, 0)
    assert mosaic.shape == (HEIGHT, WIDTH, 3)


def test_bbox_mosaic4():
    bbox = np.array([0, 0, WIDTH, HEIGHT])[np.newaxis, ...]
    for i in range(4):
        mosaic_bbox = bbox_mosaic4(
            bbox, HEIGHT // 2, WIDTH // 2, i, HEIGHT, WIDTH, 0, 0
        )[0].tolist()
        assert pytest.approx(mosaic_bbox, abs=1) == [
            0,
            0,
            WIDTH // 2,
            HEIGHT // 2,
        ]


def test_keypoint_mosaic4():
    for i, (w, h) in enumerate(
        [
            (WIDTH // 2, HEIGHT // 2),
            (WIDTH, HEIGHT // 2),
            (WIDTH // 2, HEIGHT),
            (WIDTH, HEIGHT),
        ]
    ):
        mosaic_keypoint = keypoint_mosaic4(
            np.array([w, h, 0, 0])[np.newaxis, ...],
            HEIGHT // 2,
            WIDTH // 2,
            i,
            HEIGHT,
            WIDTH,
            0,
            0,
        )[0].tolist()
        assert pytest.approx(mosaic_keypoint, abs=0.25) == [w * 2, h * 2, 0, 0]


def test_Mosaic4():
    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    mosaic4 = Mosaic4(out_height=HEIGHT, out_width=WIDTH, p=1.0)
    m = mosaic4(image=[img, img, img, img], labels={})
    assert m["image"].shape == (HEIGHT, WIDTH, 3)
