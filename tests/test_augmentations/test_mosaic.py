from typing import Final

import numpy as np
import pytest
from pytest_subtests.plugin import SubTests

from luxonis_ml.data.augmentations.custom.mosaic import (
    Mosaic4,
    apply_mosaic4_to_bboxes,
    apply_mosaic4_to_images,
    apply_mosaic4_to_keypoints,
)

WIDTH: Final[int] = 640
HEIGHT: Final[int] = 480


def test_mosaic4_helpers(subtests: SubTests):
    with subtests.test("image"):
        img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
        mosaic = apply_mosaic4_to_images(
            [img, img, img, img], HEIGHT, WIDTH, 0, 0, 0
        )
        assert mosaic.shape == (HEIGHT, WIDTH, 3)

    with subtests.test("bboxes"):
        bbox = np.array([0, 0, WIDTH, HEIGHT])[np.newaxis, ...]
        for i in range(4):
            mosaic_bbox = apply_mosaic4_to_bboxes(
                bbox, HEIGHT // 2, WIDTH // 2, i, HEIGHT, WIDTH, 0, 0
            )[0].tolist()
            assert pytest.approx(mosaic_bbox, abs=1) == [
                0,
                0,
                WIDTH // 2,
                HEIGHT // 2,
            ]

    with subtests.test("keypoints"):
        for i, (w, h) in enumerate(
            [
                (WIDTH // 2, HEIGHT // 2),
                (WIDTH, HEIGHT // 2),
                (WIDTH // 2, HEIGHT),
                (WIDTH, HEIGHT),
            ]
        ):
            mosaic_keypoint = apply_mosaic4_to_keypoints(
                np.array([w, h, 0, 0])[np.newaxis, ...],
                HEIGHT // 2,
                WIDTH // 2,
                i,
                HEIGHT,
                WIDTH,
                0,
                0,
            )[0].tolist()
            assert pytest.approx(mosaic_keypoint, abs=0.25) == [
                w * 2,
                h * 2,
                0,
                0,
            ]


def test_mosaic4():
    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    mosaic4 = Mosaic4(out_height=HEIGHT, out_width=WIDTH, p=1.0)
    m = mosaic4(image=[img, img, img, img])
    assert m["image"].shape == (HEIGHT, WIDTH, 3)


def test_invalid():
    with pytest.raises(ValueError):
        Mosaic4(out_height=0, out_width=WIDTH)

    with pytest.raises(ValueError):
        Mosaic4(out_height=HEIGHT, out_width=0)

    mosaic4 = Mosaic4(out_height=HEIGHT, out_width=WIDTH, p=1.0)

    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    with pytest.raises(ValueError):
        mosaic4(image=[img, img, img])

    with pytest.raises(ValueError):
        mosaic4(image=[img, img, img, np.zeros((HEIGHT, WIDTH, 4))])
