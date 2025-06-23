from typing import Final

import numpy as np
import pytest
from albumentations.core.bbox_utils import normalize_bboxes
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
        bbox = np.array([0, 0, WIDTH, HEIGHT])[None, ...]
        expected_bbox = [
            [0, 0, 1, 1],
            [1, 0, 2, 1],
            [0, 1, 1, 2],
            [1, 1, 2, 2],
        ]
        for i in range(4):
            normalized_bbox = normalize_bboxes(bbox, (HEIGHT, WIDTH))
            mosaic_bbox = apply_mosaic4_to_bboxes(
                normalized_bbox,
                HEIGHT // 2,
                WIDTH // 2,
                i,
                HEIGHT,
                WIDTH,
                0,
                0,
            )[0].tolist()
            assert pytest.approx(mosaic_bbox, abs=0.01) == expected_bbox[i]

    with subtests.test("keypoints"):
        expected_kpts = [
            [WIDTH, HEIGHT, 0, 2],
            [3 * WIDTH, HEIGHT, 0, 0],
            [WIDTH, 3 * HEIGHT, 0, 0],
            [3 * WIDTH, 3 * HEIGHT, 0, 0],
        ]
        for i, (w, h) in enumerate(
            [
                (WIDTH // 2, HEIGHT // 2),
                (WIDTH, HEIGHT // 2),
                (WIDTH // 2, HEIGHT),
                (WIDTH, HEIGHT),
            ]
        ):
            mosaic_keypoint = apply_mosaic4_to_keypoints(
                np.array([w, h, 0, 2])[None, ...],
                HEIGHT // 2,
                WIDTH // 2,
                i,
                HEIGHT,
                WIDTH,
                0,
                0,
            )[0].tolist()
            assert pytest.approx(mosaic_keypoint, abs=0.1) == expected_kpts[i]


def test_mosaic4():
    img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
    mosaic4 = Mosaic4(out_height=HEIGHT, out_width=WIDTH, p=1.0)
    m = mosaic4(image=[img, img, img, img])
    assert m["image"].shape == (HEIGHT, WIDTH, 3)


def test_invalid():
    with pytest.raises(ValueError, match="`out_height` must be larger"):
        Mosaic4(out_height=0, out_width=WIDTH)

    with pytest.raises(ValueError, match="`out_width` must be larger"):
        Mosaic4(out_height=HEIGHT, out_width=0)
