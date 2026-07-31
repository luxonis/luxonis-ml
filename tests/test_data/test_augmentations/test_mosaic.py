from typing import Final

import numpy as np
import pytest
from albumentations.core.bbox_utils import normalize_bboxes
from pytest_subtests.plugin import SubTests

from luxonis_ml.data.augmentations.custom.mosaic import Mosaic4

WIDTH: Final[int] = 640
HEIGHT: Final[int] = 480


def test_mosaic4_helpers(subtests: SubTests):
    with subtests.test("image"):
        img = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
        mosaic = Mosaic4._apply_mosaic4_to_images(
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
            mosaic_bbox = Mosaic4._apply_mosaic4_to_bboxes(
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
            mosaic_keypoint = Mosaic4._apply_mosaic4_to_keypoints(
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


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        # Bare `np.array([])` entries have no width of their own and take
        # the populated sample's.
        (
            [
                np.array([]),
                np.zeros((2, 7), np.float32),
                np.array([]),
                np.array([]),
            ],
            (2, 7),
        ),
        # With nothing populated the width still has to come from the empty
        # rows, not from the fallback.
        ([np.zeros((0, 7), np.float32)] * 4, (0, 7)),
    ],
)
def test_mosaic4_pads_empty_samples_to_the_batch_width(
    batch: list[np.ndarray],
    expected: tuple[int, int],
    subtests: SubTests,
) -> None:
    """Samples with no boxes must not narrow the merged array."""
    mosaic = Mosaic4(out_height=HEIGHT, out_width=WIDTH, p=1.0)
    shapes = [(HEIGHT // 2, WIDTH // 2)] * 4

    with subtests.test("bboxes"):
        merged = mosaic.apply_to_bboxes(batch, shapes, x_crop=0, y_crop=0)
        assert merged.shape == expected

    with subtests.test("keypoints"):
        merged = mosaic.apply_to_keypoints(batch, shapes, x_crop=0, y_crop=0)
        assert merged.shape == expected


def test_invalid():
    with pytest.raises(ValueError, match="`height` must be larger"):
        Mosaic4(out_height=0, out_width=WIDTH)

    with pytest.raises(ValueError, match="`width` must be larger"):
        Mosaic4(out_height=HEIGHT, out_width=0)
