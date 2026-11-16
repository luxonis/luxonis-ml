import numpy as np
import pytest
from pytest_subtests.plugin import SubTests

from luxonis_ml.data.augmentations.custom.mixup import MixUp


@pytest.mark.parametrize("augmentation_data", [2], indirect=True)
@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_mixup(
    height: int,
    width: int,
    augmentation_data: dict[str, list[np.ndarray]],
    subtests: SubTests,
    alpha: float,
):
    mixup = MixUp(p=1.0, alpha=alpha)
    transformed = mixup(**augmentation_data)  # type: ignore
    assert set(transformed.keys()) == set(augmentation_data.keys())
    for value in transformed.values():
        assert isinstance(value, np.ndarray)

    with subtests.test("image"):
        assert transformed["image"].shape == (height, width, 3)
        assert np.allclose(
            (
                augmentation_data["image"][0] * alpha
                + augmentation_data["image"][1] * (1 - alpha)
            ),
            transformed["image"],
        )

    with subtests.test("bboxes"):
        assert transformed["bboxes"].shape == (2, 6)
        assert np.allclose(
            np.concatenate(augmentation_data["bboxes"]), transformed["bboxes"]
        )

    with subtests.test("keypoints"):
        assert transformed["keypoints"].shape == (2, 5)
        assert np.allclose(
            np.concatenate(augmentation_data["keypoints"]),
            transformed["keypoints"],
        )

    with subtests.test("mask"):
        assert transformed["mask"].shape == (height, width, 1)
        assert np.allclose(
            augmentation_data["mask"][0].astype(bool)
            | augmentation_data["mask"][1].astype(bool),
            transformed["mask"].astype(bool),
        )


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        # A bare `np.array([])` has no width of its own and takes the
        # populated sample's.
        ([np.array([]), np.zeros((2, 7), np.float32)], (2, 7)),
        # With nothing populated the width still has to come from the empty
        # rows, not from the fallback.
        (
            [np.zeros((0, 7), np.float32), np.zeros((0, 7), np.float32)],
            (0, 7),
        ),
    ],
)
def test_mixup_pads_empty_samples_to_the_batch_width(
    batch: list[np.ndarray],
    expected: tuple[int, int],
    subtests: SubTests,
) -> None:
    """A sample with no boxes must not narrow the merged array."""
    mixup = MixUp(p=1.0)
    shapes = [(16, 16), (16, 16)]

    with subtests.test("bboxes"):
        assert mixup.apply_to_bboxes(list(batch), shapes).shape == expected

    with subtests.test("keypoints"):
        assert mixup.apply_to_keypoints(list(batch), shapes).shape == expected


@pytest.mark.parametrize("alpha", [0.7, 0.5, 0.3])
def test_mixup_gives_each_pixel_the_classes_of_one_mask(alpha: float):
    """Keep the dominant pixel's classes, filling only its empty pixels."""
    cat, dog, both, empty = [1, 0], [0, 1], [1, 1], [0, 0]
    dominant = np.array(
        [[cat, cat, empty, empty, both, empty]], dtype=np.uint8
    )
    other = np.array([[dog, empty, dog, empty, dog, both]], dtype=np.uint8)
    masks = [dominant, other] if alpha >= 0.5 else [other, dominant]

    output = MixUp(p=1.0).apply_to_mask(
        masks, image_shapes=[(1, 6), (1, 6)], alpha=alpha
    )

    assert output.tolist() == [[cat, cat, dog, empty, both, both]]


@pytest.mark.parametrize(("alpha", "overlap_class"), [(0.7, 1), (0.3, 3)])
def test_mixup_accepts_two_dimensional_masks(
    alpha: float, overlap_class: int
) -> None:
    first = np.array([[1, 0, 0], [0, 2, 0]], dtype=np.uint8)
    second = np.array([[3, 0, 4], [0, 0, 0]], dtype=np.uint8)

    output = MixUp(p=1.0).apply_to_mask(
        [first, second], [(2, 3), (2, 3)], alpha=alpha
    )

    expected = np.array([[overlap_class, 0, 4], [0, 2, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("keep_aspect_ratio", [True, False])
def test_mixup_preserves_single_channel_masks_when_resizing(
    *,
    keep_aspect_ratio: bool,
) -> None:
    first = np.zeros((4, 4, 1), dtype=np.uint8)
    first[0, 0, 0] = 1
    second = np.ones((1, 2, 1), dtype=np.uint8)

    output = MixUp(p=1.0, keep_aspect_ratio=keep_aspect_ratio).apply_to_mask(
        [first, second], [(4, 4), (1, 2)], alpha=0.3
    )

    expected = np.zeros((4, 4, 1), dtype=np.uint8)
    if keep_aspect_ratio:
        expected[1:3] = 1
    else:
        expected[:] = 1
    expected[0, 0, 0] = 1
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("empty_first", [True, False])
def test_mixup_keeps_the_populated_semantic_mask(*, empty_first: bool) -> None:
    empty = np.array([])
    populated = np.ones((2, 2, 1), dtype=np.uint8)
    masks = [empty, populated] if empty_first else [populated, empty]

    output = MixUp(p=1.0).apply_to_mask(masks, [(2, 2), (2, 2)], alpha=0.7)

    np.testing.assert_array_equal(output, populated)


def test_mixup_leaves_the_callers_batch_alone() -> None:
    mixup = MixUp(p=1.0)
    batch = [np.array([]), np.zeros((2, 7), dtype=np.float32)]

    mixup.apply_to_bboxes(list(batch), [(16, 16), (16, 16)])

    assert batch[0].shape == (0,)


def test_invalid():
    with pytest.raises(ValueError, match="must be in range"):
        MixUp(alpha=(-1, 1))
    with pytest.raises(ValueError, match="must be in range"):
        MixUp(alpha=(1, -1))
    with pytest.raises(ValueError, match="ascending order"):
        MixUp(alpha=(0.8, 0.2))
