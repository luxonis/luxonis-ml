"""Tests for the LDF <-> Albumentations boundary conversions.

These functions are where label layouts change shape, so they are covered
both by explicit shape cases and by round-trip properties: converting a label
out and straight back must return what went in.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from luxonis_ml.data.augmentations.utils import (
    postprocess_bboxes,
    postprocess_keypoints,
    postprocess_mask,
    preprocess_bboxes,
    preprocess_keypoints,
    preprocess_mask,
    yield_batches,
)

round_trip = settings(max_examples=100, deadline=None, derandomize=True)

# Coordinates kept clear of the image border so that no keypoint is clipped
# or marked invisible on the way back.
normalized = st.floats(
    min_value=0.05, max_value=0.85, allow_nan=False, allow_infinity=False
)


def test_preprocess_mask_moves_channels_last() -> None:
    mask = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert preprocess_mask(mask).shape == (3, 4, 2)


def test_postprocess_mask_promotes_a_flat_mask() -> None:
    """A transform may hand back a 2D mask; it becomes a single channel."""
    assert postprocess_mask(np.ones((5, 6))).shape == (1, 5, 6)


def test_postprocess_mask_moves_channels_first() -> None:
    assert postprocess_mask(np.ones((5, 6, 2))).shape == (2, 5, 6)


@given(
    n_channels=st.integers(min_value=1, max_value=4),
    height=st.integers(min_value=1, max_value=8),
    width=st.integers(min_value=1, max_value=8),
)
@round_trip
def test_mask_round_trip(n_channels: int, height: int, width: int) -> None:
    mask = np.arange(n_channels * height * width, dtype=np.uint8).reshape(
        n_channels, height, width
    )

    assert np.array_equal(postprocess_mask(preprocess_mask(mask)), mask)


def test_postprocess_bboxes_handles_no_boxes() -> None:
    """An empty bbox array still has to yield usable output shapes."""
    boxes, ordering = postprocess_bboxes(np.zeros((0, 6)))

    assert boxes.shape == (0, 5)
    assert ordering.shape == (0,)


def test_postprocess_bboxes_drops_boxes_below_the_area_threshold() -> None:
    boxes = np.array(
        [[0.0, 0.0, 0.5, 0.5, 1.0, 0.0], [0.0, 0.0, 0.01, 0.01, 2.0, 1.0]]
    )

    kept, ordering = postprocess_bboxes(boxes, area_threshold=0.1)

    assert kept.shape == (1, 5)
    assert ordering.tolist() == [0]


def test_preprocess_bboxes_appends_an_index_column() -> None:
    boxes = np.array([[1.0, 0.1, 0.2, 0.3, 0.4], [2.0, 0.5, 0.5, 0.2, 0.2]])

    out = preprocess_bboxes(boxes)

    assert out.shape == (2, 6)
    assert out[:, -1].tolist() == [0.0, 1.0]
    assert out[:, 4].tolist() == [1.0, 2.0]


@given(
    x=normalized,
    y=normalized,
    w=st.floats(min_value=0.05, max_value=0.1),
    h=st.floats(min_value=0.05, max_value=0.1),
    class_id=st.integers(min_value=0, max_value=5),
)
@round_trip
def test_bbox_round_trip(
    x: float, y: float, w: float, h: float, class_id: int
) -> None:
    boxes = np.array([[float(class_id), x, y, w, h]])

    out, ordering = postprocess_bboxes(
        preprocess_bboxes(boxes.copy()), area_threshold=0.0
    )

    assert ordering.tolist() == [0]
    # preprocess nudges the far edge by 1e-6 to avoid zero-area boxes.
    np.testing.assert_allclose(out, boxes, atol=1e-5)


def test_preprocess_keypoints_scales_to_pixels() -> None:
    keypoints = np.array([[0.5, 0.25, 2.0, 1.0, 0.0, 1.0]])

    out = preprocess_keypoints(keypoints, height=8, width=4)

    assert out.tolist() == [[2.0, 2.0, 2.0], [4.0, 0.0, 1.0]]


def test_postprocess_keypoints_marks_out_of_bounds_invisible() -> None:
    keypoints = np.array([[5.0, 5.0, 2.0], [12.0, -1.0, 2.0]])

    out = postprocess_keypoints(keypoints, np.array([0]), 10, 10, 2)

    assert out[0, 2] == 2.0
    assert out[0, 5] == 0.0, "the out-of-bounds keypoint must lose visibility"


def test_postprocess_keypoints_reorders_by_surviving_bboxes() -> None:
    keypoints = np.array([[1.0, 1.0, 2.0], [9.0, 9.0, 1.0]])

    out = postprocess_keypoints(keypoints, np.array([1, 0]), 10, 10, 1)

    np.testing.assert_allclose(out, [[0.9, 0.9, 1.0], [0.1, 0.1, 2.0]])


@given(
    n_instances=st.integers(min_value=1, max_value=3),
    n_keypoints=st.integers(min_value=1, max_value=3),
    size=st.integers(min_value=16, max_value=64),
    coords=st.data(),
)
@round_trip
def test_keypoint_round_trip(
    n_instances: int, n_keypoints: int, size: int, coords: st.DataObject
) -> None:
    keypoints = np.array(
        [
            [
                value
                for _ in range(n_keypoints)
                for value in (
                    coords.draw(normalized),
                    coords.draw(normalized),
                    2.0,
                )
            ]
            for _ in range(n_instances)
        ]
    )

    pixels = preprocess_keypoints(keypoints.copy(), height=size, width=size)
    out = postprocess_keypoints(
        pixels, np.arange(n_instances), size, size, n_keypoints
    )

    np.testing.assert_allclose(out, keypoints, atol=1e-6)


@given(
    keypoints=npst.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(0, 6), st.just(3)),
        elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    )
)
@round_trip
def test_postprocess_keypoints_never_leaves_the_image(
    keypoints: np.ndarray,
) -> None:
    """Coordinates are always clipped into the normalized unit square."""
    out = postprocess_keypoints(
        keypoints, np.arange(keypoints.shape[0]), 10, 10, 1
    )

    assert np.all(out[:, 0::3] >= 0)
    assert np.all(out[:, 0::3] <= 1)
    assert np.all(out[:, 1::3] >= 0)
    assert np.all(out[:, 1::3] <= 1)


def test_yield_batches_groups_by_key() -> None:
    samples = [{"a": 1}, {"a": 2}, {"a": 3}]

    assert list(yield_batches(samples, 2)) == [{"a": [1, 2]}, {"a": [3]}]


@given(
    n_samples=st.integers(min_value=1, max_value=10),
    batch_size=st.integers(min_value=1, max_value=5),
)
@round_trip
def test_yield_batches_covers_every_sample_once(
    n_samples: int, batch_size: int
) -> None:
    samples = [{"a": i} for i in range(n_samples)]

    batches = list(yield_batches(samples, batch_size))

    assert [value for batch in batches for value in batch["a"]] == list(
        range(n_samples)
    )
    assert all(len(batch["a"]) <= batch_size for batch in batches)


@pytest.mark.parametrize("n_keypoints", [2, 3])
def test_postprocess_keypoints_keeps_annotation_grouping(
    n_keypoints: int,
) -> None:
    """Rows are regrouped so each output row is one annotation."""
    n_instances = 3
    keypoints = np.zeros((n_instances * n_keypoints, 3))

    out = postprocess_keypoints(
        keypoints, np.arange(n_instances), 10, 10, n_keypoints
    )

    assert out.shape == (n_instances, n_keypoints * 3)
