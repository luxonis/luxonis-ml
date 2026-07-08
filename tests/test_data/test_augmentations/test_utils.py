import numpy as np

from luxonis_ml.data.augmentations.utils import (
    postprocess_bboxes,
    preprocess_bbox_corners,
    preprocess_bboxes,
)


def test_preprocess_postprocess_obb_bboxes() -> None:
    bboxes = np.array([[2.0, 0.1, 0.2, 0.3, 0.4, 15.0]])

    preprocessed = preprocess_bboxes(bboxes, bbox_counter=5)

    assert preprocessed.shape == (1, 7)
    assert np.allclose(
        preprocessed,
        np.array([[0.1, 0.2, 0.400001, 0.600001, 15.0, 2.0, 5.0]]),
    )

    postprocessed, ordering = postprocess_bboxes(preprocessed)

    assert np.allclose(
        postprocessed,
        np.array([[2.0, 0.1, 0.2, 0.300001, 0.400001, 15.0]]),
    )
    assert ordering.tolist() == [5]


def test_preprocess_postprocess_obb_corners() -> None:
    bboxes = np.array([[2.0, 0.1, 0.2, 0.3, 0.4, 30.0]])

    preprocessed = preprocess_bboxes(bboxes, bbox_counter=0)
    corners = preprocess_bbox_corners(bboxes, height=100, width=200)
    postprocessed, ordering = postprocess_bboxes(
        preprocessed,
        bbox_corners=corners,
        image_height=100,
        image_width=200,
    )

    assert np.allclose(postprocessed, bboxes)
    assert ordering.tolist() == [0]


def test_empty_obb_bboxes() -> None:
    preprocessed = preprocess_bboxes(np.empty((0, 6)), bbox_counter=0)
    postprocessed, ordering = postprocess_bboxes(np.empty((0, 7)))

    assert preprocessed.shape == (0, 7)
    assert postprocessed.shape == (0, 6)
    assert ordering.shape == (0,)
