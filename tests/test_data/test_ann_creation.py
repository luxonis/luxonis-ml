import numpy as np
import pydantic
import pytest

from luxonis_ml.data.datasets.annotation import (
    BBoxAnnotation,
    KeypointAnnotation,
    PolylineSegmentationAnnotation,
)


def test_bbox_no_auto_clip():
    base_dict = {"x": 0, "y": 0, "w": 0, "h": 0}
    for k in ["x", "y", "w", "h"]:
        for v in [-2.1, 2.3, -3.3, 3]:
            with pytest.raises(pydantic.ValidationError):
                curr_dict = base_dict.copy()
                curr_dict[k] = v
                BBoxAnnotation(**curr_dict)


def test_bbox_auto_clip():
    base_dict = {"x": 0, "y": 0, "w": 0, "h": 0}
    for k in ["x", "y", "w", "h"]:
        for v in [-1.1, 1.3, -1.3, 2]:
            curr_dict = base_dict.copy()
            curr_dict[k] = v
            bbox_ann = BBoxAnnotation(**curr_dict)
            assert 0 <= bbox_ann.x <= 1
            assert 0 <= bbox_ann.y <= 1
            assert 0 <= bbox_ann.w <= 1
            assert 0 <= bbox_ann.h <= 1


def test_bbox_clip_sum():
    bbox_ann = BBoxAnnotation(**{"x": 0.9, "y": 0, "w": 0.2, "h": 0})
    assert bbox_ann.x + bbox_ann.w <= 1
    bbox_ann = BBoxAnnotation(**{"x": 1.2, "y": 0, "w": 0.2, "h": 0})
    assert bbox_ann.x + bbox_ann.w <= 1
    bbox_ann = BBoxAnnotation(**{"x": 0, "y": 0.9, "w": 0, "h": 0.2})
    assert bbox_ann.y + bbox_ann.h <= 1
    bbox_ann = BBoxAnnotation(**{"x": 0, "y": 1.2, "w": 0, "h": 0.2})
    assert bbox_ann.y + bbox_ann.h <= 1


def test_kpt_no_auto_clip():
    with pytest.raises(pydantic.ValidationError):
        KeypointAnnotation(**{"keypoints": [(-2.1, 1.1, 0)]})
    with pytest.raises(pydantic.ValidationError):
        KeypointAnnotation(**{"keypoints": [(0.1, 2.1, 1)]})
    with pytest.raises(pydantic.ValidationError):
        KeypointAnnotation(**{"keypoints": [(0.1, 1.1, 2), (0.1, 2.1, 1)]})


def test_kpt_auto_clip():
    kpt_ann = KeypointAnnotation(**{"keypoints": [(-1.1, 1.1, 0)]})
    assert (
        0 <= kpt_ann.keypoints[0][0] <= 1 and 0 <= kpt_ann.keypoints[0][1] <= 1
    )
    kpt_ann = KeypointAnnotation(**{"keypoints": [(0.1, 1.1, 1)]})
    assert (
        0 <= kpt_ann.keypoints[0][0] <= 1 and 0 <= kpt_ann.keypoints[0][1] <= 1
    )
    kpt_ann = KeypointAnnotation(**{"keypoints": [(-2, 2, 2)]})
    assert (
        0 <= kpt_ann.keypoints[0][0] <= 1 and 0 <= kpt_ann.keypoints[0][1] <= 1
    )


def test_poly_no_auto_clip():
    with pytest.raises(pydantic.ValidationError):
        PolylineSegmentationAnnotation(
            **{"points": [(-2.1, 1.1), (-2.1, 2.1), (-0.1, -2.1)]}
        )


def test_combine_to_numpy_handle_overlaps():
    class_mapping = {"class1": 0, "class2": 1, "class3": 2}
    height, width = 4, 4

    polylines = [
        np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)
        / [width, height],
        np.array([(1, 0), (2, 0), (2, 1), (1, 1)], dtype=float)
        / [width, height],
        np.array([(2, 2), (3, 2), (3, 3), (2, 3)], dtype=float)
        / [width, height],
    ]

    annotations = [
        PolylineSegmentationAnnotation(
            **{"points": polylines[0], "class": "class1"}
        ),
        PolylineSegmentationAnnotation(
            **{"points": polylines[1], "class": "class2"}
        ),
        PolylineSegmentationAnnotation(
            **{"points": polylines[2], "class": "class3"}
        ),
    ]

    combined = PolylineSegmentationAnnotation.combine_to_numpy(
        annotations, class_mapping, height=height, width=width
    )

    expected_combined = np.zeros(
        (len(class_mapping), height, width), dtype=np.uint8
    )
    expected_combined[0, :, :] = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    expected_combined[1, :, :] = np.array(
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    expected_combined[2, :, :] = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
        dtype=np.uint8,
    )

    assert combined.shape == (len(class_mapping), height, width)
    assert np.array_equal(
        combined, expected_combined
    ), f"Expected {expected_combined}, but got {combined}"
