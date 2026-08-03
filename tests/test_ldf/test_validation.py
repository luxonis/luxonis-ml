"""Regression tests for the annotation validators.

Each test pins a defect found while reviewing the extraction of the
annotation schemas into `luxonis_ml.ldf` and fails without the matching fix.
"""

from pathlib import Path

import numpy as np
import pydantic
import pytest

from luxonis_ml.ldf import (
    BBoxAnnotation,
    DatasetRecord,
    Detection,
    KeypointAnnotation,
    SegmentationAnnotation,
    load_annotation,
)


def test_bbox_missing_field_is_a_validation_error():
    """Indexing the raw input used to raise a bare ``KeyError: 'h'``."""
    with pytest.raises(pydantic.ValidationError) as info:
        BBoxAnnotation(x=0.1, y=0.1, w=0.1)  # type: ignore[call-arg]
    assert [(e["loc"], e["type"]) for e in info.value.errors()] == [
        (("h",), "missing")
    ]

    with pytest.raises(pydantic.ValidationError) as info:
        Detection(class_name="car", boundingbox={"x": 0.1, "y": 0.1, "w": 0.1})  # type: ignore[arg-type]
    assert [(e["loc"], e["type"]) for e in info.value.errors()] == [
        (("boundingbox", "h"), "missing")
    ]


def test_bbox_non_numeric_field_is_a_validation_error():
    """Comparing the raw input used to raise a bare ``TypeError``."""
    with pytest.raises(pydantic.ValidationError):
        BBoxAnnotation(x="a", y=0.1, w=0.1, h=0.1)  # type: ignore[arg-type]


def test_bbox_non_mapping_input_is_a_validation_error():
    with pytest.raises(pydantic.ValidationError):
        BBoxAnnotation.model_validate([0.1, 0.1, 0.1, 0.1])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # Out-of-range coordinates are clipped to [0, 1].
        ({"x": -0.5, "y": 0.1, "w": 0.2, "h": 0.2}, (0, 0.1, 0.2, 0.2)),
        # `x + w > 1` clips the width so the sum is 1.
        ({"x": 0.5, "y": 0.1, "w": 0.9, "h": 0.2}, (0.5, 0.1, 0.5, 0.2)),
    ],
)
def test_bbox_clipping_does_not_mutate_the_input(
    values: dict[str, float], expected: tuple[float, float, float, float]
):
    original = dict(values)
    box = BBoxAnnotation.model_validate(values)

    assert values == original
    assert (box.x, box.y, box.w, box.h) == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bbox_clipping_applies_to_numpy_scalars(dtype: type):
    """`np.float32` is not a `float` subclass, but must still be clipped."""
    box = BBoxAnnotation.model_validate(
        {"x": dtype(-0.5), "y": dtype(0.1), "w": dtype(0.2), "h": dtype(0.2)}
    )

    assert box.x == 0


def test_keypoint_clipping_does_not_mutate_the_input():
    keypoints = [(0.5, 0.5, 2), (1.5, 0.2, 1)]
    values = {"keypoints": list(keypoints)}

    annotation = KeypointAnnotation.model_validate(values)

    assert values == {"keypoints": keypoints}
    assert annotation.keypoints == [(0.5, 0.5, 2), (1.0, 0.2, 1)]


def test_load_annotation_does_not_mutate_the_input():
    """`load_annotation` validates the caller's mapping directly."""
    data = {"x": 0.5, "y": 0.5, "w": 0.9, "h": 0.2}
    original = dict(data)

    load_annotation("boundingbox", data)

    assert data == original


def test_mask_size_wins_over_supplied_height_and_width():
    """A stale `height`/`width` used to override the encoded mask size."""
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:5] = 1

    annotation = SegmentationAnnotation(
        mask=mask,  # type: ignore[call-arg]
        height=999,
        width=999,
    )

    assert (annotation.height, annotation.width) == (4, 6)
    np.testing.assert_array_equal(annotation.to_numpy(), mask)


def test_record_rejects_both_file_and_files(tempdir: Path):
    """`files` used to be silently replaced by the single `file`."""
    with pytest.raises(pydantic.ValidationError, match="not both"):
        DatasetRecord(
            file=tempdir / "image.png",  # type: ignore[call-arg]
            files={
                "left": tempdir / "left.png",
                "right": tempdir / "right.png",
            },
        )


def test_scale_to_boxes_error_names_the_actual_field():
    with pytest.raises(pydantic.ValidationError, match=r"`scale_to_boxes`"):
        Detection(
            class_name="car",
            scale_to_boxes=True,
            keypoints={"keypoints": [(0.5, 0.5, 2)]},  # type: ignore[arg-type]
        )
