import json
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import polars as pl
import pydantic
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from pytest_subtests import SubTests

from luxonis_ml.data.datasets.annotation import (
    ArrayAnnotation,
    BBoxAnnotation,
    ClassificationAnnotation,
    DatasetRecord,
    Detection,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    KeypointVisibility,
    NormalizedFloat,
    SegmentationAnnotation,
    load_annotation,
)
from luxonis_ml.data.utils.parquet import (
    DEFAULT_METADATA,
    ParquetFileManager,
    ParquetRecord,
)

Keypoint: TypeAlias = tuple[
    NormalizedFloat, NormalizedFloat, KeypointVisibility
]

# Coordinates are normalized to [0, 1]; anything within [-2, 2] is
# clipped into that range and anything beyond it is rejected.
normalized = st.floats(0, 1)
clippable = st.floats(-2, 2)
out_of_clipping_range = st.floats(2, 1e6, exclude_min=True) | st.floats(
    -1e6, -2, exclude_max=True
)
binary_masks = npst.arrays(
    dtype=np.uint8,
    shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=8),
    elements=st.integers(0, 1),
)


@st.composite
def masks_with_classes(
    draw: st.DrawFn,
) -> tuple[np.ndarray, list[int], int]:
    """Draw a stack of equally sized binary masks with a class ID each."""
    shape = draw(
        npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=6)
    )
    n_annotations = draw(st.integers(1, 4))
    masks = draw(
        npst.arrays(
            np.uint8, (n_annotations, *shape), elements=st.integers(0, 1)
        )
    )

    n_classes = draw(st.integers(1, 4))
    classes = draw(
        st.lists(
            st.integers(0, n_classes - 1),
            min_size=n_annotations,
            max_size=n_annotations,
        )
    )
    return masks, classes, n_classes


def test_valid_identifier():
    Detection._check_valid_identifier("variable", label="")
    Detection._check_valid_identifier("variable_name", label="")
    Detection._check_valid_identifier("variable-name", label="")

    with pytest.raises(ValueError, match="can only contain alphanumeric"):
        Detection._check_valid_identifier("variable name", label="")

    with pytest.raises(ValueError, match="can only contain alphanumeric"):
        Detection._check_valid_identifier("?variable_name", label="")

    with pytest.raises(ValueError, match="can only contain alphanumeric"):
        Detection._check_valid_identifier("12variable_name", label="")

    with pytest.raises(ValueError, match="can only contain alphanumeric"):
        Detection._check_valid_identifier("variable/name", label="")


def test_load_annotation():
    assert load_annotation(
        "boundingbox", {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    ) == BBoxAnnotation(x=0.1, y=0.2, w=0.3, h=0.4)
    with pytest.raises(ValueError, match="Unknown label type"):
        load_annotation("invalid_name", {})  # type: ignore


def test_dataset_record(tempdir: Path):
    def compare_parquet_rows(
        record: DatasetRecord, expected_rows: list[ParquetRecord]
    ) -> None:
        rows = list(record.to_parquet_rows())
        for row in rows:
            # for compatibility with Windows
            row["file"] = Path(row["file"])  # type: ignore
        assert rows == expected_rows

    left = (tempdir / "left.jpg").resolve()
    right = (tempdir / "right.jpg").resolve()

    cv2.imwrite(str(left), np.zeros((100, 100, 3)))
    cv2.imwrite(str(right), np.zeros((100, 100, 3)))
    empty_metadata = {"sample_metadata": DEFAULT_METADATA}
    record = DatasetRecord(file=left)  # type: ignore
    assert record.file == left

    compare_parquet_rows(
        record,
        [
            {
                "file": left,  # type: ignore
                "source_name": "image",
                "task_name": "",
                "class_name": None,
                "instance_id": None,
                "task_type": None,
                "annotation": None,
                **empty_metadata,
            }
        ],
    )

    record = DatasetRecord(
        file=left,  # type: ignore
        sample_metadata={"source": "camera-a"},
    )
    compare_parquet_rows(
        record,
        [
            {
                "file": left,  # type: ignore
                "source_name": "image",
                "task_name": "",
                "class_name": None,
                "instance_id": None,
                "task_type": None,
                "annotation": None,
                "sample_metadata": json.dumps({"source": "camera-a"}),
            }
        ],
    )

    record = DatasetRecord(
        file=left,  # type: ignore
        annotation={
            "class": "person",
            "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        },
    )
    compare_parquet_rows(
        record,
        [
            {
                "file": left,  # type: ignore
                "source_name": "image",
                "task_name": "",
                "class_name": "person",
                "instance_id": -1,
                "task_type": "boundingbox",
                "annotation": '{"x":0.1,"y":0.2,"w":0.3,"h":0.4}',
                **empty_metadata,
            },
            {
                "file": left,  # type: ignore
                "source_name": "image",
                "task_name": "",
                "class_name": "person",
                "instance_id": -1,
                "task_type": "classification",
                "annotation": "{}",
                **empty_metadata,
            },
        ],
    )

    record = DatasetRecord(
        files={
            "left": left,
            "right": right,
        }
    )
    with pytest.raises(ValueError, match="must have exactly one file"):
        _ = record.file


def test_bbox_annotation(subtests: SubTests):
    with subtests.test("simple"):
        bbox = BBoxAnnotation(x=0.1, y=0.2, w=0.3, h=0.4)
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.w == 0.3
        assert bbox.h == 0.4

    with subtests.test("numpy"):
        bbox = BBoxAnnotation(x=0.1, y=0.2, w=0.3, h=0.4)
        assert np.allclose(
            bbox.to_numpy(class_id=4), np.array([4, 0.1, 0.2, 0.3, 0.4])
        )
        bboxes = [
            bbox,
            BBoxAnnotation(x=0.2, y=0.3, w=0.4, h=0.5),
            BBoxAnnotation(x=0.3, y=0.4, w=0.5, h=0.6),
        ]
        assert np.allclose(
            BBoxAnnotation.combine_to_numpy(bboxes, [1, 2, 3]),
            np.array(
                [
                    [1, 0.1, 0.2, 0.3, 0.4],
                    [2, 0.2, 0.3, 0.4, 0.5],
                    [3, 0.3, 0.4, 0.5, 0.6],
                ]
            ),
        )


@given(x=clippable, y=clippable, w=clippable, h=clippable)
def test_bbox_is_clipped_into_the_unit_square(
    x: float, y: float, w: float, h: float
):
    bbox = BBoxAnnotation(x=x, y=y, w=w, h=h)

    assert 0 <= bbox.x <= 1
    assert 0 <= bbox.y <= 1
    assert 0 <= bbox.w <= 1
    assert 0 <= bbox.h <= 1
    assert bbox.x + bbox.w <= 1
    assert bbox.y + bbox.h <= 1


@given(x=normalized, y=normalized, w=normalized, h=normalized)
def test_bbox_already_inside_the_unit_square_is_kept_as_is(
    x: float, y: float, w: float, h: float
):
    assume(x + w <= 1)
    assume(y + h <= 1)

    bbox = BBoxAnnotation(x=x, y=y, w=w, h=h)

    assert (bbox.x, bbox.y, bbox.w, bbox.h) == (x, y, w, h)


@given(x=clippable, y=clippable, w=clippable, h=clippable)
def test_bbox_clipping_is_idempotent(x: float, y: float, w: float, h: float):
    clipped = BBoxAnnotation(x=x, y=y, w=w, h=h)

    assert BBoxAnnotation(**clipped.model_dump()) == clipped


@given(
    value=out_of_clipping_range, field=st.sampled_from(["x", "y", "w", "h"])
)
def test_bbox_rejects_values_outside_the_clipping_range(
    value: float, field: str
):
    values = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, field: value}

    with pytest.raises(ValueError, match="outside of automatic clipping"):
        BBoxAnnotation(**values)  # type: ignore


@given(
    keypoints=st.lists(
        st.tuples(clippable, clippable, st.integers(0, 2)),
        min_size=1,
        max_size=5,
    )
)
def test_keypoints_are_clipped_into_the_unit_square(
    keypoints: list[Keypoint],
):
    annotation = KeypointAnnotation(keypoints=keypoints)

    assert len(annotation.keypoints) == len(keypoints)
    for (x, y, visibility), (*_, expected_visibility) in zip(
        annotation.keypoints, keypoints, strict=True
    ):
        assert 0 <= x <= 1
        assert 0 <= y <= 1
        assert visibility == expected_visibility

    assert annotation.to_numpy().shape == (3 * len(keypoints),)


@given(
    keypoints=st.lists(
        st.tuples(clippable, clippable, st.integers(0, 2)),
        min_size=1,
        max_size=5,
    )
)
def test_clipping_keypoints_does_not_touch_the_caller(
    keypoints: list[Keypoint],
):
    original = list(keypoints)

    KeypointAnnotation(keypoints=keypoints)

    assert keypoints == original


@given(
    keypoints=st.lists(
        st.tuples(normalized, normalized, st.integers(0, 2)),
        min_size=1,
        max_size=5,
    )
)
def test_keypoints_already_inside_the_unit_square_are_kept_as_is(
    keypoints: list[Keypoint],
):
    annotation = KeypointAnnotation(keypoints=keypoints)

    assert annotation.keypoints == keypoints


@given(
    value=out_of_clipping_range,
    axis=st.sampled_from(["x", "y"]),
    n_valid=st.integers(0, 3),
)
def test_keypoints_reject_values_outside_the_clipping_range(
    value: float, axis: str, n_valid: int
):
    """A single out-of-range keypoint invalidates the whole
    annotation.
    """
    valid: Keypoint = (0.5, 0.5, 2)
    invalid: Keypoint = (value, 0.5, 2) if axis == "x" else (0.5, value, 2)
    keypoints: list[Keypoint] = [*n_valid * [valid], invalid]

    with pytest.raises(pydantic.ValidationError):
        KeypointAnnotation(keypoints=keypoints)


@given(visibility=st.integers().filter(lambda value: value not in {0, 1, 2}))
def test_keypoints_reject_unknown_visibility(visibility: int):
    with pytest.raises(pydantic.ValidationError):
        KeypointAnnotation(keypoints=[(0.5, 0.5, visibility)])  # type: ignore


def test_keypoints_annotation(subtests: SubTests):
    with subtests.test("numpy"):
        keypoints = KeypointAnnotation(keypoints=[(0.1, 0.2, 2)])
        assert np.allclose(keypoints.to_numpy(), np.array([0.1, 0.2, 2]))
        keypoints_list = [
            keypoints,
            KeypointAnnotation(keypoints=[(0.2, 0.3, 0)]),
            KeypointAnnotation(keypoints=[(0.3, 0.4, 1)]),
        ]
        assert np.allclose(
            KeypointAnnotation.combine_to_numpy(keypoints_list),
            np.array([[0.1, 0.2, 2], [0.2, 0.3, 0], [0.3, 0.4, 1]]),
        )


@given(mask=binary_masks)
def test_segmentation_mask_survives_the_rle_roundtrip(mask: np.ndarray):
    annotation = SegmentationAnnotation(mask=mask)  # type: ignore

    assert (annotation.height, annotation.width) == mask.shape
    assert np.array_equal(annotation.to_numpy(), mask)


@given(mask=binary_masks)
def test_segmentation_accepts_the_rle_it_produced(mask: np.ndarray):
    from_mask = SegmentationAnnotation(mask=mask)  # type: ignore
    from_counts = SegmentationAnnotation(
        counts=from_mask.counts,
        height=from_mask.height,
        width=from_mask.width,
    )

    assert from_counts == from_mask
    assert np.array_equal(from_counts.to_numpy(), mask)


@given(masks_and_classes=masks_with_classes())
def test_semantic_segmentation_assigns_each_pixel_to_one_class(
    masks_and_classes: tuple[np.ndarray, list[int], int],
):
    masks, classes, n_classes = masks_and_classes
    annotations = [
        SegmentationAnnotation(mask=mask)  # type: ignore
        for mask in masks
    ]

    combined = SegmentationAnnotation.combine_to_numpy(
        annotations, classes, n_classes
    )

    assert combined.shape == (n_classes, *masks.shape[1:])
    assert combined.max(initial=0) <= 1
    # Classes never overlap, and no pixel is lost or invented.
    assert combined.sum(axis=0).max(initial=0) <= 1
    assert np.array_equal(combined.any(axis=0), masks.any(axis=0))
    # The first annotation takes precedence over all the others.
    assert np.all(combined[classes[0]] >= masks[0])


@given(masks_and_classes=masks_with_classes())
def test_instance_segmentation_keeps_overlapping_instances(
    masks_and_classes: tuple[np.ndarray, list[int], int],
):
    masks, *_ = masks_and_classes
    annotations = [
        InstanceSegmentationAnnotation(mask=mask)  # type: ignore
        for mask in masks
    ]

    combined = InstanceSegmentationAnnotation.combine_to_numpy(annotations)

    assert np.array_equal(combined, masks)


def test_segmentation_annotation(subtests: SubTests, tempdir: Path):
    mask = np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )
    with subtests.test("mask"):
        seg = SegmentationAnnotation(mask=mask)  # type: ignore
        assert seg.height == 4
        assert seg.width == 4
        assert seg.counts == b"11213ON0"
        assert seg.model_dump() == {
            "height": 4,
            "width": 4,
            "counts": b"11213ON0",
        }
        np.save(tempdir / "mask.npy", mask)
        seg = SegmentationAnnotation(mask=tempdir / "mask.npy")  # type: ignore
        assert seg.height == 4
        assert seg.width == 4
        assert seg.counts == b"11213ON0"
        cv2.imwrite(str(tempdir / "mask.png"), mask)
        seg = SegmentationAnnotation(mask=tempdir / "mask.png")  # type: ignore
        assert seg.height == 4
        assert seg.width == 4
        assert seg.counts == b"11213ON0"

    with subtests.test("polyline"):
        seg = SegmentationAnnotation(
            points=[(0, 0), (1, 0), (1, 1), (0, 1)],  # type: ignore
            height=4,
            width=4,
        )
        assert seg.height == 4
        assert seg.width == 4
        assert seg.counts == b"0`0"
        seg_clipped = SegmentationAnnotation(
            points=[(-0.1, 0), (1.1, 0), (1, 1.5), (-0.6, 1)],  # type: ignore
            height=4,
            width=4,
        )
        assert seg == seg_clipped

        with pytest.raises(ValueError, match="must be integers"):
            SegmentationAnnotation(
                points=[(0, 0), (1, 0), (1, 1)],  # type: ignore
                height=4,
                width="4",
            )
        with pytest.raises(ValueError, match="2D points"):
            SegmentationAnnotation(
                points=[(0, 0, 0), (1, 0, 4)],  # type: ignore
                height=4,
                width=4,
            )

    with subtests.test("rle_bytes"):
        seg = SegmentationAnnotation(counts=b"11213ON0", height=4, width=4)
        assert seg.height == 4
        assert seg.width == 4
        assert seg.counts == b"11213ON0"
        assert np.array_equal(seg.to_numpy(), mask)

    with subtests.test("rle_ints"):
        seg = SegmentationAnnotation(
            # counts are computed using FORTRAN order
            counts=[1, 1, 2, 2, 5, 1, 3, 1],  # type: ignore
            height=4,
            width=4,
        )
        assert seg.height == 4
        assert seg.width == 4
        assert seg.counts == b"11213ON0"
        assert np.array_equal(seg.to_numpy(), mask)

        with pytest.raises(ValueError, match="must be integers"):
            SegmentationAnnotation(
                counts=[1, 1, 2, 2, 5, 1, 3],  # type: ignore
                height=4,
                width="4",  # type: ignore
            )

    with subtests.test("numpy_simple"):
        masks = np.array(
            [
                [
                    [0, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                ],
                [
                    [1, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ]
        )

        annotations = [
            SegmentationAnnotation(mask=masks[i])  # type: ignore
            for i in range(len(masks))
        ]

        combined = SegmentationAnnotation.combine_to_numpy(
            annotations, [0, 1], 2
        )
        assert np.array_equal(combined, masks)

    with subtests.test("numpy_overlap"):
        masks = np.array(
            [
                [
                    [0, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                ],
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ]
        )

        annotations = [
            SegmentationAnnotation(mask=masks[i])  # type: ignore
            for i in range(len(masks))
        ]

        combined = SegmentationAnnotation.combine_to_numpy(
            annotations, [0, 1], 2
        )
        assert np.array_equal(
            combined,
            np.array(
                [
                    [
                        [0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 1],
                    ],
                    [
                        [1, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ]
            ),
        )

    with subtests.test("numpy_instance_segmentation"):
        masks = np.array(
            [
                [
                    [0, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                ],
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ]
        )

        annotations = [
            InstanceSegmentationAnnotation(mask=masks[i])  # type: ignore
            for i in range(len(masks))
        ]

        combined = InstanceSegmentationAnnotation.combine_to_numpy(
            annotations, [0, 1], 2
        )
        assert np.array_equal(combined, masks)

    with subtests.test("invalid"):
        with pytest.raises(ValueError, match="Extra inputs"):
            SegmentationAnnotation(
                mask=mask,  # type: ignore
                points=[(1, 0), (2, 1)],  # type: ignore
            )

        with pytest.raises(ValueError, match="Unsupported mask format"):
            SegmentationAnnotation(mask="file.jpeg")  # type: ignore

        with pytest.raises(ValueError, match="Failed to load mask from image"):
            SegmentationAnnotation(mask="file.png")  # type: ignore

        with pytest.raises(TypeError, match="Mask must be either"):
            SegmentationAnnotation(mask=[1, 2, 3])  # type: ignore

        np.save(tempdir / "mask.npy", mask[None, None, ...])
        with pytest.raises(ValueError, match="Mask must be a 2D binary array"):
            SegmentationAnnotation(mask=tempdir / "mask.npy")  # type: ignore

        with pytest.raises(ValueError, match="at least 3 points"):
            SegmentationAnnotation(
                points=[(1, 0), (0, 1)],  # type: ignore
                height=4,
                width=4,
            )

        with pytest.raises(ValueError, match="outside of automatic clipping"):
            SegmentationAnnotation(
                points=[  # type: ignore
                    (-2.1, 0),
                    (1.1, 0),
                    (1, 1.5),
                    (-0.6, 1),
                ],
                height=4,
                width=4,
            )

        with pytest.raises(ValueError, match="Field required"):
            SegmentationAnnotation(width=4)  # type: ignore

        with pytest.raises(
            ValueError, match="RLE counts must be a list of positive integers"
        ):
            SegmentationAnnotation(
                counts=[-1, 1, 2, 2],  # type: ignore
                height=4,
                width=4,
            )


def test_array_annotation(subtests: SubTests, tempdir: Path):
    arr = np.random.rand(100, 100)
    arr_path = tempdir / "array.npy"
    np.save(arr_path, arr)

    with subtests.test("simple"):
        annotation = ArrayAnnotation(path=arr_path)
        assert (
            ArrayAnnotation.model_validate_json(annotation.model_dump_json())
            == annotation
        )

    with subtests.test("numpy"):
        annotation = ArrayAnnotation(path=arr_path)
        assert np.array_equal(annotation.to_numpy(), arr)

        annotations = [ArrayAnnotation(path=arr_path) for _ in range(5)]
        array = ArrayAnnotation.combine_to_numpy(
            annotations, [0, 1, 2, 2, 1], 4
        )
        assert array.shape == (5, 4, 100, 100)
        assert np.allclose(array[0, 0, ...], arr)
        assert np.allclose(array[1, 1, ...], arr)
        assert np.allclose(array[2, 2, ...], arr)
        assert np.allclose(array[3, 2, ...], arr)
        assert np.allclose(array[4, 1, ...], arr)

    with subtests.test("invalid"):
        with pytest.raises(ValueError, match="Path does not"):
            ArrayAnnotation(path=Path("non_existent.npy"))

        cv2.imwrite(str(tempdir / "image.png"), np.zeros((100, 100, 3)))
        with pytest.raises(ValueError, match=r"must be a .npy file"):
            ArrayAnnotation(path=tempdir / "image.png")


def test_classification_annotation():
    arr = ClassificationAnnotation.combine_to_numpy(
        [ClassificationAnnotation() for _ in range(5)], [0, 1, 2, 2, 1], 6
    )
    assert np.array_equal(arr, np.array([1, 1, 1, 0, 0, 0]))


def test_detection(subtests: SubTests):
    with subtests.test("rescaling"):
        detection = Detection(
            **{
                "class": "person",
                "scale_to_boxes": True,
                "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.5},
                "keypoints": {
                    "keypoints": [(0.2, 0.4, 2), (0.5, 0.8, 2)],
                },
            }
        )

        assert detection.keypoints is not None
        assert detection.keypoints.keypoints == [
            (0.2 * 0.5 + 0.1, 0.4 * 0.5 + 0.2, 2),
            (0.5 * 0.5 + 0.1, 0.8 * 0.5 + 0.2, 2),
        ]

    with subtests.test("no_rescaling"):
        detection = Detection(
            **{
                "class": "person",
                "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.5},
                "keypoints": {
                    "keypoints": [(0.2, 0.4, 2), (0.5, 0.8, 2)],
                },
            }
        )

        assert detection.keypoints is not None
        assert detection.keypoints.keypoints == [(0.2, 0.4, 2), (0.5, 0.8, 2)]

    with (
        subtests.test("invalid"),
        pytest.raises(ValueError, match="no bounding box is provided"),
    ):
        Detection(
            **{
                "class": "person",
                "scale_to_boxes": True,
                "keypoints": {
                    "keypoints": [(0.2, 0.4, 2), (0.5, 0.8, 2)],
                },
            }
        )


def test_record(tempdir: Path):
    detection = Detection(
        **{
            "class": "person",
            "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.5},
            "keypoints": {
                "keypoints": [(0.2, 0.4, 2), (0.5, 0.8, 2)],
            },
            "segmentation": {
                "mask": np.array(
                    [
                        [0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 1],
                    ]
                ),
            },
            "instance_segmentation": {
                "mask": np.array(
                    [
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ),
            },
            "metadata": {"age": 25},
            "sub_detections": {
                "head": {
                    "boundingbox": {
                        "x": 0.2,
                        "y": 0.3,
                        "w": 0.1,
                        "h": 0.1,
                    },
                }
            },
        }
    )
    filename = str((tempdir / "image.jpg").resolve())
    cv2.imwrite(filename, np.zeros((256, 256, 3), dtype=np.uint8))
    record = DatasetRecord(
        file=filename,  # type: ignore
        annotation=detection,
        task_name="test",
    )
    common = {
        "file": filename,
        "source_name": "image",
        "instance_id": -1,
        "sample_metadata": DEFAULT_METADATA,
    }
    expected_rows = [
        {
            **common,
            "task_name": "test",
            "class_name": "person",
            "task_type": "boundingbox",
            "annotation": '{"x":0.1,"y":0.2,"w":0.5,"h":0.5}',
        },
        {
            **common,
            "task_name": "test",
            "class_name": "person",
            "task_type": "keypoints",
            "annotation": '{"keypoints":[[0.2,0.4,2],[0.5,0.8,2]]}',
        },
        {
            **common,
            "task_name": "test",
            "class_name": "person",
            "task_type": "segmentation",
            "annotation": '{"height":4,"width":4,"counts":"11213ON0"}',
        },
        {
            **common,
            "task_name": "test",
            "class_name": "person",
            "task_type": "instance_segmentation",
            "annotation": '{"height":4,"width":4,"counts":"02208"}',
        },
        {
            **common,
            "task_name": "test",
            "class_name": "person",
            "task_type": "metadata/age",
            "annotation": "25",
        },
        {
            **common,
            "task_name": "test",
            "class_name": "person",
            "task_type": "classification",
            "annotation": "{}",
        },
        {
            **common,
            "task_name": "test/head",
            "class_name": None,
            "task_type": "boundingbox",
            "annotation": '{"x":0.2,"y":0.3,"w":0.1,"h":0.1}',
        },
    ]
    assert list(record.to_parquet_rows()) == expected_rows


def test_parquet_file_manager_adds_missing_metadata_column(
    tempdir: Path,
) -> None:
    annotations_path = tempdir / "annotations"
    annotations_path.mkdir()
    parquet_path = annotations_path / "0000000000.parquet"
    pl.DataFrame(
        [
            {
                "file": "old.jpg",
                "source_name": "image",
                "task_name": "",
                "class_name": None,
                "instance_id": None,
                "task_type": None,
                "annotation": None,
                "uuid": "old",
                "group_id": "old-group",
            }
        ]
    ).write_parquet(parquet_path)

    new_metadata = json.dumps({"source": "new"})
    new_row: ParquetRecord = {
        "file": "new.jpg",
        "source_name": "image",
        "task_name": "",
        "class_name": None,
        "instance_id": None,
        "task_type": None,
        "annotation": None,
        "sample_metadata": new_metadata,
    }

    with ParquetFileManager(annotations_path) as manager:
        manager.write("new", new_row, "new-group")

    df = pl.read_parquet(parquet_path)
    assert df["sample_metadata"].to_list() == [DEFAULT_METADATA, new_metadata]
