from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pydantic
import pytest
from pytest_subtests import SubTests

from luxonis_ml.data.datasets.annotation import (
    ArrayAnnotation,
    BBoxAnnotation,
    ClassificationAnnotation,
    DatasetRecord,
    Detection,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    SegmentationAnnotation,
    check_valid_identifier,
    load_annotation,
)


def test_valid_identifier():
    check_valid_identifier("variable", label="")
    check_valid_identifier("variable_name", label="")
    check_valid_identifier("variable-name", label="")

    with pytest.raises(ValueError):
        check_valid_identifier("variable name", label="")

    with pytest.raises(ValueError):
        check_valid_identifier("?variable_name", label="")

    with pytest.raises(ValueError):
        check_valid_identifier("12variable_name", label="")

    with pytest.raises(ValueError):
        check_valid_identifier("variable/name", label="")


def test_load_annotation():
    assert load_annotation(
        "boundingbox", {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    ) == BBoxAnnotation(x=0.1, y=0.2, w=0.3, h=0.4)
    with pytest.raises(ValueError):
        load_annotation("invalid_name", {})


def test_dataset_record(tempdir: Path):
    def compare_parquet_rows(
        record: DatasetRecord, expected_rows: List[Dict[str, Any]]
    ):
        rows = list(record.to_parquet_rows())
        for row in rows:
            del row["created_at"]  # type: ignore
        assert rows == expected_rows

    cv2.imwrite(str(tempdir / "left.jpg"), np.zeros((100, 100, 3)))
    cv2.imwrite(str(tempdir / "right.jpg"), np.zeros((100, 100, 3)))
    record = DatasetRecord(file=tempdir / "left.jpg")  # type: ignore
    assert record.file == tempdir / "left.jpg"

    compare_parquet_rows(
        record,
        [
            {
                "file": "tests/data/tempdir/left.jpg",
                "source_name": "image",
                "task_name": "detection",
                "class_name": None,
                "instance_id": None,
                "task_type": None,
                "annotation": None,
            }
        ],
    )

    record = DatasetRecord(
        file=tempdir / "left.jpg",  # type: ignore
        annotation={
            "class": "person",
            "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        },
    )
    compare_parquet_rows(
        record,
        [
            {
                "file": "tests/data/tempdir/left.jpg",
                "source_name": "image",
                "task_name": "detection",
                "class_name": "person",
                "instance_id": -1,
                "task_type": "boundingbox",
                "annotation": '{"x":0.1,"y":0.2,"w":0.3,"h":0.4}',
            },
            {
                "file": "tests/data/tempdir/left.jpg",
                "source_name": "image",
                "task_name": "detection",
                "class_name": "person",
                "instance_id": -1,
                "task_type": "classification",
                "annotation": "{}",
            },
        ],
    )

    with pytest.raises(ValueError):
        record = DatasetRecord(
            files={
                "left": tempdir / "left.jpg",
                "right": tempdir / "right.jpg",
            }
        )
        _ = record.file


def test_bbox_annotation(subtests: SubTests):
    with subtests.test("simple"):
        bbox = BBoxAnnotation(x=0.1, y=0.2, w=0.3, h=0.4)
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.w == 0.3
        assert bbox.h == 0.4

    with subtests.test("no_auto_clip"):
        base_dict = {"x": 0, "y": 0, "w": 0, "h": 0}
        for k in ["x", "y", "w", "h"]:
            for v in [-2.1, 2.3, -3.3, 3]:
                with pytest.raises(pydantic.ValidationError):
                    curr_dict = base_dict.copy()
                    curr_dict[k] = v
                    BBoxAnnotation(**curr_dict)  # type: ignore

        bbox_ann = BBoxAnnotation(x=0.9, y=0, w=0.2, h=0)
        assert bbox_ann.x + bbox_ann.w <= 1
        bbox_ann = BBoxAnnotation(x=1.2, y=0, w=0.2, h=0)
        assert bbox_ann.x + bbox_ann.w <= 1
        bbox_ann = BBoxAnnotation(x=0, y=0.9, w=0, h=0.2)
        assert bbox_ann.y + bbox_ann.h <= 1
        bbox_ann = BBoxAnnotation(x=0, y=1.2, w=0, h=0.2)
        assert bbox_ann.y + bbox_ann.h <= 1

    with subtests.test("auto_clip"):
        base_dict = {"x": 0, "y": 0, "w": 0, "h": 0}
        for k in ["x", "y", "w", "h"]:
            for v in [-1.1, 1.3, -1.3, 2]:
                curr_dict = base_dict.copy()
                curr_dict[k] = v
                bbox_ann = BBoxAnnotation(**curr_dict)  # type: ignore
                assert 0 <= bbox_ann.x <= 1
                assert 0 <= bbox_ann.y <= 1
                assert 0 <= bbox_ann.w <= 1
                assert 0 <= bbox_ann.h <= 1

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
            BBoxAnnotation.combine_to_numpy(bboxes, [1, 2, 3], ...),
            np.array(
                [
                    [1, 0.1, 0.2, 0.3, 0.4],
                    [2, 0.2, 0.3, 0.4, 0.5],
                    [3, 0.3, 0.4, 0.5, 0.6],
                ]
            ),
        )


def test_keypoints_annotation(subtests: SubTests):
    with subtests.test("no_auto_clip"):
        with pytest.raises(pydantic.ValidationError):
            KeypointAnnotation(keypoints=[(-2.1, 1.1, 0)])
        with pytest.raises(pydantic.ValidationError):
            KeypointAnnotation(keypoints=[(0.1, 2.1, 1)])
        with pytest.raises(pydantic.ValidationError):
            KeypointAnnotation(keypoints=[(0.1, 1.1, 2), (0.1, 2.1, 1)])

    with subtests.test("auto_clip"):
        kpt_ann = KeypointAnnotation(keypoints=[(-1.1, 1.1, 0)])
        assert (
            0 <= kpt_ann.keypoints[0][0] <= 1
            and 0 <= kpt_ann.keypoints[0][1] <= 1
        )
        kpt_ann = KeypointAnnotation(keypoints=[(0.1, 1.1, 1)])
        assert (
            0 <= kpt_ann.keypoints[0][0] <= 1
            and 0 <= kpt_ann.keypoints[0][1] <= 1
        )
        kpt_ann = KeypointAnnotation(keypoints=[(-2, 2, 2)])
        assert (
            0 <= kpt_ann.keypoints[0][0] <= 1
            and 0 <= kpt_ann.keypoints[0][1] <= 1
        )
    with subtests.test("numpy"):
        keypoints = KeypointAnnotation(keypoints=[(0.1, 0.2, 2)])
        assert np.allclose(keypoints.to_numpy(), np.array([0.1, 0.2, 2]))
        keypoints_list = [
            keypoints,
            KeypointAnnotation(keypoints=[(0.2, 0.3, 0)]),
            KeypointAnnotation(keypoints=[(0.3, 0.4, 1)]),
        ]
        assert np.allclose(
            KeypointAnnotation.combine_to_numpy(keypoints_list, ..., ...),
            np.array([[0.1, 0.2, 2], [0.2, 0.3, 0], [0.3, 0.4, 1]]),
        )


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
        assert (
            seg.model_dump_json()
            == '{"height":4,"width":4,"counts":"11213ON0"}'
        )
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
        with pytest.raises(ValueError):
            SegmentationAnnotation(
                mask=mask,  # type: ignore
                points=[(1, 0), (2, 1)],  # type: ignore
            )

        with pytest.raises(ValueError):
            SegmentationAnnotation(
                points=[(1, 0), (0, 1)],  # type: ignore
                height=4,
                width=4,
            )

        with pytest.raises(ValueError):
            SegmentationAnnotation(mask="file.jpeg")  # type: ignore

        with pytest.raises(ValueError):
            SegmentationAnnotation(mask="file.png")  # type: ignore

        with pytest.raises(ValueError):
            SegmentationAnnotation(mask=[1, 2, 3])  # type: ignore

        with pytest.raises(ValueError):
            np.save(tempdir / "mask.npy", mask[None, None, ...])
            SegmentationAnnotation(mask=tempdir / "mask.npy")  # type: ignore

        with pytest.raises(ValueError):
            SegmentationAnnotation(
                points=[(-2.1, 0), (1.1, 0), (1, 1.5), (-0.6, 1)],  # type: ignore
                height=4,
                width=4,
            )

        with pytest.raises(ValueError):
            SegmentationAnnotation(width=4)  # type: ignore

        with pytest.raises(ValueError):
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
        assert annotation.model_dump_json() == f'{{"path":"{arr_path}"}}'

    with subtests.test("numpy"):
        annotation = ArrayAnnotation(path=arr_path)
        assert np.array_equal(annotation.to_numpy(), arr)

        annotations = [ArrayAnnotation(path=arr_path) for _ in range(5)]
        array = ArrayAnnotation.combine_to_numpy(annotations, [0, 1, 2, 2, 1])
        assert np.allclose(array[0, 0, ...], arr)
        assert np.allclose(array[1, 1, ...], arr)
        assert np.allclose(array[2, 2, ...], arr)
        assert np.allclose(array[3, 2, ...], arr)
        assert np.allclose(array[4, 1, ...], arr)

    with subtests.test("invalid"):
        with pytest.raises(ValueError):
            ArrayAnnotation(path=Path("non_existent.npy"))

        with pytest.raises(ValueError):
            cv2.imwrite(str(tempdir / "image.png"), np.zeros((100, 100, 3)))
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

    with subtests.test("invalid"):
        with pytest.raises(ValueError):
            Detection(
                **{
                    "class": "person",
                    "scale_to_boxes": True,
                    "keypoints": {
                        "keypoints": [(0.2, 0.4, 2), (0.5, 0.8, 2)],
                    },
                }
            )

    with subtests.test("full"):
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
        expected_rows = [
            {
                "class_name": "person",
                "instance_id": -1,
                "task_type": "boundingbox",
                "annotation": '{"x":0.1,"y":0.2,"w":0.5,"h":0.5}',
            },
            {
                "class_name": "person",
                "instance_id": -1,
                "task_type": "keypoints",
                "annotation": '{"keypoints":[[0.2,0.4,2],[0.5,0.8,2]]}',
            },
            {
                "class_name": "person",
                "instance_id": -1,
                "task_type": "segmentation",
                "annotation": '{"height":4,"width":4,"counts":"11213ON0"}',
            },
            {
                "class_name": "person",
                "instance_id": -1,
                "task_type": "instance_segmentation",
                "annotation": '{"height":4,"width":4,"counts":"02208"}',
            },
            {
                "class_name": "person",
                "instance_id": -1,
                "task_type": "metadata/age",
                "annotation": "25",
            },
            {
                "class_name": "person",
                "instance_id": -1,
                "task_type": "classification",
                "annotation": "{}",
            },
            {
                "class_name": None,
                "instance_id": -1,
                "task_type": "head/boundingbox",
                "annotation": '{"x":0.2,"y":0.3,"w":0.1,"h":0.1}',
            },
        ]
        assert list(detection.to_parquet_rows()) == expected_rows
