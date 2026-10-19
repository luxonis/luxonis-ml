import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from luxonis_ml.data.exporters import (
    UltralyticsNDJSONExporter,
    YoloV8InstanceSegmentationExporter,
    YoloV8KeypointsExporter,
)
from luxonis_ml.data.exporters.segmentation_mask_directory_exporter import (
    _write_mask,
)
from luxonis_ml.data.exporters.tensorflow_csv_exporter import _csv_rows
from luxonis_ml.data.exporters.voc_exporter import _voc_objects
from luxonis_ml.data.exporters.yolov8_keypoints_exporter import (
    _flat_keypoints,
)
from luxonis_ml.enums import DatasetType

BOX = json.dumps({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4})


def _keypoint_row(keypoints: Any, class_name: str | None = "person") -> dict:
    return {
        "task_type": "keypoints",
        "class_name": class_name,
        "annotation": json.dumps({"keypoints": keypoints}),
    }


def test_keypoint_line_skips_rows_it_cannot_export(tempdir: Path):
    exporter = YoloV8KeypointsExporter("ds", tempdir, None)

    assert (
        exporter._keypoint_line(_keypoint_row([[0.1, 0.2, 2]], None)) is None
    )
    assert exporter._keypoint_line(_keypoint_row([])) is None
    assert exporter._keypoint_line(_keypoint_row([0.1, 0.2])) is None
    assert exporter.class_names == ["person"]


def test_keypoint_line_gives_pairs_full_visibility(tempdir: Path):
    exporter = YoloV8KeypointsExporter("ds", tempdir, None)

    line = exporter._keypoint_line(_keypoint_row([[0.25, 0.5], [0.75, 0.5]]))

    assert line is not None
    assert line.split()[7::3] == ["2", "2"]
    assert exporter.kpt_shape == (2, 2)


def test_flat_keypoints_default_an_unreadable_visibility():
    keypoints = np.array([[0.5, 0.5, np.nan]])

    assert _flat_keypoints(keypoints) == "0.500000000000 0.500000000000 2"


def test_polygon_lines_need_a_class(tempdir: Path):
    exporter = YoloV8InstanceSegmentationExporter("ds", tempdir, None)
    row = {
        "task_type": "instance_segmentation",
        "class_name": None,
        "annotation": "{}",
    }

    assert exporter._polygon_lines(row, tempdir / "image.png") == []


def _box_rows() -> pl.DataFrame:
    """One usable box between a row without a class and one without a box."""
    return pl.DataFrame(
        {
            "task_type": ["boundingbox"] * 3,
            "class_name": [None, "car", "car"],
            "annotation": [BOX, None, BOX],
        }
    )


def test_csv_rows_keep_boxes_with_a_class():
    rows = _csv_rows(_box_rows(), "0.jpg", 100, 50)

    assert [(r["class"], r["xmin"], r["ymin"]) for r in rows] == [
        ("car", 10, 10)
    ]


def test_voc_objects_keep_boxes_with_a_class():
    objects = _voc_objects(_box_rows(), 100, 50)

    assert [(o["name"], o["bbox"]) for o in objects] == [
        ("car", (10, 10, 40, 30))
    ]


def test_pose_lines_skip_instances_without_keypoints(tempdir: Path):
    exporter = UltralyticsNDJSONExporter(
        "ds",
        tempdir,
        None,
        dataset_type=DatasetType.ULTRALYTICSNDJSONKEYPOINTS,
        ndjson_task="pose",
    )
    box = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    instances = {
        ("car", 0): {"class_id": 0, "boundingbox": box},
        ("car", 1): {
            "class_id": 0,
            "boundingbox": box,
            "keypoints": {"keypoints": [[0.2, 0.2, 2]]},
        },
    }

    poses = exporter._pose_lines(instances)

    assert len(poses) == 1
    assert poses[0][-3:] == [0.2, 0.2, 2]


def test_segmentation_mask_must_fit_in_8_bits(tempdir: Path):
    mask = np.full((2, 2), 300, dtype=np.uint16)

    with pytest.raises(ValueError, match="exceeds 255"):
        _write_mask(mask, tempdir / "mask.png")
