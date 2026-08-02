"""End-to-end parsing of the real datasets kept in cloud storage."""

from pathlib import Path

import pytest

from luxonis_ml.data import (
    LuxonisDataset,
    LuxonisLoader,
)
from luxonis_ml.data.utils import get_task_type
from luxonis_ml.utils import environ


def _resolve_url(url: str, storage_url: str) -> str:
    """Return the fetchable URL, or skip if credentials are missing."""
    if not url.startswith(("roboflow://", "ultralytics://")):
        return f"{storage_url}/{url}"
    if url.startswith("roboflow://") and environ.ROBOFLOW_API_KEY is None:
        pytest.skip("Roboflow API key is not set")
    if (
        url.startswith("ultralytics://")
        and environ.ULTRALYTICS_API_KEY is None
    ):
        pytest.skip("Ultralytics API key is not set")
    return url


@pytest.mark.parametrize(
    ("url", "expected_task_types"),
    [
        (
            "COCO_people_subset.zip",
            {
                "boundingbox",
                "keypoints",
                "segmentation",
                "classification",
                "instance_segmentation",
            },
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
            {"boundingbox", "classification"},
        ),
        (
            "roboflow://team-roboflow/coco-128/2/coco",
            {"boundingbox", "classification"},
        ),
        (
            "ultralytics://ultralytics/datasets/coco8",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.voc.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.darknet.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.yolov4pytorch.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.mt-yolov6.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.createml.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.tensorflow.zip",
            {"boundingbox", "classification"},
        ),
        (
            "D2_Tile.png-mask-semantic.zip",
            {"segmentation", "classification"},
        ),
        (
            "Flowers_Classification.v2-raw.folder.zip",
            {"classification"},
        ),
        (
            "D2_ParkingLot.zip",
            {"boundingbox", "segmentation", "classification", "keypoints"},
        ),
        (
            "D2_ParkingLot_Native.zip",
            {
                "boundingbox",
                "instance_segmentation",
                "classification",
                "keypoints",
                "metadata/color",
                "metadata/brand",
            },
        ),
        (
            "horse_pose.v8i.yolov8.zip",
            {"boundingbox", "classification", "keypoints"},
        ),
        (
            "medical-pills.zip",
            {"boundingbox", "classification"},
        ),
        (
            "crack-seg.zip",
            {
                "boundingbox",
                "classification",
                "instance_segmentation",
            },
        ),
    ],
)
def test_dir_parser(
    dataset_name: str,
    url: str,
    expected_task_types: set[str],
    storage_url: str,
    tempdir: Path,
):
    url = _resolve_url(url, storage_url)

    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) > 0
        loader = LuxonisLoader(dataset)
        _, ann = next(iter(loader))
        task_types = {get_task_type(task) for task in ann}
        assert task_types == expected_task_types
    finally:
        dataset.delete_dataset(delete_local=True)


@pytest.mark.parametrize(
    ("url", "dataset_type", "expected_task_types"),
    [
        (
            "COCO_people_subset.zip",
            "coco",
            {
                "boundingbox",
                "keypoints",
                "segmentation",
                "classification",
                "instance_segmentation",
            },
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
            "coco",
            {"boundingbox", "classification"},
        ),
        (
            "roboflow://team-roboflow/coco-128/2/coco",
            "coco",
            {"boundingbox", "classification"},
        ),
        (
            "ultralytics://ultralytics/datasets/coco8",
            "ultralytics-ndjson",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.voc.zip",
            "voc",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.darknet.zip",
            "darknet",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.yolov4pytorch.zip",
            "yolov4",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.mt-yolov6.zip",
            "yolov6",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.createml.zip",
            "createml",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.tensorflow.zip",
            "tfcsv",
            {"boundingbox", "classification"},
        ),
        (
            "D2_Tile.png-mask-semantic.zip",
            "segmask",
            {"segmentation", "classification"},
        ),
        (
            "Flowers_Classification.v2-raw.folder.zip",
            "clsdir",
            {"classification"},
        ),
        (
            "D2_ParkingLot.zip",
            "solo",
            {"boundingbox", "segmentation", "classification", "keypoints"},
        ),
        (
            "D2_ParkingLot_Native.zip",
            "native",
            {
                "boundingbox",
                "instance_segmentation",
                "classification",
                "keypoints",
                "metadata/color",
                "metadata/brand",
            },
        ),
        (
            "horse_pose.v8i.yolov8.zip",
            "yolov8",
            {"boundingbox", "classification", "keypoints"},
        ),
        (
            "medical-pills.zip",
            "yolov8",
            {"boundingbox", "classification"},
        ),
        (
            "crack-seg.zip",
            "yolov8",
            {
                "boundingbox",
                "classification",
                "instance_segmentation",
            },
        ),
    ],
)
def test_dir_parser_explicit_type(
    dataset_name: str,
    url: str,
    dataset_type: str,
    expected_task_types: set[str],
    storage_url: str,
    tempdir: Path,
):
    url = _resolve_url(url, storage_url)

    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) > 0
        loader = LuxonisLoader(dataset)
        _, ann = next(iter(loader))
        task_types = {get_task_type(task) for task in ann}
        assert task_types == expected_task_types
    finally:
        dataset.delete_dataset(delete_local=True)
