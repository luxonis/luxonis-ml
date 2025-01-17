from typing import Final, Set

import pytest

from luxonis_ml.data import LuxonisLoader, LuxonisParser
from luxonis_ml.data.utils import get_task_type
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import environ

WORK_DIR: Final[str] = "tests/data/parser_datasets"


@pytest.fixture(scope="module", autouse=True)
def prepare_dir():
    import os
    import shutil

    os.makedirs(WORK_DIR, exist_ok=True)
    yield
    shutil.rmtree(WORK_DIR)


@pytest.mark.parametrize(
    ("dataset_type", "url", "expected_task_types"),
    [
        (
            DatasetType.COCO,
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
            DatasetType.COCO,
            "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
            {"boundingbox", "classification"},
        ),
        (
            DatasetType.COCO,
            "roboflow://team-roboflow/coco-128/2/coco",
            {"boundingbox", "classification"},
        ),
        (
            DatasetType.VOC,
            "Thermal_Dogs_and_People.v1-resize-416x416.voc.zip",
            {"boundingbox", "classification"},
        ),
        (
            DatasetType.DARKNET,
            "Thermal_Dogs_and_People.v1-resize-416x416.darknet.zip",
            {"boundingbox", "classification"},
        ),
        (
            DatasetType.YOLOV4,
            "Thermal_Dogs_and_People.v1-resize-416x416.yolov4pytorch.zip",
            {"boundingbox", "classification"},
        ),
        (
            DatasetType.YOLOV6,
            "Thermal_Dogs_and_People.v1-resize-416x416.mt-yolov6.zip",
            {"boundingbox", "classification"},
        ),
        (
            DatasetType.CREATEML,
            "Thermal_Dogs_and_People.v1-resize-416x416.createml.zip",
            {"boundingbox", "classification"},
        ),
        (
            DatasetType.TFCSV,
            "Thermal_Dogs_and_People.v1-resize-416x416.tensorflow.zip",
            {"boundingbox", "classification"},
        ),
        (
            DatasetType.SEGMASK,
            "D2_Tile.png-mask-semantic.zip",
            {"segmentation", "classification"},
        ),
        (
            DatasetType.CLSDIR,
            "Flowers_Classification.v2-raw.folder.zip",
            {"classification"},
        ),
        (
            DatasetType.SOLO,
            "D2_ParkingLot.zip",
            {"boundingbox", "segmentation", "classification", "keypoints"},
        ),
        (
            DatasetType.NATIVE,
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
    ],
)
def test_dir_parser(
    dataset_type: DatasetType,
    url: str,
    expected_task_types: Set[str],
    storage_url: str,
):
    if not url.startswith("roboflow://"):
        url = f"{storage_url}/{url}"

    elif environ.ROBOFLOW_API_KEY is None:
        pytest.skip("Roboflow API key is not set")

    parser = LuxonisParser(
        url,
        dataset_name=f"test-{dataset_type}",
        delete_existing=True,
        save_dir=WORK_DIR,
    )
    dataset = parser.parse()
    assert len(dataset) > 0
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == expected_task_types
    dataset.delete_dataset()
