from typing import Final, List

import pytest

from luxonis_ml.data import LabelType, LuxonisLoader, LuxonisParser
from luxonis_ml.enums import DatasetType

URL_PREFIX: Final[str] = "gs://luxonis-test-bucket/luxonis-ml-test-data"
WORK_DIR: Final[str] = "tests/data/parser_datasets"


@pytest.fixture(scope="module", autouse=True)
def prepare_dir():
    import os
    import shutil

    os.makedirs(WORK_DIR, exist_ok=True)
    yield
    shutil.rmtree(WORK_DIR)


@pytest.mark.parametrize(
    ("dataset_type", "url", "expected_label_types"),
    [
        (
            DatasetType.COCO,
            "COCO_people_subset.zip",
            [
                LabelType.BOUNDINGBOX,
                LabelType.KEYPOINTS,
                LabelType.SEGMENTATION,
                LabelType.CLASSIFICATION,
            ],
        ),
        (
            DatasetType.COCO,
            "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
            [LabelType.BOUNDINGBOX, LabelType.CLASSIFICATION],
        ),
        (
            DatasetType.VOC,
            "Thermal_Dogs_and_People.v1-resize-416x416.voc.zip",
            [LabelType.BOUNDINGBOX, LabelType.CLASSIFICATION],
        ),
        (
            DatasetType.DARKNET,
            "Thermal_Dogs_and_People.v1-resize-416x416.darknet.zip",
            [LabelType.BOUNDINGBOX, LabelType.CLASSIFICATION],
        ),
        (
            DatasetType.YOLOV4,
            "Thermal_Dogs_and_People.v1-resize-416x416.yolov4pytorch.zip",
            [LabelType.BOUNDINGBOX, LabelType.CLASSIFICATION],
        ),
        (
            DatasetType.YOLOV6,
            "Thermal_Dogs_and_People.v1-resize-416x416.mt-yolov6.zip",
            [LabelType.BOUNDINGBOX, LabelType.CLASSIFICATION],
        ),
        (
            DatasetType.CREATEML,
            "Thermal_Dogs_and_People.v1-resize-416x416.createml.zip",
            [LabelType.BOUNDINGBOX, LabelType.CLASSIFICATION],
        ),
        (
            DatasetType.TFCSV,
            "Thermal_Dogs_and_People.v1-resize-416x416.tensorflow.zip",
            [LabelType.BOUNDINGBOX, LabelType.CLASSIFICATION],
        ),
        (
            DatasetType.SEGMASK,
            "D2_Tile.png-mask-semantic.zip",
            [LabelType.SEGMENTATION, LabelType.CLASSIFICATION],
        ),
        (
            DatasetType.CLSDIR,
            "Flowers_Classification.v2-raw.folder.zip",
            [LabelType.CLASSIFICATION],
        ),
        (
            DatasetType.SOLO,
            "D1_ParkingSlot-solo.zip",
            [LabelType.BOUNDINGBOX, LabelType.SEGMENTATION],
        ),
    ],
)
def test_dir_parser(
    dataset_type: DatasetType, url: str, expected_label_types: List[LabelType]
):
    parser = LuxonisParser(
        f"{URL_PREFIX}/{url}",
        dataset_name=f"test-{dataset_type}",
        delete_existing=True,
        save_dir=WORK_DIR,
    )
    dataset = parser.parse()
    assert len(dataset) > 0
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    label_types = {label_type for _, label_type in ann.values()}
    assert label_types == set(expected_label_types)
    dataset.delete_dataset()


def test_custom_tasks():
    parser = LuxonisParser(
        f"{URL_PREFIX}/Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
        dataset_name="test-custom-tasks",
        delete_existing=True,
        save_dir=WORK_DIR,
        task_mapping={LabelType.BOUNDINGBOX: "object_detection"},
    )
    dataset = parser.parse()
    assert len(dataset) > 0
    tasks = dataset.get_tasks()
    assert set(tasks) == {"object_detection", "classification"}
