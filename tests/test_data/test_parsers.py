from typing import Final, List

import pytest

from luxonis_ml.data import LuxonisLoader, LuxonisParser
from luxonis_ml.enums import DatasetType, LabelType

URL_PREFIX: Final[str] = "gs://luxonis-test-bucket/luxonis-ml-test-data"
SAVE_DIR: Final[str] = "tests/data/parser_datasets"


@pytest.fixture(scope="module", autouse=True)
def prepare_dir():
    import os
    import shutil

    os.makedirs(SAVE_DIR, exist_ok=True)
    yield
    shutil.rmtree(SAVE_DIR)


@pytest.mark.parametrize(
    ("dataset_type", "url", "expected_tasks"),
    [
        (
            DatasetType.COCO,
            "COCO_people_subset.zip",
            [
                LabelType.BOUNDINGBOX,
                LabelType.KEYPOINT,
                LabelType.SEGMENTATION,
                LabelType.CLASSIFICATION,
            ],
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
    ],
)
def test_dir_parser(
    dataset_type: DatasetType, url: str, expected_tasks: List[LabelType]
):
    parser = LuxonisParser(
        f"{URL_PREFIX}/{url}",
        dataset_name=f"test-{dataset_type}",
        delete_existing=True,
        save_dir=SAVE_DIR,
    )
    dataset = parser.parse()
    assert len(dataset) > 0
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    tasks = ann.keys()
    assert set(tasks) == set(expected_tasks)
    dataset.delete_dataset()
