import platform
from typing import Final, Set, cast

import pytest

from luxonis_ml.data import (
    Augmentations,
    BucketStorage,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
)
from luxonis_ml.enums import DatasetType

SKELETONS: Final[dict] = {
    "keypoints": {
        "person": {
            "labels": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ],
            "edges": [
                [15, 13],
                [13, 11],
                [16, 14],
                [14, 12],
                [11, 12],
                [5, 11],
                [6, 12],
                [5, 6],
                [5, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [1, 2],
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
            ],
        }
    }
}
URL_PREFIX: Final[str] = "gs://luxonis-test-bucket/luxonis-ml-test-data"
WORK_DIR: Final[str] = "tests/data/parser_datasets"
DATASET_NAME: Final[str] = "__test_coco"
TASKS: Final[Set[str]] = {"segmentation", "classification", "keypoints", "boundingbox"}


@pytest.mark.parametrize(
    ("bucket_storage",),
    [
        (BucketStorage.LOCAL,),
        (BucketStorage.S3,),
        (BucketStorage.GCS,),
    ],
)
def test_dataset(bucket_storage: BucketStorage, subtests):
    os_name = platform.system().lower()
    dataset_name = f"{DATASET_NAME}-{bucket_storage.value}-{os_name}"
    with subtests.test("test_create", bucket_storage=bucket_storage):
        parser = LuxonisParser(
            f"{URL_PREFIX}/COCO_people_subset.zip",
            dataset_name=dataset_name,
            save_dir=WORK_DIR,
            dataset_type=DatasetType.COCO,
            bucket_storage=bucket_storage,
            delete_existing=True,
            delete_remote=True,
        )
        dataset = cast(LuxonisDataset, parser.parse())
        assert LuxonisDataset.exists(dataset_name, bucket_storage=bucket_storage)
        assert dataset.get_classes()[0] == ["person"]
        assert set(dataset.get_tasks()) == TASKS
        assert dataset.get_skeletons() == SKELETONS

    if "dataset" not in locals():
        pytest.exit("Dataset creation failed")

    with subtests.test("test_load", bucket_storage=bucket_storage):
        loader = LuxonisLoader(dataset)
        for img, labels in loader:
            assert img is not None
            for task in TASKS:
                assert task in labels

    with subtests.test("test_load_aug", bucket_storage=bucket_storage):
        aug_config = [
            {
                "name": "Mosaic4",
                "params": {"out_width": 416, "out_height": 416, "p": 1.0},
            },
            {"name": "Defocus", "params": {"p": 1.0}},
            {"name": "Sharpen", "params": {"p": 1.0}},
            {"name": "Flip", "params": {"p": 1.0}},
            {"name": "RandomRotate90", "params": {"p": 1.0}},
        ]
        augmentations = Augmentations([512, 512], aug_config)
        loader = LuxonisLoader(dataset, augmentations=augmentations)
        for img, labels in loader:
            assert img is not None
            for task in TASKS:
                assert task in labels

    with subtests.test("test_delete", bucket_storage=bucket_storage):
        dataset.delete_dataset(delete_remote=True)
        assert not LuxonisDataset.exists(dataset_name, bucket_storage=bucket_storage)
