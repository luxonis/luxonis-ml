from typing import Final, cast

import pytest

from luxonis_ml.data import (
    Augmentations,
    BucketStorage,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
)
from luxonis_ml.enums import DatasetType

URL_PREFIX: Final[str] = "gs://luxonis-test-bucket/luxonis-ml-test-data"
WORK_DIR: Final[str] = "tests/data/parser_datasets"
DATASET_NAME: Final[str] = "__test_coco"


@pytest.fixture(scope="function", autouse=True)
def delete_dataset():
    yield
    LuxonisDataset(DATASET_NAME).delete_dataset()


@pytest.mark.parametrize(
    ("bucket_storage",),
    [
        (BucketStorage.LOCAL,),
        (BucketStorage.S3,),
        (BucketStorage.GCS,),
    ],
)
def test_dataset(bucket_storage: BucketStorage, subtests):
    with subtests.test("test_create", bucket_storage=bucket_storage):
        parser = LuxonisParser(
            f"{URL_PREFIX}/COCO_people_subset.zip",
            dataset_name=DATASET_NAME,
            delete_existing=True,
            save_dir=WORK_DIR,
            dataset_type=DatasetType.COCO,
            bucket_storage=bucket_storage,
        )
        dataset = cast(LuxonisDataset, parser.parse())
        assert LuxonisDataset.exists(DATASET_NAME)

    with subtests.test("test_load", bucket_storage=bucket_storage):
        loader = LuxonisLoader(dataset)
        for img, labels in loader:
            assert img is not None
            assert "segmentation" in labels
            assert "classification" in labels
            assert "keypoints" in labels
            assert "boundingbox" in labels

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
            assert "segmentation" in labels
            assert "classification" in labels
            assert "keypoints" in labels
            assert "boundingbox" in labels

    with subtests.test("test_delete", bucket_storage=bucket_storage):
        dataset.delete_dataset()
        assert not LuxonisDataset.exists(DATASET_NAME)
