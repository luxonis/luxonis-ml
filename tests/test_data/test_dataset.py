import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Final, List, Set

import cv2
import numpy as np
import pytest
from pytest_subtests.plugin import SubTests

from luxonis_ml.data import (
    BucketStorage,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
    LuxonisSource,
)
from luxonis_ml.data.utils.task_utils import (
    get_task_type,
)
from luxonis_ml.enums import DatasetType
from luxonis_ml.typing import ConfigItem

# TODO: Test array

SKELETONS: Final[dict] = {
    "coco": (
        [
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
        [
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
    ),
}
URL_PREFIX: Final[str] = "gs://luxonis-test-bucket/luxonis-ml-test-data"
WORK_DIR: Final[str] = "tests/data/parser_datasets"
TASKS: Final[Set[str]] = {
    "segmentation",
    "keypoints",
    "boundingbox",
}
DATA_DIR = Path("tests/data/test_dataset")

AUG_CONFIG: List[ConfigItem] = [
    {
        "name": "Mosaic4",
        "params": {"out_width": 416, "out_height": 416, "p": 1.0},
    },
    {"name": "Defocus", "params": {"p": 1.0}},
    {"name": "Sharpen", "params": {"p": 1.0}},
    {"name": "Flip", "params": {"p": 1.0}},
    {"name": "RandomRotate90", "params": {"p": 1.0}},
]


@pytest.fixture(autouse=True, scope="module")
def prepare_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    yield

    shutil.rmtree(DATA_DIR)


def make_image(i: int) -> Path:
    path = DATA_DIR / f"img_{i}.jpg"
    if not path.exists():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[0:10, 0:10] = np.random.randint(
            0, 255, (10, 10, 3), dtype=np.uint8
        )
        cv2.imwrite(str(path), img)
    return path


def test_dataset(
    bucket_storage: BucketStorage, dataset_name: str, subtests: SubTests
):
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
        parser.parse()
        dataset = LuxonisDataset(dataset_name, bucket_storage=bucket_storage)
        assert LuxonisDataset.exists(
            dataset_name, bucket_storage=bucket_storage
        )
        assert dataset.get_classes()[0] == ["person"]
        assert set(dataset.get_task_names()) == {"coco"}
        assert dataset.get_skeletons() == SKELETONS
        assert dataset.identifier == dataset_name

    if "dataset" not in locals():
        pytest.exit("Dataset creation failed")

    with subtests.test("test_source"):
        assert dataset.source.to_document() == LuxonisSource().to_document()
        dataset.update_source(LuxonisSource("test"))
        assert (
            dataset.source.to_document() == LuxonisSource("test").to_document()
        )

    with subtests.test("test_load", bucket_storage=bucket_storage):
        loader = LuxonisLoader(dataset)
        for img, labels in loader:
            assert img is not None
            for task in TASKS:
                assert f"coco/{task}" in labels

    with subtests.test("test_load_aug", bucket_storage=bucket_storage):
        loader = LuxonisLoader(
            dataset,
            width=512,
            height=512,
            augmentation_config=AUG_CONFIG,
            augmentation_engine="albumentations",
        )
        for img, labels in loader:
            assert img is not None
            for task in TASKS:
                assert f"coco/{task}" in labels

    with subtests.test("test_delete", bucket_storage=bucket_storage):
        dataset.delete_dataset(delete_remote=True)
        assert not LuxonisDataset.exists(
            dataset_name, bucket_storage=bucket_storage
        )


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_dataset_fail():
    dataset = LuxonisDataset("__test_fail", delete_existing=True)

    def generator():
        for i in range(10):
            img = make_image(i)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                },
            }

    dataset.add(generator(), batch_size=2)

    with pytest.raises(ValueError):
        dataset.set_skeletons()


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_loader_iterator():
    dataset = LuxonisParser(
        f"{URL_PREFIX}/COCO_people_subset.zip",
        dataset_name="_iterator_test",
        save_dir=WORK_DIR,
        dataset_type=DatasetType.COCO,
        delete_existing=True,
    ).parse()
    loader = LuxonisLoader(dataset)

    def _raise(*_):
        raise IndexError

    loader._load_data = _raise  # type: ignore
    with pytest.raises(IndexError):
        _ = loader[0]


def test_make_splits(bucket_storage: BucketStorage, dataset_name: str):
    definitions: Dict[str, List[str]] = defaultdict(list)

    _start_index: int = 0

    def generator(step=15):
        nonlocal _start_index
        definitions.clear()
        for i in range(_start_index, _start_index + step):
            path = make_image(i)
            yield {
                "file": str(path),
                "annotation": {
                    "class": ["dog", "cat"][i % 2],
                },
            }
            definitions[["train", "val", "test"][i % 3]].append(str(path))
        _start_index += step

    dataset = LuxonisDataset(
        dataset_name,
        delete_existing=True,
        delete_remote=True,
        bucket_storage=bucket_storage,
    )
    dataset.add(generator())
    assert len(dataset) == 15
    assert dataset.get_splits() is None
    dataset.make_splits(definitions)
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits.keys()) == {"train", "val", "test"}
    for split, split_data in splits.items():
        assert (
            len(split_data) == 5
        ), f"Split {split} has {len(split_data)} samples"

    dataset.add(generator())
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        assert (
            len(split_data) == 5
        ), f"Split {split} has {len(split_data)} samples"
    dataset.make_splits(definitions)  # type: ignore
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        assert (
            len(split_data) == 10
        ), f"Split {split} has {len(split_data)} samples"

    dataset.add(generator())
    dataset.make_splits((1, 0, 0))
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        expected_length = 25 if split == "train" else 10
        assert (
            len(split_data) == expected_length
        ), f"Split {split} has {len(split_data)} samples"

    with pytest.raises(ValueError):
        dataset.make_splits()

    with pytest.raises(ValueError):
        dataset.make_splits((0.7, 0.1, 1))

    with pytest.raises(ValueError):
        dataset.make_splits((0.7, 0.1, 0.1, 0.1))  # type: ignore

    with pytest.raises(ValueError):
        dataset.make_splits((0.7, 0.1, 1), definitions=definitions)  # type: ignore

    with pytest.raises(ValueError):
        dataset.make_splits({"train": 1.5})

    with pytest.raises(ValueError):
        dataset.make_splits(
            {split: defs * 2 for split, defs in splits.items()}
        )

    dataset.add(generator(10))
    dataset.make_splits({"custom_split": 1.0})
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits.keys()) == {"train", "val", "test", "custom_split"}
    for split, split_data in splits.items():
        expected_length = 25 if split == "train" else 10
        assert (
            len(split_data) == expected_length
        ), f"Split {split} has {len(split_data)} samples"

    dataset.make_splits(replace_old_splits=True)
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        expected_length = {"train": 44, "val": 6, "test": 5}
        assert (
            len(split_data) == expected_length[split]
        ), f"Split {split} has {len(split_data)} samples"


# TODO: Test array


def test_metadata(bucket_storage: BucketStorage, dataset_name: str):
    def generator():
        img = make_image(0)
        for i in range(10):
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "metadata": {
                        "color": "red" if i % 2 == 0 else "blue",
                        "distance": 5.0,
                        "id": 127 + i,
                    },
                },
            }

    dataset = LuxonisDataset(
        dataset_name,
        delete_existing=True,
        delete_remote=True,
        bucket_storage=bucket_storage,
    )
    dataset.add(generator())
    dataset.make_splits()
    loader = LuxonisLoader(dataset)
    for _, labels in loader:
        print(labels.keys())
        labels = {get_task_type(k): v for k, v in labels.items()}
        assert {
            "metadata/color",
            "metadata/distance",
            "metadata/id",
            "classification",
        } == set(labels.keys())

        assert labels["metadata/color"].tolist() == ["red", "blue"] * 5
        assert labels["metadata/distance"].tolist() == [5.0] * 10
        assert labels["metadata/id"].tolist() == list(range(127, 137))


def test_no_labels():
    dataset = LuxonisDataset("__no_labels", delete_existing=True)

    def generator():
        for i in range(10):
            img = make_image(i)
            yield {
                "file": img,
            }

    dataset.add(generator())
    dataset.make_splits()

    loader = LuxonisLoader(dataset)
    for _, labels in loader:
        assert labels == {}

    loader = LuxonisLoader(
        dataset,
        width=512,
        height=512,
        augmentation_config=[{"name": "Flip", "params": {}}],
    )
    for _, labels in loader:
        assert labels == {}


def test_deep_nested_labels():
    def generator():
        for i in range(10):
            yield {
                "file": make_image(i),
                "annotation": {
                    "class": "car",
                    "boundingbox": {
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.1,
                        "h": 0.1,
                    },
                    "sub_detections": {
                        "license_plate": {
                            "boundingbox": {
                                "x": 0.2,
                                "y": 0.2,
                                "w": 0.1,
                                "h": 0.1,
                            },
                            "metadata": {
                                "text": "ABC123",
                            },
                        },
                        "driver": {
                            "boundingbox": {
                                "x": 0.3,
                                "y": 0.3,
                                "w": 0.1,
                                "h": 0.1,
                            },
                            "keypoints": {
                                "keypoints": [(0.1, 0.1, 2), (0.2, 0.2, 2)]
                            },
                        },
                    },
                },
            }

    dataset = LuxonisDataset("__deep_nested_labels", delete_existing=True)
    dataset.add(generator())
    dataset.make_splits()

    loader = LuxonisLoader(
        dataset,
        height=512,
        width=512,
        augmentation_config=AUG_CONFIG,
        augmentation_engine="albumentations",
    )
    _, labels = next(iter(loader))
    assert {
        "detection/classification",
        "detection/boundingbox",
        "detection/driver/boundingbox",
        "detection/driver/keypoints",
        "detection/license_plate/boundingbox",
        "detection/license_plate/metadata/text",
    } == set(labels.keys())


def test_partial_labels():
    dataset = LuxonisDataset("__partial_labels", delete_existing=True)

    def generator():
        for i in range(8):
            img = make_image(i)
            if i < 2:
                yield {
                    "file": img,
                }
            elif i < 4:
                yield {
                    "file": img,
                    "annotation": {
                        "class": "dog",
                    },
                }
            elif i < 6:
                yield {
                    "file": img,
                    "annotation": {
                        "class": "dog",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.1,
                            "h": 0.1,
                        },
                        "keypoints": {
                            "keypoints": [[0.1, 0.1, 0], [0.2, 0.2, 1]]
                        },
                    },
                }
            elif i < 8:
                yield {
                    "file": img,
                    "annotation": {
                        "class": "dog",
                        "segmentation": {
                            "mask": np.random.rand(512, 512) > 0.5
                        },
                    },
                }

    dataset.add(generator()).make_splits()
    loader = LuxonisLoader(
        dataset,
        width=512,
        height=512,
        augmentation_config=[{"name": "Rotate", "params": {}}],
    )

    all_labels = set()
    for _, labels in loader:
        all_labels.update(labels.keys())

    assert {
        "detection/classification",
        "detection/boundingbox",
        "detection/keypoints",
        "detection/segmentation",
    } == all_labels
