from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pytest
from pytest_subtests.plugin import SubTests
from utils import create_image

from luxonis_ml.data import (
    BucketStorage,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
    LuxonisSource,
)
from luxonis_ml.data.utils.task_utils import get_task_type
from luxonis_ml.enums import DatasetType


def compare_loader_output(loader: LuxonisLoader, tasks: Set[str]):
    all_labels = set()
    for _, labels in loader:
        all_labels.update(labels.keys())
    assert all_labels == tasks


def test_dataset(
    bucket_storage: BucketStorage,
    dataset_name: str,
    augmentation_config: List[Dict[str, Any]],
    subtests: SubTests,
    storage_url: str,
    tempdir: Path,
):
    with subtests.test("test_create", bucket_storage=bucket_storage):
        parser = LuxonisParser(
            f"{storage_url}/COCO_people_subset.zip",
            dataset_name=dataset_name,
            save_dir=tempdir,
            dataset_type=DatasetType.COCO,
            bucket_storage=bucket_storage,
            delete_existing=True,
            delete_remote=True,
            task_name="coco",
        )
        parser.parse()
        dataset = LuxonisDataset(dataset_name, bucket_storage=bucket_storage)
        assert LuxonisDataset.exists(
            dataset_name, bucket_storage=bucket_storage
        )
        assert set(dataset.get_task_names()) == {"coco"}
        assert dataset.get_classes().get("coco") == ["person"]
        assert dataset.get_skeletons() == {
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
            for task in {"segmentation", "keypoints", "boundingbox"}:
                assert f"coco/{task}" in labels

    with subtests.test("test_load_aug", bucket_storage=bucket_storage):
        loader = LuxonisLoader(
            dataset,
            width=512,
            height=512,
            augmentation_config=augmentation_config,
            augmentation_engine="albumentations",
        )
        for img, labels in loader:
            assert img is not None
            for task in {"segmentation", "keypoints", "boundingbox"}:
                assert f"coco/{task}" in labels

    with subtests.test("test_delete", bucket_storage=bucket_storage):
        dataset.delete_dataset(delete_remote=True)
        assert not LuxonisDataset.exists(
            dataset_name, bucket_storage=bucket_storage
        )


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_dataset_fail(tempdir: Path):
    dataset = LuxonisDataset("test_fail", delete_existing=True)

    def generator():
        for i in range(10):
            img = create_image(i, tempdir)
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
def test_loader_iterator(storage_url: str, tempdir: Path):
    dataset = LuxonisParser(
        f"{storage_url}/COCO_people_subset.zip",
        dataset_name="_iterator_test",
        save_dir=tempdir,
        dataset_type=DatasetType.COCO,
        delete_existing=True,
        task_name="coco",
    ).parse()
    loader = LuxonisLoader(dataset)

    def _raise(*_):
        raise IndexError

    loader._load_data = _raise  # type: ignore
    with pytest.raises(IndexError):
        _ = loader[0]


def test_make_splits(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    definitions: Dict[str, List[str]] = defaultdict(list)

    _start_index: int = 0

    def generator(step=15):
        nonlocal _start_index
        definitions.clear()
        for i in range(_start_index, _start_index + step):
            path = create_image(i, tempdir)
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


def test_metadata(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    def generator():
        img = create_image(0, tempdir)
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


def test_no_labels(tempdir: Path):
    dataset = LuxonisDataset("no_labels", delete_existing=True)

    def generator():
        for i in range(10):
            img = create_image(i, tempdir)
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


def test_deep_nested_labels(
    augmentation_config: List[Dict[str, Any]], tempdir: Path
):
    def generator():
        for i in range(10):
            yield {
                "file": create_image(i, tempdir),
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

    dataset = LuxonisDataset("deep_nested_labels", delete_existing=True)
    dataset.add(generator())
    dataset.make_splits()

    loader = LuxonisLoader(
        dataset,
        height=512,
        width=512,
        augmentation_config=augmentation_config,
        augmentation_engine="albumentations",
    )
    compare_loader_output(
        loader,
        {
            "/classification",
            "/boundingbox",
            "/driver/boundingbox",
            "/driver/keypoints",
            "/license_plate/boundingbox",
            "/license_plate/metadata/text",
        },
    )


def test_partial_labels(tempdir: Path):
    dataset = LuxonisDataset("partial_labels", delete_existing=True)

    def generator():
        for i in range(8):
            img = create_image(i, tempdir)
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
        augmentation_config=[{"name": "Rotate"}],
    )

    compare_loader_output(
        loader,
        {
            "/classification",
            "/boundingbox",
            "/keypoints",
            "/segmentation",
        },
    )


def test_clone_dataset(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    dataset = LuxonisDataset(
        dataset_name,
        bucket_storage=bucket_storage,
        delete_existing=True,
        delete_remote=True,
    )

    def generator1():
        for i in range(3):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }

    dataset.add(generator1())
    dataset.make_splits({"train": 0.6, "val": 0.4})

    cloned_dataset = dataset.clone(new_dataset_name=dataset_name + "_cloned")

    assert cloned_dataset.get_splits() == dataset.get_splits()
    assert cloned_dataset.get_classes() == dataset.get_classes()
    assert cloned_dataset.get_task_names() == dataset.get_task_names()
    assert cloned_dataset.get_skeletons() == dataset.get_skeletons()

    df_cloned = cloned_dataset._load_df_offline()
    df_original = dataset._load_df_offline()
    assert df_cloned is not None
    assert df_original is not None
    assert df_cloned.equals(df_original)


def test_merge_datasets(
    bucket_storage: BucketStorage,
    dataset_name: str,
    tempdir: Path,
    subtests: SubTests,
):
    dataset_name = f"{dataset_name}_{bucket_storage.value}"
    dataset1_name = f"{dataset_name}_1"
    dataset1 = LuxonisDataset(
        dataset1_name,
        bucket_storage=bucket_storage,
        delete_existing=True,
        delete_remote=True,
    )

    def generator1():
        for i in range(3):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }

    dataset1.add(generator1())
    dataset1.make_splits({"train": 0.6, "val": 0.4})

    dataset2_name = f"{dataset_name}_2"
    dataset2 = LuxonisDataset(
        dataset2_name,
        bucket_storage=bucket_storage,
        delete_existing=True,
        delete_remote=True,
    )

    def generator2():
        for i in range(3, 6):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "dog",
                    "boundingbox": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2},
                },
            }

    dataset2.add(generator2())
    dataset2.make_splits({"train": 0.6, "val": 0.4})

    with subtests.test("test_inplace"):
        cloned_dataset1 = dataset1.clone(
            new_dataset_name=f"{dataset1_name}_cloned"
        )
        cloned_dataset1_merged_with_dataset2 = cloned_dataset1.merge_with(
            dataset2, inplace=True
        )

        classes = cloned_dataset1_merged_with_dataset2.get_classes()
        assert set(classes[""]) == {"person", "dog"}

    with subtests.test("test_out_of_place"):
        dataset1_merged_with_dataset2 = dataset1.merge_with(
            dataset2,
            inplace=False,
            new_dataset_name=f"{dataset1_name}_{dataset2_name}_merged",
        )

    classes = dataset1_merged_with_dataset2.get_classes()
    assert set(classes[""]) == {"person", "dog"}

    df_merged = dataset1_merged_with_dataset2._load_df_offline()
    df_cloned_merged = dataset1.merge_with(
        dataset2, inplace=True
    )._load_df_offline()
    assert df_merged is not None
    assert df_cloned_merged is not None
    assert df_merged.equals(df_cloned_merged)
