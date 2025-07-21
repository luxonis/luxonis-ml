import shutil
from collections import defaultdict
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
from pytest_subtests.plugin import SubTests

from luxonis_ml.data import (
    BucketStorage,
    Category,
    LuxonisComponent,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
    LuxonisSource,
    UpdateMode,
)
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.utils.task_utils import get_task_type
from luxonis_ml.enums import DatasetType
from luxonis_ml.typing import Params

from .utils import create_dataset, create_image, get_loader_output


def test_dataset(
    bucket_storage: BucketStorage,
    dataset_name: str,
    augmentation_config: list[Params],
    subtests: SubTests,
    storage_url: str,
    tempdir: Path,
):
    with subtests.test("test_create"):
        dataset = LuxonisParser(
            f"{storage_url}/COCO_people_subset.zip",
            dataset_name=dataset_name,
            save_dir=tempdir,
            dataset_type=DatasetType.COCO,
            bucket_storage=bucket_storage,
            delete_local=True,
            delete_remote=True,
            task_name="coco",
        ).parse()

        assert LuxonisDataset.exists(
            dataset_name, bucket_storage=bucket_storage
        )
        assert set(dataset.get_task_names()) == {"coco"}
        assert dataset.get_classes().get("coco") == {"person": 0}
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
                sorted(
                    [
                        (15, 13),
                        (13, 11),
                        (16, 14),
                        (14, 12),
                        (11, 12),
                        (5, 11),
                        (6, 12),
                        (5, 6),
                        (5, 7),
                        (6, 8),
                        (7, 9),
                        (8, 10),
                        (1, 2),
                        (0, 1),
                        (0, 2),
                        (1, 3),
                        (2, 4),
                        (3, 5),
                        (4, 6),
                    ]
                ),
            ),
        }
        assert dataset.get_n_keypoints() == {"coco": 17}

        assert dataset.identifier == dataset_name

    if "dataset" not in locals():
        pytest.exit("Dataset creation failed")

    with subtests.test("test_source"):
        assert dataset.source == LuxonisSource(
            name="default",
            components={"image": LuxonisComponent(name="image")},
            main_component="image",
        )
        dataset.update_source(
            LuxonisSource(
                name="test",
                components={"image": LuxonisComponent(name="image")},
                main_component="image",
            )
        )
        assert dataset.source == LuxonisSource(
            name="test",
            components={"image": LuxonisComponent(name="image")},
            main_component="image",
        )
        assert dataset.get_source_names() == ["image"]

    with subtests.test("test_load"):
        loader = LuxonisLoader(dataset)
        for img, labels in loader:
            assert img is not None
            for task in ["segmentation", "keypoints", "boundingbox"]:
                assert f"coco/{task}" in labels

    with subtests.test("test_load_aug"):
        loader = LuxonisLoader(
            dataset,
            width=512,
            height=512,
            augmentation_config=augmentation_config,
            augmentation_engine="albumentations",
        )
        for img, labels in loader:
            assert img is not None
            for task in ["segmentation", "keypoints", "boundingbox"]:
                assert f"coco/{task}" in labels

    with subtests.test("test_delete"):
        dataset.delete_dataset(delete_remote=True, delete_local=True)
        assert not LuxonisDataset.exists(
            dataset_name, bucket_storage=bucket_storage
        )


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_dataset_fail(dataset_name: str, tempdir: Path):
    def generator() -> DatasetIterator:
        for i in range(10):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                },
            }

    dataset = create_dataset(dataset_name, generator())

    with pytest.raises(ValueError, match="Must provide either"):
        dataset.set_skeletons()

    with pytest.raises(ValueError, match="Must set delete_remote"):
        dataset.delete_dataset()


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_loader_iterator(storage_url: str, tempdir: Path):
    dataset = LuxonisParser(
        f"{storage_url}/COCO_people_subset.zip",
        dataset_name="_iterator_test",
        save_dir=tempdir,
        dataset_type=DatasetType.COCO,
        delete_local=True,
        task_name="coco",
    ).parse()
    loader = LuxonisLoader(dataset)

    def _raise(*_) -> NoReturn:
        raise IndexError

    loader._load_data = _raise  # type: ignore
    with pytest.raises(IndexError):
        _ = loader[0]


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_make_splits(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    definitions: dict[str, list[str]] = defaultdict(list)

    _start_index: int = 0

    def generator(step: int = 15) -> DatasetIterator:
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

    dataset = create_dataset(
        dataset_name, generator(), bucket_storage, splits=False
    )

    assert len(dataset) == 15
    assert dataset.get_splits() is None
    dataset.make_splits(definitions)
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits.keys()) == {"train", "val", "test"}
    for split, split_data in splits.items():
        assert len(split_data) == 5, (
            f"Split {split} has {len(split_data)} samples"
        )

    dataset.add(generator())
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        assert len(split_data) == 5, (
            f"Split {split} has {len(split_data)} samples"
        )
    dataset.make_splits(definitions)  # type: ignore
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        assert len(split_data) == 10, (
            f"Split {split} has {len(split_data)} samples"
        )

    dataset.add(generator())
    dataset.make_splits((1, 0, 0))
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        expected_length = 25 if split == "train" else 10
        assert len(split_data) == expected_length, (
            f"Split {split} has {len(split_data)} samples"
        )

    with pytest.raises(ValueError, match="No new files"):
        dataset.make_splits()

    with pytest.raises(ValueError, match="Splits cannot be empty"):
        dataset.make_splits({})

    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        dataset.make_splits((0.7, 0.1, 1))

    with pytest.raises(ValueError, match="must be a tuple of 3 floats"):
        dataset.make_splits((0.7, 0.1, 0.1, 0.1))  # type: ignore

    with pytest.raises(ValueError, match="Cannot provide both splits and"):
        dataset.make_splits((0.7, 0.1, 0.2), definitions=definitions)

    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        dataset.make_splits({"train": 1.5})

    dataset.add(generator(10))
    dataset.make_splits({"custom_split": 1.0})
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits.keys()) == {"train", "val", "test", "custom_split"}
    for split, split_data in splits.items():
        expected_length = 25 if split == "train" else 10
        assert len(split_data) == expected_length, (
            f"Split {split} has {len(split_data)} samples"
        )

    dataset.make_splits(replace_old_splits=True)
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        expected_length = {"train": 44, "val": 6, "test": 5}
        assert len(split_data) == expected_length[split], (
            f"Split {split} has {len(split_data)} samples"
        )


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_metadata(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    def generator() -> DatasetIterator:
        img = create_image(0, tempdir)
        for i in range(10):
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "metadata": {
                        "color": Category("red" if i % 2 == 0 else "blue"),
                        "distance": 5.0 if i == 0 else 5,
                        "id": 127 + i,
                        "license_plate": "xyz",
                    },
                },
            }

    dataset = create_dataset(dataset_name, generator(), bucket_storage)
    loader = LuxonisLoader(dataset)
    for _, labels in loader:
        labels = {get_task_type(k): v for k, v in labels.items()}
        assert {
            "metadata/color",
            "metadata/distance",
            "metadata/id",
            "metadata/license_plate",
            "classification",
        } == set(labels.keys())

        assert labels["metadata/color"].tolist() == [0, 1] * 5
        assert labels["metadata/distance"].tolist() == [5.0] * 10
        assert labels["metadata/id"].tolist() == list(range(127, 137))
        assert labels["metadata/license_plate"].tolist() == ["xyz"] * 10

    loader = LuxonisLoader(dataset, keep_categorical_as_strings=True)
    for _, labels in loader:
        labels = {get_task_type(k): v for k, v in labels.items()}
        assert labels["metadata/color"].tolist() == ["red", "blue"] * 5

    assert dataset.get_categorical_encodings() == {
        "/metadata/color": {"red": 0, "blue": 1}
    }

    assert dataset.get_metadata_types() == {
        "/metadata/color": "Category",
        "/metadata/distance": "float",
        "/metadata/id": "int",
        "/metadata/license_plate": "str",
    }


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_no_labels(dataset_name: str, tempdir: Path, subtests: SubTests):
    def generator(total: bool) -> DatasetIterator:
        for i in range(10):
            img = create_image(i, tempdir)
            if i == 0:
                if total:
                    yield {
                        "file": img,
                    }
                else:
                    yield {
                        "file": img,
                        "annotation": {
                            "class": "person",
                            "boundingbox": {
                                "x": 0.1,
                                "y": 0.1,
                                "w": 0.1,
                                "h": 0.1,
                            },
                            "keypoints": {
                                "keypoints": [[0.1, 0.1, 0], [0.2, 0.2, 1]]
                            },
                            "segmentation": {
                                "mask": np.random.rand(512, 512) > 0.5
                            },
                            "instance_segmentation": {
                                "mask": np.random.rand(512, 512) > 0.5
                            },
                            "metadata": {
                                "color": "red",
                            },
                            "sub_detections": {
                                "head": {
                                    "boundingbox": {
                                        "x": 0.2,
                                        "y": 0.2,
                                        "w": 0.1,
                                        "h": 0.1,
                                    },
                                },
                            },
                        },
                    }

    for total in [True, False]:
        with subtests.test(f"test_{'total' if total else 'almost'}_empty"):
            dataset = create_dataset(dataset_name, generator(total=total))

            if total:
                expected_tasks = set()
            else:
                expected_tasks = {
                    "/classification",
                    "/boundingbox",
                    "/keypoints",
                    "/segmentation",
                    "/instance_segmentation",
                    "/metadata/color",
                    "/head/boundingbox",
                }

            for _, labels in LuxonisLoader(dataset):
                assert set(labels.keys()) == expected_tasks

            augmented_loader = LuxonisLoader(
                dataset,
                width=512,
                height=512,
                augmentation_config=[{"name": "Affine", "params": {}}],
            )
            for _, labels in augmented_loader:
                assert set(labels.keys()) == expected_tasks

        if total is False:
            with subtests.test("test_almost_empty_exclude_empty"):
                for i, (_, labels) in enumerate(
                    LuxonisLoader(dataset, exclude_empty_annotations=True)
                ):
                    if i == 0:
                        assert set(labels.keys()) == expected_tasks
                    else:
                        assert not labels


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_deep_nested_labels(
    dataset_name: str, augmentation_config: list[Params], tempdir: Path
):
    def generator() -> DatasetIterator:
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
                            "class": "CO",
                            "boundingbox": {
                                "x": 0.2,
                                "y": 0.2,
                                "w": 0.1,
                                "h": 0.1,
                            },
                            "metadata": {
                                "text": "ABC123",
                            },
                            "sub_detections": {
                                "text": {
                                    "boundingbox": {
                                        "x": 0.3,
                                        "y": 0.3,
                                        "w": 0.1,
                                        "h": 0.1,
                                    }
                                },
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

    dataset = create_dataset(dataset_name, generator())
    assert dataset.get_tasks() == {
        "": ["boundingbox", "classification"],
        "/driver": ["boundingbox", "keypoints"],
        "/license_plate": [
            "boundingbox",
            "classification",
            "metadata/text",
        ],
        "/license_plate/text": ["boundingbox"],
    }
    assert dataset.get_classes() == {
        "": {"car": 0},
        "/driver": {},
        "/license_plate": {"CO": 0},
        "/license_plate/text": {},
    }
    loader = LuxonisLoader(
        dataset,
        height=512,
        width=512,
        augmentation_config=augmentation_config,
        augmentation_engine="albumentations",
    )
    assert get_loader_output(loader) == {
        "/classification",
        "/boundingbox",
        "/driver/boundingbox",
        "/driver/keypoints",
        "/license_plate/boundingbox",
        "/license_plate/classification",
        "/license_plate/metadata/text",
        "/license_plate/text/boundingbox",
    }


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_partial_labels(dataset_name: str, tempdir: Path):
    def generator() -> DatasetIterator:
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

    dataset = create_dataset(dataset_name, generator())

    loader = LuxonisLoader(
        dataset,
        width=512,
        height=512,
        augmentation_config=[{"name": "Rotate"}],
    )

    assert get_loader_output(loader) == {
        "/classification",
        "/boundingbox",
        "/keypoints",
        "/segmentation",
    }


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_clone_dataset(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    def generator1() -> DatasetIterator:
        for i in range(3):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }

    dataset = create_dataset(
        dataset_name,
        generator1(),
        bucket_storage,
        splits={"train": 0.6, "val": 0.4},
    )

    cloned_dataset = dataset.clone(new_dataset_name=dataset_name + "_cloned")
    if bucket_storage == BucketStorage.GCS:  # test GCS push/pull
        local_stats = cloned_dataset.get_statistics()
        cloned_dataset = LuxonisDataset(
            cloned_dataset.dataset_name,
            bucket_storage=bucket_storage,
            delete_local=True,
            delete_remote=False,
        )
        cloned_dataset.pull_from_cloud(update_mode=UpdateMode.MISSING)
        synced_stats = cloned_dataset.get_statistics()
        assert local_stats == synced_stats

    assert cloned_dataset.get_splits() == dataset.get_splits()
    assert cloned_dataset.get_classes() == dataset.get_classes()
    assert cloned_dataset.get_task_names() == dataset.get_task_names()
    assert cloned_dataset.get_skeletons() == dataset.get_skeletons()

    df_cloned = cloned_dataset._load_df_offline()
    df_original = dataset._load_df_offline()
    assert df_cloned is not None
    assert df_original is not None
    assert df_cloned.equals(df_original)
    cloned_dataset.delete_dataset(delete_local=True, delete_remote=True)


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_merge_datasets(
    bucket_storage: BucketStorage,
    dataset_name: str,
    tempdir: Path,
    subtests: SubTests,
):
    dataset_name = f"{dataset_name}_{bucket_storage.value}"
    dataset1_name = f"{dataset_name}_1"
    dataset2_name = f"{dataset_name}_2"

    def generator1() -> DatasetIterator:
        for i in range(3):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }

    def generator2() -> DatasetIterator:
        for i in range(3, 6):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "dog",
                    "boundingbox": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2},
                },
            }

    dataset1 = create_dataset(
        dataset1_name,
        generator1(),
        bucket_storage,
        splits={"train": 0.6, "val": 0.4},
    )

    dataset2 = create_dataset(
        dataset2_name,
        generator2(),
        bucket_storage,
        splits={"train": 0.6, "val": 0.4},
    )

    with subtests.test("test_inplace"):
        cloned_dataset1 = dataset1.clone(
            new_dataset_name=f"{dataset1_name}_cloned"
        )
        cloned_dataset1_merged_with_dataset2 = cloned_dataset1.merge_with(
            dataset2, inplace=True
        )

        if bucket_storage == BucketStorage.GCS:  # test GCS push/pull
            local_stats = cloned_dataset1_merged_with_dataset2.get_statistics()
            cloned_dataset1_merged_with_dataset2 = LuxonisDataset(
                cloned_dataset1_merged_with_dataset2.dataset_name,
                bucket_storage=bucket_storage,
                delete_local=True,
                delete_remote=False,
            )
            cloned_dataset1_merged_with_dataset2.pull_from_cloud(
                update_mode=UpdateMode.MISSING
            )
            synced_stats = (
                cloned_dataset1_merged_with_dataset2.get_statistics()
            )
            synced_stats["class_distributions"][""]["boundingbox"] = sorted(
                synced_stats["class_distributions"][""]["boundingbox"],
                key=lambda x: x["class_name"],
            )
            local_stats["class_distributions"][""]["boundingbox"] = sorted(
                local_stats["class_distributions"][""]["boundingbox"],
                key=lambda x: x["class_name"],
            )
            assert local_stats == synced_stats

        cloned_dataset1_merged_with_dataset2_stats = (
            cloned_dataset1_merged_with_dataset2.get_statistics()
        )
        found = {
            (item["count"], item["class_name"])
            for item in cloned_dataset1_merged_with_dataset2_stats[
                "class_distributions"
            ][""]["boundingbox"]
        }
        assert found == {(3, "person"), (3, "dog")}

        classes = cloned_dataset1_merged_with_dataset2.get_classes()
        assert set(classes[""]) == {"person", "dog"}

    with subtests.test("test_out_of_place"):
        dataset1_merged_with_dataset2 = dataset1.merge_with(
            dataset2,
            inplace=False,
            new_dataset_name=f"{dataset1_name}_{dataset2_name}_merged",
        )

        if bucket_storage == BucketStorage.GCS:  # test GCS push/pull
            local_stats = dataset1_merged_with_dataset2.get_statistics()
            dataset1_merged_with_dataset2 = LuxonisDataset(
                dataset1_merged_with_dataset2.dataset_name,
                bucket_storage=bucket_storage,
                delete_local=True,
                delete_remote=False,
            )
            dataset1_merged_with_dataset2.pull_from_cloud(
                update_mode=UpdateMode.MISSING
            )
            synced_stats = dataset1_merged_with_dataset2.get_statistics()
            synced_stats["class_distributions"][""]["boundingbox"] = sorted(
                synced_stats["class_distributions"][""]["boundingbox"],
                key=lambda x: x["class_name"],
            )
            local_stats["class_distributions"][""]["boundingbox"] = sorted(
                local_stats["class_distributions"][""]["boundingbox"],
                key=lambda x: x["class_name"],
            )
            assert local_stats == synced_stats

    classes = dataset1_merged_with_dataset2.get_classes()
    assert set(classes[""]) == {"person", "dog"}
    dataset1_merged_with_dataset2_stats = (
        dataset1_merged_with_dataset2.get_statistics()
    )
    found = {
        (item["count"], item["class_name"])
        for item in dataset1_merged_with_dataset2_stats["class_distributions"][
            ""
        ]["boundingbox"]
    }
    assert found == {(3, "person"), (3, "dog")}

    df_merged = dataset1_merged_with_dataset2._load_df_offline()
    df_cloned_merged = dataset1.merge_with(
        dataset2, inplace=True
    )._load_df_offline()
    assert df_merged is not None
    assert df_cloned_merged is not None
    assert df_merged.equals(df_cloned_merged)

    dataset1.delete_dataset(delete_local=True, delete_remote=True)
    cloned_dataset1.delete_dataset(delete_local=True, delete_remote=True)
    dataset2.delete_dataset(delete_local=True, delete_remote=True)
    dataset1_merged_with_dataset2.delete_dataset(
        delete_local=True, delete_remote=True
    )


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_merge_datasets_specific_split(
    bucket_storage: BucketStorage,
    dataset_name: str,
    tempdir: Path,
):
    dataset_name = f"{dataset_name}_{bucket_storage.value}"
    dataset1_name = f"{dataset_name}_1"
    dataset2_name = f"{dataset_name}_2"

    def generator1() -> DatasetIterator:
        for i in range(3):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }

    def generator2() -> DatasetIterator:
        for i in range(3, 6):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "dog",
                    "boundingbox": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2},
                },
            }

    dataset1 = create_dataset(
        dataset1_name,
        generator1(),
        bucket_storage,
        splits={"train": 0.6, "val": 0.4},
    )

    dataset2 = create_dataset(
        dataset2_name,
        generator2(),
        bucket_storage,
        splits={"train": 0.6, "val": 0.4},
    )

    merged_dataset = dataset1.merge_with(
        dataset2,
        inplace=False,
        new_dataset_name=f"{dataset1_name}_{dataset2_name}_merged",
        splits_to_merge=["train"],
    )

    merged_stats = merged_dataset.get_statistics()
    assert {
        (item["count"], item["class_name"])
        for item in merged_stats["class_distributions"][""]["boundingbox"]
    } == {(3, "person"), (2, "dog")}
    merged_splits = merged_dataset.get_splits()
    dataset1_splits = dataset1.get_splits()
    dataset2_splits = dataset2.get_splits()
    assert merged_splits is not None
    assert dataset1_splits is not None
    assert dataset2_splits is not None
    assert set(merged_splits["train"]) == set(dataset1_splits["train"]) | set(
        dataset2_splits["train"]
    )
    assert set(merged_splits["val"]) == set(dataset1_splits["val"])

    dataset1.delete_dataset(delete_local=True, delete_remote=True)
    dataset2.delete_dataset(delete_local=True, delete_remote=True)
    merged_dataset.delete_dataset(delete_local=True, delete_remote=True)


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_clone_dataset_specific_split(
    bucket_storage: BucketStorage,
    dataset_name: str,
    tempdir: Path,
):
    def generator() -> DatasetIterator:
        for i in range(3):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }

    dataset = create_dataset(
        dataset_name,
        generator(),
        bucket_storage,
        splits={"train": 0.6, "val": 0.4},
    )
    cloned_dataset = dataset.clone(
        new_dataset_name=f"{dataset_name}_cloned",
        splits_to_clone=["train"],
    )
    dataset_splits = dataset.get_splits()
    cloned_splits = cloned_dataset.get_splits()
    assert cloned_splits is not None
    assert dataset_splits is not None
    assert set(cloned_splits["train"]) == set(dataset_splits["train"])
    assert "val" not in cloned_splits

    cloned_stats = cloned_dataset.get_statistics()
    assert {
        (item["count"], item["class_name"])
        for item in cloned_stats["class_distributions"][""]["boundingbox"]
    } == {(2, "person")}

    cloned_dataset.delete_dataset(delete_local=True, delete_remote=True)


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_classes_per_task(dataset_name: str, tempdir: Path):
    def generator() -> DatasetIterator:
        img = create_image(0, tempdir)
        yield {
            "file": img,
            "annotation": {
                "class": "person",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                "instance_id": 0,
            },
        }
        # Yield a second annotation with only an `instance_id` to check that we don't encounter the issue: "Detected new classes for task group '': []".
        yield {
            "file": img,
            "annotation": {
                "keypoints": {"keypoints": [[0.1, 0.1, 0], [0.2, 0.2, 1]]},
                "instance_id": 0,
            },
        }

    dataset = create_dataset(dataset_name, generator())

    assert dataset.get_classes() == {"": {"person": 0}}


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_keypoints_solo(dataset_name: str, tempdir: Path):
    def generator() -> DatasetIterator:
        for i in range(4):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "keypoints": {"keypoints": [[0.1, 0.1, 0], [0.2, 0.2, 1]]},
                },
            }

    augs = [
        {"name": "Normalize"},
        {"name": "Defocus", "params": {"p": 1}},
        {
            "name": "Mosaic4",
            "params": {"out_width": 512, "out_height": 512, "p": 1},
        },
    ]
    dataset = create_dataset(dataset_name, generator())
    loader = LuxonisLoader(
        dataset, height=512, width=512, augmentation_config=augs
    )
    for _ in loader:
        pass


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_dataset_push_pull(
    dataset_name: str, tempdir: Path, subtests: SubTests
):
    def generator(start: int, end: int) -> DatasetIterator:
        """Generate sample dataset items with bounding boxes."""
        for i in range(start, end):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                    "instance_id": i,
                },
            }

    with subtests.test("create_initial_dataset"):
        original_dataset = create_dataset(
            dataset_name,
            generator(0, 3),
            bucket_storage=BucketStorage.LOCAL,
            delete_local=True,
            delete_remote=True,
            splits=(1, 0, 0),
        )
        original_stats = original_dataset.get_statistics()

    with subtests.test("verify_dataset_overwrite"):
        overwritten_dataset = create_dataset(
            dataset_name,
            generator(0, 3),
            bucket_storage=BucketStorage.LOCAL,
            delete_local=False,
            delete_remote=False,
            splits=False,
        )
        assert overwritten_dataset.get_statistics() == original_stats

    with subtests.test("push_to_cloud"):
        overwritten_dataset.push_to_cloud(bucket_storage=BucketStorage.GCS)

        overwritten_dataset.delete_dataset(
            delete_local=True, delete_remote=False
        )
        del overwritten_dataset

        assert not LuxonisDataset.exists(
            dataset_name, bucket_storage=BucketStorage.LOCAL
        )
        assert LuxonisDataset.exists(
            dataset_name, bucket_storage=BucketStorage.GCS
        )

    with subtests.test("pull_from_cloud_local_media_empty"):
        cloud_dataset = LuxonisDataset(
            dataset_name,
            bucket_storage=BucketStorage.GCS,
            delete_local=True,
            delete_remote=False,
        )

        cloud_dataset.pull_from_cloud(update_mode=UpdateMode.MISSING)

        assert cloud_dataset.get_statistics() == original_stats
        assert sum(1 for _ in LuxonisLoader(cloud_dataset)) == 3

    with subtests.test("pull_from_cloud_local_media_full"):
        cloud_dataset.delete_dataset(delete_local=True, delete_remote=False)
        shutil.rmtree(tempdir)
        del cloud_dataset

        cloud_dataset_again = LuxonisDataset(
            dataset_name,
            bucket_storage=BucketStorage.GCS,
            delete_local=True,
            delete_remote=False,
        )

        cloud_dataset_again.pull_from_cloud(update_mode=UpdateMode.MISSING)

        assert cloud_dataset_again.get_statistics() == original_stats
        assert sum(1 for _ in LuxonisLoader(cloud_dataset_again)) == 3

        cloud_dataset_again.delete_dataset(
            delete_local=False, delete_remote=True
        )

    with subtests.test("push_to_cloud_local_media_full_loader_resync"):
        local_dataset = LuxonisDataset(
            dataset_name,
            bucket_storage=BucketStorage.LOCAL,
            delete_local=False,
        )
        local_dataset.push_to_cloud(bucket_storage=BucketStorage.GCS)
        assert cloud_dataset_again.get_statistics() == original_stats

        cloud_dataset = LuxonisDataset(
            dataset_name,
            bucket_storage=BucketStorage.GCS,
            delete_local=True,
            delete_remote=False,
        )

        loader = LuxonisLoader(cloud_dataset)
        assert sum(1 for _ in loader) == 3
        assert cloud_dataset.get_statistics() == original_stats


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_merge_on_different_machines(dataset_name: str, tempdir: Path):
    def generator(start: int, end: int) -> DatasetIterator:
        """Generate sample dataset items with bounding boxes."""
        for i in range(start, end):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                    "instance_id": i,
                },
            }

    dataset1 = create_dataset(
        dataset_name + "_1",
        generator(0, 3),
        bucket_storage=BucketStorage.GCS,
        delete_local=True,
        delete_remote=True,
        splits=(1, 0, 0),
    )
    dataset2 = create_dataset(
        dataset_name + "_2",
        generator(3, 6),
        bucket_storage=BucketStorage.GCS,
        delete_local=True,
        delete_remote=True,
        splits=(1, 0, 0),
    )
    shutil.rmtree(tempdir)
    dataset1.pull_from_cloud()
    dataset2.pull_from_cloud()
    dataset1.delete_dataset(delete_remote=True)
    dataset2.delete_dataset(delete_remote=True)
    dataset1 = LuxonisDataset(dataset_name + "_1")
    dataset2 = LuxonisDataset(dataset_name + "_2")
    assert len(list(dataset1.media_path.glob("*"))) == 3
    assert len(list(dataset2.media_path.glob("*"))) == 3
    dataset3 = dataset1.merge_with(
        dataset2, inplace=False, new_dataset_name=dataset_name
    )
    loader = LuxonisLoader(dataset3)
    assert sum(1 for _ in loader) == 6
    dataset3.export(tempdir)
    assert (
        len(
            list(
                Path.cwd().glob(
                    f"{tempdir}/{dataset3.dataset_name}/train/images/*"
                )
            )
        )
        == 6
    )
