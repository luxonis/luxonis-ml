from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NoReturn

import numpy as np
import pytest
from pytest_subtests.plugin import SubTests

from luxonis_ml.data import (
    BucketStorage,
    Category,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
    LuxonisSource,
)
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.utils.task_utils import get_task_type
from luxonis_ml.enums import DatasetType
from luxonis_ml.typing import Params

from .utils import create_dataset, create_image, get_loader_output


def test_dataset(
    bucket_storage: BucketStorage,
    dataset_name: str,
    augmentation_config: List[Params],
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
            delete_existing=True,
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
        assert dataset.source == LuxonisSource()
        dataset.update_source(LuxonisSource(name="test"))
        assert dataset.source == LuxonisSource(name="test")

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
        dataset.delete_dataset(delete_remote=True)
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

    def _raise(*_) -> NoReturn:
        raise IndexError

    loader._load_data = _raise  # type: ignore
    with pytest.raises(IndexError):
        _ = loader[0]


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_make_splits(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    definitions: Dict[str, List[str]] = defaultdict(list)

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

    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        dataset.make_splits((0.7, 0.1, 1))

    with pytest.raises(ValueError, match="must be a tuple of 3 floats"):
        dataset.make_splits((0.7, 0.1, 0.1, 0.1))  # type: ignore

    with pytest.raises(ValueError, match="Cannot provide both splits and"):
        dataset.make_splits((0.7, 0.1, 0.2), definitions=definitions)

    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        dataset.make_splits({"train": 1.5})

    with pytest.raises(ValueError, match="Dataset size is smaller than"):
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
                augmentation_config=[{"name": "Flip", "params": {}}],
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
    dataset_name: str, augmentation_config: List[Params], tempdir: Path
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

    assert cloned_dataset.get_splits() == dataset.get_splits()
    assert cloned_dataset.get_classes() == dataset.get_classes()
    assert cloned_dataset.get_task_names() == dataset.get_task_names()
    assert cloned_dataset.get_skeletons() == dataset.get_skeletons()

    df_cloned = cloned_dataset._load_df_offline()
    df_original = dataset._load_df_offline()
    assert df_cloned is not None
    assert df_original is not None
    assert df_cloned.equals(df_original)


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
