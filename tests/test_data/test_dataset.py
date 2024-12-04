import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Final, List, Set

import cv2
import numpy as np
import pytest

from luxonis_ml.data import (
    Augmentations,
    BucketStorage,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
    LuxonisSource,
)
from luxonis_ml.data.utils.data_utils import rgb_to_bool_masks
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import LuxonisFileSystem

SKELETONS: Final[dict] = {
    "keypoints": (
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
DATASET_NAME: Final[str] = "test-dataset"
TASKS: Final[Set[str]] = {
    "segmentation",
    "classification",
    "keypoints",
    "boundingbox",
}
DATA_DIR = Path("tests/data/test_dataset")


@pytest.fixture(autouse=True, scope="module")
def prepare_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    yield

    shutil.rmtree(DATA_DIR)


def make_image(i) -> Path:
    path = DATA_DIR / f"img_{i}.jpg"
    if not path.exists():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[0:10, 0:10] = np.random.randint(
            0, 255, (10, 10, 3), dtype=np.uint8
        )
        cv2.imwrite(str(path), img)
    return path


@pytest.mark.parametrize(
    ("bucket_storage",),
    [
        (BucketStorage.LOCAL,),
        (BucketStorage.S3,),
        (BucketStorage.GCS,),
    ],
)
def test_dataset(
    bucket_storage: BucketStorage,
    platform_name: str,
    python_version: str,
    subtests,
):
    dataset_name = f"{DATASET_NAME}-{bucket_storage.value}-{platform_name}-{python_version}"
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
        assert set(dataset.get_tasks()) == TASKS
        assert dataset.get_skeletons() == SKELETONS
        assert dataset.identifier == dataset_name

    if "dataset" not in locals():
        pytest.exit("Dataset creation failed")

    with subtests.test("test_source"):
        print(dataset.source.to_document())
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
                    "type": "classification",
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

    loader._load_image_with_annotations = _raise
    with pytest.raises(IndexError):
        _ = loader[0]


@pytest.mark.parametrize(
    ("bucket_storage",),
    [
        (BucketStorage.LOCAL,),
        (BucketStorage.GCS,),
    ],
)
def test_make_splits(
    bucket_storage: BucketStorage, platform_name: str, python_version: str
):
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
                    "type": "classification",
                    "class": ["dog", "cat"][i % 2],
                },
            }
            definitions[["train", "val", "test"][i % 3]].append(str(path))
        _start_index += step

    dataset = LuxonisDataset(
        f"_test_split-{bucket_storage.value}-{platform_name}-{python_version}",
        delete_existing=True,
        delete_remote=True,
        bucket_storage=bucket_storage,
    )
    dataset.add(generator())
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
    dataset.make_splits(definitions)
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
        dataset.make_splits((0.7, 0.1, 1), definitions=definitions)

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


def test_complex_dataset():
    url = f"{URL_PREFIX}/D1_ParkingSlotTest"
    base_path = LuxonisFileSystem.download(url, WORK_DIR)
    mask_brand_path = base_path / "mask_brand"
    mask_color_path = base_path / "mask_color"
    kpt_mask_path = base_path / "keypoints_mask_vehicle"

    def generator():
        filenames: Dict[int, Path] = {}
        for base_path in [kpt_mask_path, mask_brand_path, mask_color_path]:
            for sequence_path in list(sorted(base_path.glob("sequence.*"))):
                frame_data = sequence_path / "step0.frame_data.json"
                with open(frame_data) as f:
                    data = json.load(f)["captures"][0]
                    frame_data = data["annotations"]
                    sequence_num = int(sequence_path.suffix[1:])
                    filename = data["filename"]
                    if filename is not None:
                        filename = sequence_path / filename
                        filenames[sequence_num] = filename
                    else:
                        filename = filenames[sequence_num]
                    W, H = data["dimension"]

                annotations = {
                    anno["@type"].split(".")[-1]: anno for anno in frame_data
                }

                bbox_classes = {}
                bboxes = {}

                for bbox_annotation in annotations.get(
                    "BoundingBox2DAnnotation", defaultdict(list)
                )["values"]:
                    class_ = (
                        bbox_annotation["labelName"].split("-")[-1].lower()
                    )
                    if class_ == "motorbiek":
                        class_ = "motorbike"
                    x, y = bbox_annotation["origin"]
                    w, h = bbox_annotation["dimension"]
                    instance_id = bbox_annotation["instanceId"]
                    bbox_classes[instance_id] = class_
                    bboxes[instance_id] = [x / W, y / H, w / W, h / H]
                    yield {
                        "file": filename,
                        "annotation": {
                            "type": "boundingbox",
                            "class": class_,
                            "x": x / W,
                            "y": y / H,
                            "w": w / W,
                            "h": h / H,
                            "instance_id": instance_id,
                        },
                    }

                for kpt_annotation in annotations.get(
                    "KeypointAnnotation", defaultdict(list)
                )["values"]:
                    keypoints = kpt_annotation["keypoints"]
                    instance_id = kpt_annotation["instanceId"]
                    class_ = bbox_classes[instance_id]
                    bbox = bboxes[instance_id]
                    kpts = []

                    if class_ == "motorbike":
                        keypoints = keypoints[:3]
                    else:
                        keypoints = keypoints[3:]

                    for kp in keypoints:
                        x, y = kp["location"]
                        kpts.append([x / W, y / H, kp["state"]])

                    yield {
                        "file": filename,
                        "annotation": {
                            "type": "detection",
                            "class": class_,
                            "task": class_,
                            "keypoints": kpts,
                            "instance_id": instance_id,
                            "boundingbox": {
                                "x": bbox[0],
                                "y": bbox[1],
                                "w": bbox[2],
                                "h": bbox[3],
                            },
                        },
                    }

                vehicle_type_segmentation = annotations[
                    "SemanticSegmentationAnnotation"
                ]
                mask = cv2.cvtColor(
                    cv2.imread(
                        str(
                            sequence_path
                            / vehicle_type_segmentation["filename"]
                        )
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                classes = {
                    inst["labelName"]: inst["pixelValue"][:3]
                    for inst in vehicle_type_segmentation["instances"]
                }
                if base_path == kpt_mask_path:
                    task = "vehicle_type_segmentation"
                elif base_path == mask_brand_path:
                    task = "brand_segmentation"
                else:
                    task = "color_segmentation"
                for class_, mask_ in rgb_to_bool_masks(
                    mask, classes, add_background_class=True
                ):
                    yield {
                        "file": filename,
                        "annotation": {
                            "type": "mask",
                            "class": class_,
                            "task": task,
                            "mask": mask_,
                        },
                    }
                if base_path == mask_color_path:
                    yield {
                        "file": filename,
                        "annotation": {
                            "type": "mask",
                            "class": "vehicle",
                            "task": "vehicle_segmentation",
                            "mask": mask.astype(bool)[..., 0]
                            | mask.astype(bool)[..., 1]
                            | mask.astype(bool)[..., 2],
                        },
                    }

    dataset = LuxonisDataset("__D1ParkingSLot-test", delete_existing=True)
    dataset.add(generator())
    dataset.make_splits()
    assert len(dataset) == 200
    assert set(dataset.get_tasks()) == {
        "boundingbox",
        "motorbike-boundingbox",
        "motorbike-keypoints",
        "car-boundingbox",
        "car-keypoints",
        "vehicle_type_segmentation",
        "brand_segmentation",
        "color_segmentation",
        "vehicle_segmentation",
    }
    assert dataset.get_classes()[1] == {
        "boundingbox": sorted(["motorbike", "car"]),
        "motorbike-boundingbox": ["motorbike"],
        "motorbike-keypoints": ["motorbike"],
        "car-boundingbox": ["car"],
        "car-keypoints": ["car"],
        "vehicle_type_segmentation": ["background", "car", "motorbike"],
        "brand_segmentation": sorted(
            [
                "background",
                "chrysler",
                "bmw",
                "ducati",
                "dodge",
                "ferrari",
                "infiniti",
                "land-rover",
                "roll-royce",
                "saab",
                "Kawasaki",
                "moto",
                "truimph",
                "alfa-romeo",
                "harley",
                "honda",
                "jeep",
                "aprilia",
                "piaggio",
                "yamaha",
                "buick",
                "pontiac",
                "isuzu",
            ]
        ),
        "color_segmentation": ["background", "blue", "green", "red"],
        "vehicle_segmentation": ["vehicle"],
    }


@pytest.mark.parametrize(
    ("bucket_storage",),
    [
        (BucketStorage.LOCAL,),
        (BucketStorage.GCS,),
    ],
)
def test_uncommon_label_types(
    bucket_storage: BucketStorage, platform_name: str, python_version: str
):
    arr = np.random.rand(10)

    def generator():
        for i in range(10):
            img = make_image(i)
            yield {
                "file": img,
                "annotation": {
                    "type": "label",
                    "value": "dog",
                },
            }
            yield {
                "file": img,
                "annotation": {
                    "type": "label",
                    "value": "cat",
                },
            }
            np.save(str(img.with_suffix(".npy")), arr)
            yield {
                "file": img,
                "annotation": {
                    "type": "array",
                    "path": str(img.with_suffix(".npy")),
                },
            }

    dataset_name = f"__uncommon_label_types-{bucket_storage.value}-{platform_name}-{python_version}"
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
        assert "label" in labels
        assert "array" in labels
        assert labels["label"][0].tolist() == [["dog"], ["cat"]]
        assert np.allclose(labels["array"][0], arr)


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

    augments = Augmentations([512, 512], [{"name": "Flip", "params": {}}])
    loader = LuxonisLoader(dataset, augmentations=augments)
    for _, labels in loader:
        assert labels == {}


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
                        "type": "classification",
                        "class": "dog",
                    },
                }
            elif i < 6:
                yield {
                    "file": img,
                    "annotation": {
                        "type": "boundingbox",
                        "class": "dog",
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.1,
                        "h": 0.1,
                    },
                }
                yield {
                    "file": img,
                    "annotation": {
                        "type": "keypoints",
                        "class": "dog",
                        "keypoints": [[0.1, 0.1, 0], [0.2, 0.2, 1]],
                    },
                }
            elif i < 8:
                yield {
                    "file": img,
                    "annotation": {
                        "type": "mask",
                        "class": "dog",
                        "mask": np.random.rand(512, 512) > 0.5,
                    },
                }

    dataset.add(generator())
    dataset.make_splits([1, 0, 0])

    augments = Augmentations([512, 512], [{"name": "Rotate", "params": {}}])
    loader = LuxonisLoader(dataset, augmentations=augments, view="train")
    for _, labels in loader:
        assert labels.get("boundingbox") is not None
        assert labels.get("classification") is not None
        assert labels.get("segmentation") is not None
        assert labels.get("keypoints") is not None
