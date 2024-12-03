import json
import shutil
import uuid
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
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import LuxonisFileSystem

# TODO: Test array

SKELETONS: Final[dict] = {
    "detection": (
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
    "segmentation/segmentation",
    "detection/keypoints",
    "detection/boundingbox",
}
DATA_DIR = Path("tests/data/test_dataset")

AUG_CONFIG = [
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
        assert set(dataset.get_tasks()) == {"detection", "segmentation"}
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
                assert task in labels

    with subtests.test("test_load_aug", bucket_storage=bucket_storage):
        augmentations = Augmentations.from_config(512, 512, AUG_CONFIG)
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

    loader._load_image_with_annotations = _raise  # type: ignore
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


def test_complex_dataset():
    url = f"{URL_PREFIX}/D2_ParkingLot"
    base_path = LuxonisFileSystem.download(url, WORK_DIR)

    def get_annotations(sequence_path):
        frame_data = sequence_path / "step0.frame_data.json"
        with open(frame_data) as f:
            data = json.load(f)["captures"][0]
            frame_data = data["annotations"]

        return {anno["@type"].split(".")[-1]: anno for anno in frame_data}

    # TODO: simplify
    def generator():
        seen = set()
        for sequence_path in base_path.glob("sequence.*"):
            filepath = sequence_path / "step0.camera.jpg"
            if not filepath.exists():
                filepath = sequence_path / "step0.camera_0.jpg"

            file_hash_uuid = str(
                uuid.uuid5(uuid.NAMESPACE_URL, filepath.read_bytes().hex())
            )
            if file_hash_uuid in seen:
                continue
            seen.add(file_hash_uuid)
            annotations = get_annotations(sequence_path)
            W, H = annotations["SemanticSegmentationAnnotation"]["dimension"]
            bbox = annotations["BoundingBox2DAnnotation"]["values"][0]
            instance_id = bbox["instanceId"]
            x, y = bbox["origin"]
            w, h = bbox["dimension"]
            label_name: str = bbox["labelName"]
            *brand, color, vehicle_type = label_name.split("-")
            vehicle_type = vehicle_type.lower()
            if vehicle_type == "motorbiek":
                vehicle_type = "motorbike"

            keypoints = []
            for kp in annotations["KeypointAnnotation"]["values"][0][
                "keypoints"
            ]:
                kpt_x, kpt_y = kp["location"]
                state = kp["state"]
                if vehicle_type == "motorbike":
                    state = 2
                keypoints.append([kpt_x / W, kpt_y / H, state])

            mask_path = annotations["SemanticSegmentationAnnotation"][
                "filename"
            ]
            mask = cv2.imread(
                str(sequence_path / mask_path), cv2.IMREAD_GRAYSCALE
            ).astype(bool)

            yield {
                "file": filepath,
                "task": vehicle_type,
                "annotation": {
                    "instance_id": instance_id,
                    "class": vehicle_type,
                    "boundingbox": {
                        "x": x / W,
                        "y": y / H,
                        "w": w / W,
                        "h": h / H,
                    },
                    "keypoints": {
                        "keypoints": keypoints,
                    },
                    "metadata": {
                        "color": color.lower(),
                        "brand": "-".join(brand),
                    },
                    "segmentation": {
                        "mask": mask,
                    },
                },
            }

    dataset = LuxonisDataset("__D2ParkingSLot-test", delete_existing=True)
    dataset.add(generator())
    dataset.make_splits()
    assert len(dataset) == 156
    assert set(dataset.get_tasks()) == {"motorbike", "car"}
    assert dataset.get_classes()[1] == {
        "motorbike": ["motorbike"],
        "car": ["car"],
    }
    loader = LuxonisLoader(dataset)
    _, labels = next(iter(loader))
    labels = {
        k.replace("car/", "").replace("motorbike/", ""): v
        for k, v in labels.items()
    }
    assert "boundingbox" in labels
    assert "segmentation" in labels
    assert "keypoints" in labels
    assert "metadata/color" in labels
    assert "metadata/brand" in labels


# TODO: Test array


@pytest.mark.parametrize(
    ("bucket_storage",),
    [
        (BucketStorage.LOCAL,),
        (BucketStorage.GCS,),
    ],
)
def test_metadata(
    bucket_storage: BucketStorage, platform_name: str, python_version: str
):
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
        assert "detection/metadata/color" in labels
        assert "detection/metadata/distance" in labels
        assert "detection/metadata/id" in labels

        assert (
            labels["detection/metadata/color"].tolist() == ["red", "blue"] * 5
        )
        assert labels["detection/metadata/distance"].tolist() == [5.0] * 10
        assert labels["detection/metadata/id"].tolist() == list(
            range(127, 137)
        )


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

    augments = Augmentations.from_config(
        512, 512, [{"name": "Flip", "params": {}}]
    )
    loader = LuxonisLoader(dataset, augmentations=augments)
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

    augmentations = Augmentations.from_config(512, 512, AUG_CONFIG)
    loader = LuxonisLoader(dataset, augmentations=augmentations)
    _, labels = next(iter(loader))
    assert {
        "detection/classification",
        "detection/boundingbox",
        "detection/driver/boundingbox",
        "detection/driver/keypoints",
        "detection/license_plate/boundingbox",
        "detection/license_plate/metadata/text",
    } == set(labels.keys())
