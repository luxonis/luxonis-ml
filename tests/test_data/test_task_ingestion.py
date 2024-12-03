import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytest

from luxonis_ml.data import BucketStorage, LuxonisDataset, LuxonisLoader
from luxonis_ml.data.utils import get_task_name, get_task_type

DATASET_NAME = "test-task-ingestion"
DATA_DIR = Path("tests/data/test_task_ingestion")
STEP = 10


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


def compute_histogram(dataset: LuxonisDataset) -> Dict[str, int]:
    classes = defaultdict(int)
    loader = LuxonisLoader(dataset, force_resync=True)
    for _, record in loader:
        for task, _ in record.items():
            if get_task_type(task) != "classification":
                classes[get_task_name(task)] += 1

    return dict(classes)


@pytest.mark.parametrize(
    ("bucket_storage",),
    [
        (BucketStorage.LOCAL,),
        (BucketStorage.S3,),
        (BucketStorage.GCS,),
    ],
)
def test_task_ingestion(
    bucket_storage: BucketStorage, platform_name: str, python_version: str
):
    dataset = LuxonisDataset(
        f"{DATASET_NAME}-{bucket_storage.value}-{platform_name}-{python_version}",
        bucket_storage=bucket_storage,
        delete_existing=True,
        delete_remote=True,
    )

    def generator1():
        for i in range(STEP):
            path = make_image(i)
            yield {
                "file": str(path),
                "task": "animals",
                "annotation": {
                    "class": "dog",
                    "boundingbox": {
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.1,
                        "h": 0.1,
                    },
                },
            }
            yield {
                "file": str(path),
                "task": "animals",
                "annotation": {
                    "class": "cat",
                    "boundingbox": {
                        "x": 0.5,
                        "y": 0.5,
                        "w": 0.1,
                        "h": 0.3,
                    },
                },
            }
            yield {
                "file": str(path),
                "task": "landmass",
                "annotation": {
                    "class": "water",
                    "segmentation": {
                        "points": [
                            (0.1, 0.1),
                            (0.2, 0.8),
                            (0.8, 0.3),
                            (0, 0.5),
                            (0.5, 0.5),
                        ],
                        "width": 512,
                        "height": 512,
                    },
                },
            }
            yield {
                "file": str(path),
                "task": "landmass",
                "annotation": {
                    "class": "grass",
                    "segmentation": {
                        "points": [(0.1, 0.5), (0.6, 0.6), (0.7, 0.7)],
                        "width": 512,
                        "height": 512,
                    },
                },
            }

    dataset.add(generator1()).make_splits((1, 0, 0))

    classes_list, classes = dataset.get_classes()

    assert set(classes_list) == {"dog", "cat", "water", "grass"}
    assert set(classes["landmass"]) == {"water", "grass"}
    assert set(classes["animals"]) == {"dog", "cat"}

    assert compute_histogram(dataset) == {"animals": STEP, "landmass": STEP}

    def generator2():
        for i in range(STEP, 2 * STEP):
            path = make_image(i)
            yield {
                "file": str(path),
                "annotation": {
                    "class": "dog",
                    "boundingbox": {
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.1,
                        "h": 0.1,
                    },
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "class": "cat",
                    "boundingbox": {
                        "x": 0.5,
                        "y": 0.5,
                        "w": 0.1,
                        "h": 0.3,
                    },
                },
            }

    dataset.add(generator2()).make_splits((1, 0, 0))
    classes_list, classes = dataset.get_classes()

    assert set(classes_list) == {"background", "dog", "cat", "water", "grass"}
    assert set(classes["landmass"]) == {"background", "water", "grass"}
    assert set(classes["animals"]) == {"dog", "cat"}

    assert compute_histogram(dataset) == {
        "animals": 2 * STEP,
        "landmass": STEP,
    }

    def generator3():
        for i in range(2 * STEP, 3 * STEP):
            path = make_image(i)
            yield {
                "file": str(path),
                "task": "animals",
                "annotation": {
                    "class": "dog",
                    "boundingbox": {
                        "x": 0.15,
                        "y": 0.25,
                        "w": 0.1,
                        "h": 0.1,
                    },
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "class": "water",
                    "segmentation": {
                        "points": [
                            (0.1, 0.7),
                            (0.5, 0.2),
                            (0.3, 0.3),
                            (0.12, 0.45),
                        ],
                        "width": 512,
                        "height": 512,
                    },
                },
            }

    dataset.add(generator3()).make_splits((1, 0, 0))
    classes_list, classes = dataset.get_classes()

    assert set(classes_list) == {"background", "dog", "cat", "water", "grass"}
    assert set(classes["landmass"]) == {"background", "water", "grass"}
    assert set(classes["animals"]) == {"dog", "cat"}

    assert compute_histogram(dataset) == {
        "animals": 3 * STEP,
        "landmass": 2 * STEP,
    }

    def generator4():
        for i in range(3 * STEP, 4 * STEP):
            path = make_image(i)
            yield {
                "file": str(path),
                "annotation": {
                    "class": "bike",
                    "boundingbox": {
                        "x": 0.9,
                        "y": 0.8,
                        "w": 0.1,
                        "h": 0.4,
                    },
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "class": "body",
                    "segmentation": {
                        "points": [
                            (0.1, 0.1),
                            (0.7, 0.5),
                            (0.3, 0.3),
                            (0.5, 0.5),
                        ],
                        "width": 512,
                        "height": 512,
                    },
                },
            }
            yield {
                "file": str(path),
                "task": "landmass-2",
                "annotation": {
                    "class": "water",
                    "segmentation": {
                        "points": [
                            (0.1, 0.1),
                            (0.8, 0.2),
                            (0.8, 0.9),
                            (0.1, 0.9),
                        ],
                        "width": 512,
                        "height": 512,
                    },
                },
            }

    dataset.add(generator4()).make_splits((1, 0, 0))
    classes_list, classes = dataset.get_classes()

    print(classes)
    assert set(classes_list) == {
        "dog",
        "cat",
        "water",
        "grass",
        "bike",
        "body",
        "background",
    }
    assert set(classes["landmass"]) == {"background", "water", "grass"}
    assert set(classes["animals"]) == {"dog", "cat"}
    assert set(classes["landmass-2"]) == {"water"}
    assert set(classes["detection"]) == {"bike"}
    assert set(classes["segmentation"]) == {"body"}

    assert compute_histogram(dataset) == {
        "animals": 3 * STEP,
        "landmass": 2 * STEP,
        "landmass-2": STEP,
        "detection": STEP,
        "segmentation": STEP,
    }
