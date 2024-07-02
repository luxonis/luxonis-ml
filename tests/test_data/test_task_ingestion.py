import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytest

from luxonis_ml.data import BucketStorage, LuxonisDataset, LuxonisLoader

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
        img[0:10, 0:10] = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)
    return path


def compute_histogram(dataset: LuxonisDataset) -> Dict[str, int]:
    classes = defaultdict(int)
    loader = LuxonisLoader(dataset)
    for _, record in loader:
        for task, _ in record.items():
            classes[task] += 1

    return dict(classes)


@pytest.mark.parametrize(
    ("bucket_storage",),
    [
        (BucketStorage.LOCAL,),
        (BucketStorage.S3,),
        (BucketStorage.GCS,),
    ],
)
def test_task_ingestion(bucket_storage: BucketStorage):
    dataset = LuxonisDataset.delete_and_create(
        DATASET_NAME,
        bucket_storage=bucket_storage,
        remote=bucket_storage != BucketStorage.LOCAL,
    )

    def generator1():
        for i in range(STEP):
            path = make_image(i)
            yield {
                "file": str(path),
                "annotation": {
                    "type": "boundingbox",
                    "class": "dog",
                    "task": "animals-boxes",
                    "x": 0.1,
                    "y": 0.1,
                    "w": 0.1,
                    "h": 0.1,
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "type": "boundingbox",
                    "class": "cat",
                    "task": "animals-boxes",
                    "x": 0.5,
                    "y": 0.5,
                    "w": 0.1,
                    "h": 0.3,
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "type": "polyline",
                    "class": "water",
                    "task": "land-segmentation",
                    "points": [
                        (0.1, 0.1),
                        (0.2, 0.8),
                        (0.8, 0.3),
                        (0, 0.5),
                        (0.5, 0.5),
                    ],
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "type": "polyline",
                    "class": "grass",
                    "task": "land-segmentation",
                    "points": [(0.1, 0.5), (0.6, 0.6), (0.7, 0.7)],
                },
            }

    dataset.add(generator1()).make_splits(ratios=(1, 0, 0))

    classes_list, classes = dataset.get_classes()

    assert set(classes_list) == {"dog", "cat", "water", "grass"}
    assert set(classes["land-segmentation"]) == {"water", "grass"}
    assert set(classes["animals-boxes"]) == {"dog", "cat"}

    assert compute_histogram(dataset) == {
        "animals-boxes": STEP,
        "land-segmentation": STEP,
    }

    def generator2():
        for i in range(STEP, 2 * STEP):
            path = make_image(i)
            yield {
                "file": str(path),
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
                "file": str(path),
                "annotation": {
                    "type": "boundingbox",
                    "class": "cat",
                    "x": 0.5,
                    "y": 0.5,
                    "w": 0.1,
                    "h": 0.3,
                },
            }

    dataset.add(generator2()).make_splits(ratios=(1, 0, 0))
    classes_list, classes = dataset.get_classes()

    assert set(classes_list) == {"dog", "cat", "water", "grass"}
    assert set(classes["land-segmentation"]) == {"water", "grass"}
    assert set(classes["animals-boxes"]) == {"dog", "cat"}

    assert compute_histogram(dataset) == {
        "animals-boxes": 2 * STEP,
        "land-segmentation": STEP,
    }

    def generator3():
        for i in range(2 * STEP, 3 * STEP):
            path = make_image(i)
            yield {
                "file": str(path),
                "annotation": {
                    "type": "boundingbox",
                    "task": "animals-boxes",
                    "class": "dog",
                    "x": 0.15,
                    "y": 0.25,
                    "w": 0.1,
                    "h": 0.1,
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "type": "polyline",
                    "class": "water",
                    "points": [(0.1, 0.7), (0.5, 0.2), (0.3, 0.3), (0.12, 0.45)],
                },
            }

    dataset.add(generator3()).make_splits(ratios=(1, 0, 0))
    classes_list, classes = dataset.get_classes()

    assert set(classes_list) == {"dog", "cat", "water", "grass"}
    assert set(classes["land-segmentation"]) == {"water", "grass"}
    assert set(classes["animals-boxes"]) == {"dog", "cat"}

    assert compute_histogram(dataset) == {
        "animals-boxes": 3 * STEP,
        "land-segmentation": 2 * STEP,
    }

    def generator4():
        for i in range(3 * STEP, 4 * STEP):
            path = make_image(i)
            yield {
                "file": str(path),
                "annotation": {
                    "type": "boundingbox",
                    "class": "bike",
                    "x": 0.9,
                    "y": 0.8,
                    "w": 0.1,
                    "h": 0.4,
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "type": "polyline",
                    "class": "body",
                    "points": [(0.1, 0.1), (0.7, 0.5), (0.3, 0.3), (0.5, 0.5)],
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "type": "polyline",
                    "class": "water",
                    "task": "land-segmentation-2",
                    "points": [(0.1, 0.1), (0.8, 0.2), (0.8, 0.9), (0.1, 0.9)],
                },
            }

    dataset.add(generator4()).make_splits(ratios=(1, 0, 0))
    classes_list, classes = dataset.get_classes()

    assert set(classes_list) == {"dog", "cat", "water", "grass", "bike", "body"}
    assert set(classes["land-segmentation"]) == {"water", "grass"}
    assert set(classes["animals-boxes"]) == {"dog", "cat"}
    assert set(classes["land-segmentation-2"]) == {"water"}
    assert set(classes["segmentation"]) == {"body"}

    assert compute_histogram(dataset) == {
        "animals-boxes": 3 * STEP,
        "land-segmentation": 2 * STEP,
        "land-segmentation-2": STEP,
        "segmentation": STEP,
        "boundingbox": STEP,
    }
