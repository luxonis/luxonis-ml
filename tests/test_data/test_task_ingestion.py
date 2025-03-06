from collections import defaultdict
from pathlib import Path
from typing import Dict

from luxonis_ml.data import (
    BucketStorage,
    LuxonisDataset,
    LuxonisLoader,
    UpdateMode,
)
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.utils import get_task_name, get_task_type

from .utils import create_image

STEP = 10


def compute_histogram(dataset: LuxonisDataset) -> Dict[str, int]:
    classes = defaultdict(int)
    loader = LuxonisLoader(
        dataset, exclude_empty_annotations=True, update_mode=UpdateMode.ALWAYS
    )
    for _, record in loader:
        for task in record:
            if get_task_type(task) != "classification":
                classes[get_task_name(task)] += 1

    return dict(classes)


def test_task_ingestion(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    dataset = LuxonisDataset(
        dataset_name,
        bucket_storage=bucket_storage,
        delete_existing=True,
        delete_remote=True,
    )

    def generator1() -> DatasetIterator:
        for i in range(STEP):
            path = create_image(i, tempdir)
            yield {
                "file": str(path),
                "task_name": "animals",
                "annotation": {
                    "class": "dog",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }
            yield {
                "file": str(path),
                "task_name": "animals",
                "annotation": {
                    "class": "cat",
                    "boundingbox": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.3},
                },
            }
            yield {
                "file": str(path),
                "task_name": "landmass",
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
                "task_name": "landmass",
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

    classes = dataset.get_classes()

    assert set(classes["landmass"]) == {"water", "grass"}
    assert set(classes["animals"]) == {"dog", "cat"}

    assert compute_histogram(dataset) == {"animals": STEP, "landmass": STEP}

    def generator2() -> DatasetIterator:
        for i in range(STEP, 2 * STEP):
            path = create_image(i, tempdir)
            yield {
                "file": str(path),
                "annotation": {
                    "class": "dog",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }
            yield {
                "file": str(path),
                "annotation": {
                    "class": "cat",
                    "boundingbox": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.3},
                },
            }

    dataset.add(generator2()).make_splits((1, 0, 0))
    classes = dataset.get_classes()
    assert set(classes["landmass"]) == {"background", "water", "grass"}
    assert set(classes["animals"]) == {"dog", "cat"}

    assert compute_histogram(dataset) == {
        "animals": 2 * STEP,
        "landmass": STEP,
    }

    def generator3() -> DatasetIterator:
        for i in range(2 * STEP, 3 * STEP):
            path = create_image(i, tempdir)
            yield {
                "file": str(path),
                "task_name": "animals",
                "annotation": {
                    "class": "dog",
                    "boundingbox": {"x": 0.15, "y": 0.25, "w": 0.1, "h": 0.1},
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
    classes = dataset.get_classes()
    assert set(classes["landmass"]) == {"background", "water", "grass"}
    assert set(classes["animals"]) == {"dog", "cat"}

    assert compute_histogram(dataset) == {
        "animals": 3 * STEP,
        "landmass": 2 * STEP,
    }

    def generator4() -> DatasetIterator:
        for i in range(3 * STEP, 4 * STEP):
            path = create_image(i, tempdir)
            yield {
                "file": str(path),
                "task_name": "detection",
                "annotation": {
                    "class": "bike",
                    "boundingbox": {"x": 0.9, "y": 0.8, "w": 0.1, "h": 0.4},
                },
            }
            yield {
                "file": str(path),
                "task_name": "segmentation",
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
                "task_name": "landmass-2",
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
    classes = dataset.get_classes()

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
