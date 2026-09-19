from collections import defaultdict
from pathlib import Path

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


def compute_histogram(dataset: LuxonisDataset) -> dict[str, int]:
    classes = defaultdict(int)
    loader = LuxonisLoader(
        dataset, exclude_empty_annotations=True, update_mode=UpdateMode.ALL
    )
    for _, record in loader:
        for task in record:
            if get_task_type(task) != "classification":
                classes[get_task_name(task)] += 1

    return dict(classes)


def test_generated_instance_ids_continue_across_batches(
    dataset_name: str, tempdir: Path
):
    image = create_image(0, tempdir)

    def generator() -> DatasetIterator:
        for class_name, x in [("cat", 0.1), ("dog", 0.5)]:
            yield {
                "media": image,
                "task_name": "animals",
                "annotation": {
                    "class": class_name,
                    "boundingbox": {"x": x, "y": 0.1, "w": 0.2, "h": 0.2},
                },
            }

    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.add(generator(), batch_size=1)
    dataset.make_splits({"train": [image]})

    boxes = LuxonisLoader(dataset, view="train")[0].labels[
        "animals/boundingbox"
    ]

    assert boxes[:, 0].tolist() == [0.0, 1.0]
    assert boxes[:, 1].tolist() == [0.1, 0.5]


def test_task_ingestion(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    dataset = LuxonisDataset(
        dataset_name,
        bucket_storage=bucket_storage,
        delete_local=True,
        delete_remote=True,
    )

    def generator1() -> DatasetIterator:
        for i in range(STEP):
            path = create_image(i, tempdir)
            yield {
                "media": str(path),
                "task_name": "animals",
                "annotation": {
                    "class": "dog",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }
            yield {
                "media": str(path),
                "task_name": "animals",
                "annotation": {
                    "class": "cat",
                    "boundingbox": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.3},
                },
            }
            yield {
                "media": str(path),
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
                "media": str(path),
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
                "media": str(path),
                "annotation": {
                    "class": "dog",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                },
            }
            yield {
                "media": str(path),
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
                "media": str(path),
                "task_name": "animals",
                "annotation": {
                    "class": "dog",
                    "boundingbox": {"x": 0.15, "y": 0.25, "w": 0.1, "h": 0.1},
                },
            }
            yield {
                "media": str(path),
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
                "media": str(path),
                "task_name": "detection",
                "annotation": {
                    "class": "bike",
                    "boundingbox": {"x": 0.9, "y": 0.8, "w": 0.1, "h": 0.4},
                },
            }
            yield {
                "media": str(path),
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
                "media": str(path),
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


def test_a_negative_keeps_the_stored_task_types(
    dataset_name: str, tempdir: Path
):
    """A later `add` that carries only a negative declares its task.

    The task then arrives with no task type, and the types the first `add`
    stored have to survive it.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.add(
        iter(
            [
                {
                    "media": str(create_image(0, tempdir)),
                    "task_name": "vehicles",
                    "annotation": {
                        "class": "car",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.1,
                            "h": 0.1,
                        },
                    },
                }
            ]
        )
    )
    assert dataset.get_tasks() == {
        "vehicles": ["boundingbox", "classification"]
    }

    dataset.add(
        iter(
            [
                {
                    "media": str(create_image(1, tempdir)),
                    "task_name": "vehicles",
                }
            ]
        )
    )

    assert dataset.get_tasks() == {
        "vehicles": ["boundingbox", "classification"]
    }
