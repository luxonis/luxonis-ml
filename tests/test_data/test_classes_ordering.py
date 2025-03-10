import random
from pathlib import Path
from typing import List

import numpy as np
from pytest_subtests.plugin import SubTests

from luxonis_ml.data import LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.datasets.luxonis_dataset import LuxonisDataset

from .utils import create_dataset, create_image


def build_segmentation(
    polylines: List[List[float]], img_w: int, img_h: int
) -> dict:
    points = []
    for polyline in polylines:
        poly_arr = np.array(polyline).reshape(-1, 2)
        points += [
            (poly_arr[i, 0] / img_w, poly_arr[i, 1] / img_h)
            for i in range(len(poly_arr))
        ]
    return {"height": img_h, "width": img_w, "points": points}


def test_ordering_loader_background(tempdir: Path, subtests: SubTests):
    width, height = 512, 512
    segmentation = build_segmentation(
        [[0, 0, width / 2, 0, width / 2, height / 2, 0, height / 2]],
        width,
        height,
    )

    def generator(classes_list: List[str]) -> DatasetIterator:
        for i, class_name in enumerate(classes_list):
            yield {
                "file": create_image(i, tempdir),
                "annotation": {
                    "class": class_name,
                    "segmentation": segmentation,
                },
            }

    with subtests.test("implicit"):
        classes_list = ["dog", "cat", "airplane"]
        dataset = create_dataset(
            "test_segmentation_no_background", generator(classes_list)
        )

        loader = LuxonisLoader(dataset, height=height, width=width)
        for _, labels in loader:
            classification = labels["/classification"]
            assert len(classification) == len(classes_list) + 1
            assert not np.array_equal(
                classification, [1.0] + [0.0] * len(classes_list)
            )

        assert loader.classes[""] == {
            "background": 0,
            "airplane": 1,
            "cat": 2,
            "dog": 3,
        }

    with subtests.test("explicit"):
        classes_list = ["dog", "cat", "airplane", "background"]
        dataset = create_dataset(
            "test_segmentation_with_background", generator(classes_list)
        )

        loader = LuxonisLoader(
            dataset, view="train", height=height, width=width
        )
        for _, labels in loader:
            classification = labels["/classification"]
            assert len(classification) == len(classes_list)

        assert loader.classes[""] == {
            "background": 0,
            "airplane": 1,
            "cat": 2,
            "dog": 3,
        }


def test_ordering_loader_no_backgroun(tempdir: Path):
    width, height = 512, 512
    left_seg = build_segmentation(
        [[0, 0, width / 2, 0, width / 2, height, 0, height]], width, height
    )
    right_seg = build_segmentation(
        [[width / 2, 0, width, 0, width, height, width / 2, height]],
        width,
        height,
    )

    def generator() -> DatasetIterator:
        yield {
            "file": create_image(0, tempdir),
            "annotation": {"class": "dog", "segmentation": right_seg},
        }
        yield {
            "file": create_image(0, tempdir),
            "annotation": {"class": "cat", "segmentation": left_seg},
        }

    dataset = create_dataset("test_dual_class_segmentation", generator())

    loader = LuxonisLoader(dataset, view="train", height=height, width=width)
    for _, labels in loader:
        classification = labels["/classification"]
        assert len(classification) == 2

    assert loader.classes == {"": {"cat": 0, "dog": 1}}


def test_ordering_dataset(
    dataset_name: str, tempdir: Path, subtests: SubTests
):
    class_names = list("abcdefghijKLM")

    def generator(classes: List[str]) -> DatasetIterator:
        for i, class_name in enumerate(classes):
            img = create_image(i, tempdir)
            yield {
                "file": img,
                "task_name": "ordering",
                "annotation": {
                    "class": class_name,
                },
            }

    with subtests.test("no_background"):
        random.shuffle(class_names)
        dataset = create_dataset(dataset_name, generator(class_names))
        assert dataset.get_classes()["ordering"] == {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
            "i": 8,
            "j": 9,
            "K": 10,
            "L": 11,
            "M": 12,
        }

    with subtests.test("with_background"):
        class_names.append("background")
        random.shuffle(class_names)
        dataset = create_dataset(dataset_name, generator(class_names))
        assert dataset.get_classes()["ordering"] == {
            "background": 0,
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
            "g": 7,
            "h": 8,
            "i": 9,
            "j": 10,
            "K": 11,
            "L": 12,
            "M": 13,
        }

    with subtests.test("set_classes"):
        dataset = LuxonisDataset(dataset_name)
        dataset.set_classes(["d", "b", "a", "c"])
        assert dataset.get_classes()["ordering"] == {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
        }
        dataset.set_classes(["d", "b", "a", "c", "background"])
        assert dataset.get_classes()["ordering"] == {
            "background": 0,
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
        }
        dataset.set_classes({"a": 2, "b": 1, "c": 0, "d": 3, "background": 4})
        assert dataset.get_classes()["ordering"] == {
            "a": 2,
            "b": 1,
            "c": 0,
            "d": 3,
            "background": 4,
        }
