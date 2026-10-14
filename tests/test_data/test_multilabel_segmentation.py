"""Overlapping segmentation masks in a multi-label task."""

from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator

from .utils import create_dataset, create_image

SIZE = 512  # `create_image` writes a 512x512 image.


@pytest.fixture
def overlapping(dataset_name: str, tempdir: Path) -> LuxonisDataset:
    """Build one image with ``dog`` in rows 0-299 and ``cat`` in rows
    100-399.

    Rows 100-299 have both masks. No mask covers rows 400-511, so the
    loader adds a ``background`` class there.
    """
    dog = np.zeros((SIZE, SIZE), dtype=np.uint8)
    dog[:300] = 1
    cat = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cat[100:400] = 1

    def generator() -> DatasetIterator:
        for class_name, mask in [("dog", dog), ("cat", cat)]:
            yield {
                "file": create_image(0, tempdir),
                "annotation": {
                    "class": class_name,
                    "segmentation": {"mask": mask},
                },
            }

    return create_dataset(dataset_name, generator(), splits={"train": 1.0})


@pytest.mark.parametrize(
    ("multilabel_tasks", "expected"),
    [
        pytest.param(
            None,
            {50: {"dog"}, 200: {"dog"}, 350: {"cat"}, 450: {"background"}},
            id="single-label",
        ),
        pytest.param(
            [""],
            {
                50: {"dog"},
                200: {"dog", "cat"},
                350: {"cat"},
                450: {"background"},
            },
            id="multi-label",
        ),
    ],
)
def test_classes_of_each_row(
    overlapping: LuxonisDataset,
    multilabel_tasks: list[str] | None,
    expected: dict[int, set[str]],
):
    loader = LuxonisLoader(overlapping, multilabel_tasks=multilabel_tasks)

    mask = loader[0].labels["/segmentation"]
    classes = loader._classes[""]
    assert {
        row: {name for name, i in classes.items() if mask[i, row, 0]}
        for row in expected
    } == expected


def test_unknown_multilabel_task_is_an_error(overlapping: LuxonisDataset):
    with pytest.raises(ValueError, match="does not load"):
        LuxonisLoader(overlapping, multilabel_tasks=["missing"])
