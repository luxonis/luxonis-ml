from typing import Iterator, Set, Tuple

import numpy as np

from luxonis_ml.data.utils import (
    get_task_name,
    get_task_type,
    split_task,
    task_is_metadata,
    task_type_iterator,
)


def test_task_is_metadata():
    assert task_is_metadata("a/b/metadata/c")
    assert task_is_metadata("metadata/c")
    assert not task_is_metadata("a/b/c")


def test_split_task():
    assert split_task("a/b") == ("a", "b")
    assert split_task("a") == ("", "a")
    assert split_task("a/b/c/d/e") == ("a", "b/c/d/e")


def test_get_task_name():
    assert get_task_name("a/b") == "a"
    assert get_task_name("a") == "a"
    assert get_task_name("a/b/c") == "a"


def test_get_task_type():
    assert get_task_type("a/b") == "b"
    assert get_task_type("a") == "a"
    assert get_task_type("a/b/c") == "c"
    assert get_task_type("a/b/metadata/c") == "metadata/c"
    assert get_task_type("metadata/c") == "metadata/c"


def test_task_type_iterator():
    labels = {
        "task/segmentation": np.array([]),
        "task/metadata/text": np.array([]),
        "task/instance_segmentation": np.array([]),
        "task/subtask/boundingbox": np.array([]),
        "task/subtask/segmentation": np.array([]),
    }

    def compare(
        iterator: Iterator[Tuple[str, np.ndarray]], expected: Set[str]
    ) -> None:
        tasks = set()
        for task, _ in iterator:
            tasks.add(task)
        assert tasks == expected

    compare(
        task_type_iterator(labels, "segmentation"),
        {"task/segmentation", "task/subtask/segmentation"},
    )
    compare(
        task_type_iterator(labels, "boundingbox"), {"task/subtask/boundingbox"}
    )
    compare(
        task_type_iterator(labels, "instance_segmentation"),
        {"task/instance_segmentation"},
    )
    compare(task_type_iterator(labels, "keypoints"), set())
