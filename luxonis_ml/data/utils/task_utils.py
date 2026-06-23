from collections.abc import Iterator
from functools import lru_cache

import numpy as np

from luxonis_ml.typing import Annotations, TaskType


@lru_cache
def task_is_label(task: str) -> bool:
    """Check whether a task is a label task.

    Args:
        task: Task to check.

    Returns:
        Whether the task is a label task.

    Examples:
        >>> task_is_label("labels/weather")
        True
        >>> task_is_label("camera/labels/weather")
        True
        >>> task_is_label("camera/boundingbox")
        False

    """
    return get_task_type(task).startswith("labels/")


@lru_cache
def split_task(task: str) -> tuple[str, str]:
    """Split a task into task name and task type.

    Args:
        task: Task to split.

    Returns:
        Task name and task type.

    Examples:
        >>> split_task("detector/boundingbox")
        ('detector', 'boundingbox')
        >>> split_task("classification")
        ('', 'classification')

    """
    splits = task.split("/", 1)
    if len(splits) == 1:
        return "", splits[0]
    return splits[0], splits[1]


@lru_cache
def get_task_name(task: str) -> str:
    """Return the task name from a task string.

    Args:
        task: Task string.

    Returns:
        Task name.

    Examples:
        >>> get_task_name("detector/boundingbox")
        'detector'
        >>> get_task_name("classification")
        'classification'

    """
    return task.split("/", maxsplit=1)[0]


@lru_cache
def get_task_type(task: str) -> str:
    """Return the task type from a task string.

    Example:
        >>> get_task_type("task_name/type")
        'type'
        >>> get_task_type("labels/name")
        'labels/name'
        >>> get_task_type("task_name/labels/name")
        'labels/name'

    Args:
        task: Task string, such as ``"task_name/{type}"``.

    Returns:
        Task type. Csutom label tasks are returned as ``"label/{type}"``.

    """
    parts = task.split("/")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        if parts[0] == "labels":
            return task
        return parts[1]
    if parts[-2] == "labels":
        return f"labels/{parts[-1]}"
    return task.rsplit("/", maxsplit=1)[-1]


def task_type_iterator(
    annotations: Annotations, task_type: TaskType
) -> Iterator[tuple[str, np.ndarray]]:
    """Iterate over annotations of a specific task type.

    Args:
        annotations: Annotations to iterate over.
        task_type: Label type to yield.

    Returns:
        Iterator over matching annotations.

    Examples:
        >>> annotations = {
        ...     "detector/boundingbox": np.array([1]),
        ...     "pose/keypoints": np.array([2]),
        ... }
        >>> [(task, arr.tolist()) for task, arr in task_type_iterator(annotations, "keypoints")]
        [('pose/keypoints', [2])]

    """
    for task, array in annotations.items():
        if get_task_type(task) == task_type:
            yield task, array
