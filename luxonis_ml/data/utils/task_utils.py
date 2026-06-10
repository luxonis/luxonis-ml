from collections.abc import Iterator
from functools import lru_cache

import numpy as np

from luxonis_ml.typing import Labels, TaskType


@lru_cache
def task_is_metadata(task: str) -> bool:
    """Check whether a task is a metadata task.

    Args:
        task: Task to check.

    Returns:
        Whether the task is a metadata task.

    """
    return get_task_type(task).startswith("metadata/")


@lru_cache
def split_task(task: str) -> tuple[str, str]:
    """Split a task into task name and task type.

    Args:
        task: Task to split.

    Returns:
        Task name and task type.

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

    """
    return task.split("/", maxsplit=1)[0]


@lru_cache
def get_task_type(task: str) -> str:
    """Return the task type from a task string.

    Example:
        >>> get_task_type("task_name/type")
        'type'
        >>> get_task_type("metadata/name")
        'metadata/name'
        >>> get_task_type("task_name/metadata/name")
        'metadata/name'

    Args:
        task: Task string, such as ``"task_name/type"``.

    Returns:
        Task type. Metadata tasks are returned as ``"metadata/type"``.

    """
    parts = task.split("/")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        if parts[0] == "metadata":
            return task
        return parts[1]
    if parts[-2] == "metadata":
        return f"metadata/{parts[-1]}"
    return task.rsplit("/", maxsplit=1)[-1]


def task_type_iterator(
    labels: Labels, task_type: TaskType
) -> Iterator[tuple[str, np.ndarray]]:
    """Iterate over labels of a specific task type.

    Args:
        labels: Labels to iterate over.
        task_type: Label type to yield.

    Returns:
        Iterator over matching labels.

    """
    for task, array in labels.items():
        if get_task_type(task) == task_type:
            yield task, array
