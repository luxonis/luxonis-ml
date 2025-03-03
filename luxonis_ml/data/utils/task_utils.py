from functools import lru_cache
from typing import Iterator, Tuple

import numpy as np

from luxonis_ml.typing import Labels, TaskType


@lru_cache
def task_is_metadata(task: str) -> bool:
    """Returns whether a task is a metadata task.

    @type task: str
    @param task: The task to check.
    @rtype: bool
    @return: Whether the task is a metadata task.
    """
    return get_task_type(task).startswith("metadata/")


@lru_cache
def split_task(task: str) -> Tuple[str, str]:
    """Splits a task into its task name and type.

    @type task: str
    @param task: The task to split.
    @rtype: Tuple[str, str]
    @return: A tuple containing the task name and type.
    """
    splits = task.split("/", 1)
    if len(splits) == 1:
        return "", splits[0]
    return splits[0], splits[1]


@lru_cache
def get_task_name(task: str) -> str:
    """Returns the task name from a task.

    @type task: str
    @param task: The task.
    @rtype: str
    @return: The task name.
    """
    return task.split("/")[0]


@lru_cache
def get_task_type(task: str) -> str:
    """Returns the task type from a task.

    Example:

        >>> get_task_type("task_name/type")
        'type'
        >>> get_task_type("metadata/name")
        'metadata/name'
        >>> get_task_type("task_name/metadata/name")
        'metadata/name'

    @type task: str
    @param task: The task in a format like "task_name/type".
    @rtype: str
    @return: The task type. If the task is a metadata task,
        the type will be "metadata/type".
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
    return task.split("/")[-1]


def task_type_iterator(
    labels: Labels, task_type: TaskType
) -> Iterator[Tuple[str, np.ndarray]]:
    """Iterates over labels of a specific type.

    @type labels: Labels
    @param labels: The labels to iterate over.
    @type task_type: str
    @param task_type: The type of label to iterate over.
    @rtype: Iterator[Tuple[str, np.ndarray]]
    @return: An iterator over the labels of the specified type.
    """
    for task, array in labels.items():
        if get_task_type(task) == task_type:
            yield task, array
