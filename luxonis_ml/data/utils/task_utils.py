from functools import lru_cache
from typing import Iterator, Tuple

import numpy as np

from luxonis_ml.typing import Labels, TaskType


@lru_cache()
def split_task(task: str) -> Tuple[str, str]:
    """Splits a task into its task name and type.

    @type task: str
    @param task: The task to split.
    @rtype: Tuple[str, str]
    @return: A tuple containing the task name and type.
    """
    return task.split("/")[0], task.split("/")[1]


@lru_cache()
def get_qualified_task_name(task: str) -> str:
    """Returns the qualified task name from a task."""
    parts = task.split("/")
    if "metadata" in parts:
        return "/".join(parts[:-2])
    return "/".join(parts[:-1])


@lru_cache()
def get_task_name(task: str) -> str:
    """Returns the task name from a task.

    @type task: str
    @param task: The task.
    @rtype: str
    @return: The task name.
    """
    return task.split("/")[0]


@lru_cache()
def get_task_type(task: str) -> str:
    """Returns the task type from a task.

    @type task: str
    @param task: The task.
    @rtype: str
    @return: The task type.
    """
    parts = task.split("/")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
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
    for task, arr in labels.items():
        if get_task_type(task) == task_type:
            yield task, arr
