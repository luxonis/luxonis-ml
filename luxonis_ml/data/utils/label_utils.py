from typing import Iterator, Tuple

import numpy as np

from .types import Labels


def split_task(task: str) -> Tuple[str, str]:
    """Splits a task into its task name and type.

    @type task: str
    @param task: The task to split.
    @rtype: Tuple[str, str]
    @return: A tuple containing the task name and type.
    """
    return task.split("/")[0], task.split("/")[1]


def get_task_name(task: str) -> str:
    """Returns the task name from a task.

    @type task: str
    @param task: The task.
    @rtype: str
    @return: The task name.
    """
    return task.split("/")[0]


def get_task_type(task: str) -> str:
    """Returns the task type from a task.

    @type task: str
    @param task: The task.
    @rtype: str
    @return: The task type.
    """
    return task.split("/")[1]


def task_type_iterator(
    labels: Labels, label_type: str
) -> Iterator[Tuple[str, np.ndarray]]:
    """Iterates over labels of a specific type.

    @type labels: Labels
    @param labels: The labels to iterate over.
    @type label_type: str
    @param label_type: The type of label to iterate over.
    @rtype: Iterator[Tuple[str, np.ndarray]]
    @return: An iterator over the labels of the specified type.
    """
    for task, arr in labels.items():
        task_type = get_task_type(task)
        if task_type == label_type:
            yield task, arr
