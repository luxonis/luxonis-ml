from typing import Tuple


def split_task(task: str) -> Tuple[str, str]:
    """Splits a task into its task name and type.

    @type task: str
    @param task: The task to split.
    @rtype: Tuple[str, str]
    @return: A tuple containing the task name and type.
    """
    return task.split("/")[0], task.split("/")[-1]


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
    return task.split("/")[-1]
