from typing import get_args

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from luxonis_ml.data.utils import (
    get_task_name,
    get_task_type,
    split_task,
    task_is_metadata,
    task_type_iterator,
)
from luxonis_ml.typing import TaskType

# A single segment of a task string. ``"metadata"`` is excluded because
# it is a reserved segment with its own parsing rules, tested separately.
segments = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=1,
    max_size=8,
).filter(lambda segment: segment != "metadata")

task_types = st.sampled_from(get_args(TaskType))


@given(name=segments, task_type=segments)
def test_task_splits_into_name_and_type(name: str, task_type: str):
    task = f"{name}/{task_type}"

    assert split_task(task) == (name, task_type)
    assert get_task_name(task) == name
    assert get_task_type(task) == task_type
    assert not task_is_metadata(task)


@given(task_type=segments)
def test_task_without_a_name_is_its_own_type(task_type: str):
    assert split_task(task_type) == ("", task_type)
    assert get_task_name(task_type) == task_type
    assert get_task_type(task_type) == task_type
    assert not task_is_metadata(task_type)


@given(
    name=segments,
    middle=st.lists(segments, min_size=1, max_size=3),
    task_type=segments,
)
def test_nested_task_keeps_only_first_and_last_segment(
    name: str, middle: list[str], task_type: str
):
    task = "/".join([name, *middle, task_type])

    assert get_task_name(task) == name
    assert get_task_type(task) == task_type
    assert split_task(task) == (name, "/".join([*middle, task_type]))
    assert not task_is_metadata(task)


@given(name=segments, middle=st.lists(segments, max_size=2), field=segments)
def test_metadata_type_keeps_its_prefix(
    name: str, middle: list[str], field: str
):
    task = "/".join([name, *middle, "metadata", field])

    assert get_task_type(task) == f"metadata/{field}"
    assert get_task_name(task) == name
    assert task_is_metadata(task)


@given(task=st.text())
def test_split_task_loses_no_characters(task: str):
    """Splitting is lossless: both parts always rejoin into the original
    task, and a task without a separator keeps its whole string as the
    type.
    """
    name, task_type = split_task(task)

    if "/" in task:
        assert f"{name}/{task_type}" == task
    else:
        assert (name, task_type) == ("", task)


@given(
    names=st.lists(segments, min_size=1, max_size=4, unique=True),
    task_type=task_types,
    other_type=task_types,
)
def test_task_type_iterator_yields_every_matching_task(
    names: list[str], task_type: TaskType, other_type: TaskType
):
    assume(other_type != task_type)

    matching = {
        f"{name}/{task_type}": np.array([i]) for i, name in enumerate(names)
    }
    labels = matching | {
        f"{name}/{other_type}": np.array([]) for name in names
    }

    found = dict(task_type_iterator(labels, task_type))

    assert found.keys() == matching.keys()
    for task, array in found.items():
        assert array is labels[task]


@given(
    tasks=st.lists(segments, max_size=4, unique=True),
    task_type=task_types,
)
def test_task_type_iterator_ignores_unrelated_types(
    tasks: list[str], task_type: TaskType
):
    assume(task_type not in tasks)

    labels = {task: np.array([]) for task in tasks}

    assert list(task_type_iterator(labels, task_type)) == []


def test_bare_metadata_task():
    """A task whose name is the reserved ``"metadata"`` segment is parsed
    as a metadata type rather than as a name and a type.
    """
    assert get_task_type("metadata/weather") == "metadata/weather"
    assert get_task_name("metadata/weather") == "metadata"
    assert task_is_metadata("metadata/weather")


def test_metadata_is_only_recognized_as_second_to_last_segment():
    assert get_task_type("camera/metadata/weather/extra") == "extra"
    assert not task_is_metadata("camera/metadata/weather/extra")
    assert get_task_type("camera/metadata") == "metadata"
    assert not task_is_metadata("camera/metadata")
