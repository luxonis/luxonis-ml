from typing import TYPE_CHECKING, cast

import pytest

from luxonis_ml.data.utils.cli_utils import print_info
from luxonis_ml.data.utils.enums import BucketStorage

if TYPE_CHECKING:
    from luxonis_ml.data import LuxonisDataset


class _InfoDataset:
    """The parts of a dataset that `print_info` reads."""

    identifier = "fruits"
    version = "1.0"
    _bucket_storage = BucketStorage.LOCAL
    _team_id = "team"

    def __init__(
        self,
        classes: dict[str, dict[str, int]],
        tasks: dict[str, list[str]],
        splits: dict[str, list[str]] | None,
    ) -> None:
        self._classes = classes
        self._tasks = tasks
        self._splits = splits

    def get_classes(self) -> dict[str, dict[str, int]]:
        return self._classes

    def get_tasks(self) -> dict[str, list[str]]:
        return self._tasks

    def get_source_names(self) -> list[str]:
        return ["image"]

    def get_splits(self) -> dict[str, list[str]] | None:
        return self._splits

    def __len__(self) -> int:
        return 4


def test_print_info_shows_named_tasks_and_split_sizes(
    capsys: pytest.CaptureFixture[str],
):
    dataset = _InfoDataset(
        {"fruits": {"apple": 0, "pear": 1}},
        {"fruits": ["classification", "boundingbox"]},
        {"train": ["a", "b", "c"], "val": ["d"]},
    )

    print_info(cast("LuxonisDataset", dataset))

    out = capsys.readouterr().out
    assert "Task Name" in out
    assert "apple, pear" in out
    assert "boundingbox, classification" in out
    assert "(75.0%)" in out
    assert "Total: 4" in out


def test_print_info_without_task_names_or_splits(
    capsys: pytest.CaptureFixture[str],
):
    dataset = _InfoDataset(
        {"": {"apple": 0}}, {"": ["classification"]}, splits=None
    )

    print_info(cast("LuxonisDataset", dataset))

    out = capsys.readouterr().out
    assert "Task Name" not in out
    assert "apple" in out
    assert "No splits found" in out
