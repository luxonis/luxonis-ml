"""Split preservation for real datasets kept in cloud storage."""

from pathlib import Path

import pytest

from luxonis_ml.data import (
    LuxonisDataset,
    LuxonisLoader,
)
from luxonis_ml.data.utils import get_task_type


@pytest.mark.parametrize(
    ("url", "expected_split_sizes", "loader_view"),
    [
        (
            "coco_valid_only_debug.zip",
            {"train": 0, "val": 2, "test": 1},
            "val",
        ),
        (
            "native_val_only_debug.zip",
            {"train": 0, "val": 3, "test": 0},
            "val",
        ),
    ],
)
def test_partial_split_fixture_is_preserved(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url: str,
    expected_split_sizes: dict[str, int],
    loader_view: str,
):
    dataset = LuxonisDataset.import_dataset(
        f"{storage_url.rstrip('/')}/{url}",
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert {
        split_name: len(group_ids) for split_name, group_ids in splits.items()
    } == expected_split_sizes

    loader = LuxonisLoader(dataset, view=loader_view)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == {"boundingbox", "classification"}
    dataset.delete_dataset(delete_local=True)
