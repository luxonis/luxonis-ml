"""Ultralytics NDJSON exports fetched from cloud storage."""

from pathlib import Path

import pytest

from luxonis_ml.data import (
    LuxonisDataset,
    LuxonisLoader,
)
from luxonis_ml.data.utils import get_task_type


@pytest.mark.parametrize(
    ("url_suffix", "dataset_type"),
    [
        ("fruit_ndjson.zip", None),
        ("fruit_ndjson.zip", "ultralytics-ndjson"),
        ("fruit_ndjson_remote/fruit.ndjson", None),
    ],
)
def test_ultralytics_ndjson_parser(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url_suffix: str,
    dataset_type: str | None,
):
    """The zipped export, imported by detection, by explicit type, and as
    a bare NDJSON file referencing remote images, must all come out the
    same. ``dataset_type=None`` is the auto-detection default.
    """
    dataset = LuxonisDataset.import_dataset(
        f"{storage_url.rstrip('/')}/{url_suffix}",
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        delete_local=True,
        save_dir=tempdir,
    )

    assert len(dataset) > 0
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == {
        "boundingbox",
        "classification",
    }
    dataset.delete_dataset(delete_local=True)
