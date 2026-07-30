"""Ultralytics NDJSON exports fetched from cloud storage."""

from pathlib import Path

from luxonis_ml.data import (
    LuxonisDataset,
    LuxonisLoader,
)
from luxonis_ml.data.utils import get_task_type


def test_ultralytics_ndjson_parser(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    url = f"{storage_url.rstrip('/')}/fruit_ndjson.zip"
    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
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


def test_ultralytics_ndjson_parser_explicit_type(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    url = f"{storage_url.rstrip('/')}/fruit_ndjson.zip"
    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
        dataset_type="ultralytics-ndjson",
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


def test_ultralytics_ndjson_remote_urls_parser(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    url = f"{storage_url.rstrip('/')}/fruit_ndjson_remote/fruit.ndjson"
    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
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
