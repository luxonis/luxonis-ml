import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from luxonis_ml.data import LuxonisLoader, LuxonisParser
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.datasets.luxonis_dataset import LuxonisDataset
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.enums import DatasetType


def gather_tasks(dataset: LuxonisDataset) -> set[str]:
    return {
        f"{task_name}/{task_type}"
        for task_name, task_types in dataset.get_tasks().items()
        for task_type in task_types
    }


def create_image(i: int, dir: Path) -> Path:
    path = dir / f"img_{i}.jpg"
    if not path.exists():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[0:10, 0:10] = np.random.randint(
            0, 255, (10, 10, 3), dtype=np.uint8
        )
        cv2.imwrite(str(path), img)
    return path


def get_loader_output(loader: LuxonisLoader) -> set[str]:
    all_labels = set()
    for _, labels in loader:
        all_labels.update(labels.keys())
    return all_labels


def create_dataset(
    dataset_name: str,
    generator: DatasetIterator,
    bucket_storage: BucketStorage = BucketStorage.LOCAL,
    *,
    splits: bool | dict[str, float] | tuple = True,
    delete_local: bool = True,
    delete_remote: bool = True,
    **kwargs,
) -> LuxonisDataset:
    dataset = LuxonisDataset(
        dataset_name,
        delete_local=delete_local,
        delete_remote=delete_remote,
        bucket_storage=bucket_storage,
        **kwargs,
    ).add(generator)
    if splits is True:
        dataset.make_splits()
    elif splits:
        dataset.make_splits(splits)
    return dataset


def read_dataset_metadata(dataset: LuxonisDataset) -> dict[str, Any]:
    """Read the stored ``metadata.json`` of the dataset."""
    return json.loads((dataset._metadata_path / "metadata.json").read_text())


def write_dataset_metadata(
    dataset: LuxonisDataset, dataset_metadata: dict[str, Any]
) -> LuxonisDataset:
    """Replace the stored ``metadata.json`` and open the dataset again."""
    (dataset._metadata_path / "metadata.json").write_text(
        json.dumps(dataset_metadata)
    )
    return LuxonisDataset(dataset.identifier)


def set_ldf_version(dataset: LuxonisDataset, version: str) -> LuxonisDataset:
    """Change the stored LDF version and open the dataset again."""
    dataset_metadata = read_dataset_metadata(dataset)
    dataset_metadata["ldf_version"] = version
    return write_dataset_metadata(dataset, dataset_metadata)


def keypoint_annotations(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the keypoint annotations of exported native records.

    Every keypoint detection also emits a classification record, which
    the result leaves out.
    """
    return [
        record["annotation"]["keypoints"]
        for record in records
        if "keypoints" in record.get("annotation", {})
    ]


def export_and_import(
    dataset: LuxonisDataset,
    tempdir: Path,
    dataset_type: DatasetType = DatasetType.NATIVE,
    **kwargs,
) -> LuxonisDataset:
    """Export the dataset to ``tempdir / "exported"`` and parse it back."""
    exported = dataset.export(tempdir / "exported", dataset_type, **kwargs)
    assert isinstance(exported, Path)
    return LuxonisParser(
        str(exported / dataset.identifier),
        dataset_type=dataset_type,
        dataset_name=f"{dataset.identifier}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()
