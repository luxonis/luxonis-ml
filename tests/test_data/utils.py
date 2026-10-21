import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

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


def pose_records(
    tempdir: Path, *keypoints: dict[str, Any] | list[Any], start: int = 0
) -> DatasetIterator:
    """Yield one record of the task ``pose`` for each set of keypoints."""
    for i, named in enumerate(keypoints, start=start):
        yield {
            "file": create_image(i, tempdir),
            "task_name": "pose",
            "annotation": {
                "class": "person",
                "instance_id": 0,
                "keypoints": {"keypoints": named},
            },
        }


def pose_rows(dataset: LuxonisDataset) -> list[list[float]]:
    """Return the keypoint rows of every sample, sorted."""
    return sorted(
        row
        for _, labels in LuxonisLoader(dataset)
        for row in labels["pose/keypoints"].tolist()
    )


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


def write_dataset_rows(dataset: LuxonisDataset, df: pl.DataFrame) -> None:
    """Replace the stored annotation rows of the dataset with ``df``."""
    for parquet_file in dataset._annotations_path.glob("*.parquet"):
        parquet_file.unlink()
    df.write_parquet(dataset._annotations_path / "0000000000.parquet")


def set_ldf_version(dataset: LuxonisDataset, version: str) -> LuxonisDataset:
    """Change the stored LDF version and open the dataset again."""
    dataset_metadata = read_dataset_metadata(dataset)
    dataset_metadata["ldf_version"] = version
    return write_dataset_metadata(dataset, dataset_metadata)


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
