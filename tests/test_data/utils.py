from pathlib import Path
from typing import Dict, Set, Union

import cv2
import numpy as np

from luxonis_ml.data import LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.datasets.luxonis_dataset import LuxonisDataset
from luxonis_ml.data.utils.enums import BucketStorage


def gather_tasks(dataset: LuxonisDataset) -> Set[str]:
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


def get_loader_output(loader: LuxonisLoader) -> Set[str]:
    all_labels = set()
    for _, labels in loader:
        all_labels.update(labels.keys())
    return all_labels


def create_dataset(
    dataset_name: str,
    generator: DatasetIterator,
    bucket_storage: BucketStorage = BucketStorage.LOCAL,
    *,
    splits: Union[bool, Dict[str, float]] = True,
    **kwargs,
) -> LuxonisDataset:
    dataset = LuxonisDataset(
        dataset_name,
        delete_existing=True,
        delete_remote=True,
        bucket_storage=bucket_storage,
        **kwargs,
    ).add(generator)
    if splits is True:
        dataset.make_splits()
    elif splits:
        dataset.make_splits(splits)
    return dataset
