from copy import deepcopy
from typing import Dict

import numpy as np
import pytest

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.typing import Labels, TaskType


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((320, 320, 3), dtype=np.uint8)


@pytest.fixture
def labels() -> Labels:
    return {
        "task/classification": np.array([1.0]),
        "task/boundingbox": np.array(
            [
                [0.0, 0.57, 0.30, 0.17, 0.25],
                [0.0, 0.39, 0.27, 0.20, 0.10],
            ]
        ),
        "task/keypoints": np.array(
            [
                [0.69, 0.37, 2.0, 0.0, 0.0, 0.0, 0.68, 0.36, 2.0],
                [0.51, 0.33, 2.0, 0.0, 0.0, 0.0, 0.50, 0.32, 2.0],
            ]
        ),
        "task/segmentation": np.zeros((1, 320, 320)),
    }


@pytest.fixture
def targets() -> Dict[str, TaskType]:
    return {
        "task/boundingbox": "boundingbox",
        "task/keypoints": "keypoints",
        "task/segmentation": "segmentation",
    }


def test_mosaic4(
    image: np.ndarray, labels: Labels, targets: Dict[str, TaskType]
):
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        }
    ]
    augmentations = AlbumentationsEngine(256, 256, targets, config)
    augmentations.apply([(image.copy(), deepcopy(labels)) for _ in range(4)])


def test_mixup(
    image: np.ndarray, labels: Labels, targets: Dict[str, TaskType]
):
    config = [{"name": "MixUp", "params": {"p": 1.0}}]
    augmentations = AlbumentationsEngine(256, 256, targets, config)
    augmentations.apply([(image.copy(), deepcopy(labels)) for _ in range(2)])