from typing import Dict, List

import numpy as np
import pytest
from pytest import FixtureRequest


@pytest.fixture
def height() -> int:
    return 480


@pytest.fixture
def width() -> int:
    return 640


@pytest.fixture
def augmentation_data(
    height: int, width: int, request: FixtureRequest
) -> Dict[str, List[np.ndarray]]:
    batch_size: int = request.param
    return {
        "image": [
            np.random.rand(height, width, 3) * 255 for _ in range(batch_size)
        ],
        "bboxes": [
            np.array([[0.3 + i * 0.1, 0.3 + i * 0.1, 0.1, 0.1, 0, 0]])
            for i in range(batch_size)
        ],
        "keypoints": [
            np.array([[64.0 + i * 10, 150.0 + i * 10, 0.0, 0.0, 2.0]])
            for i in range(batch_size)
        ],
        "mask": [
            np.random.randint(0, 2, (height, width)) for _ in range(batch_size)
        ],
    }
