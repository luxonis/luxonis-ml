from pathlib import Path
from typing import List

import pytest

from luxonis_ml.data import DatasetIterator, LuxonisLoader
from luxonis_ml.typing import Params
from tests.test_data.utils import create_dataset, create_image


@pytest.mark.parametrize("n_samples", range(20))
def test_empty(dataset_name: str, tempdir: Path, n_samples: int):
    config: List[Params] = [
        {
            "name": "Defocus",
            "params": {"p": 1.0},
        },
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        },
    ]

    def generator(keep_samples: int) -> DatasetIterator:
        for i in range(20):
            img = create_image(i, tempdir)
            if i < keep_samples:
                yield {
                    "file": img,
                    "annotation": {
                        "class": ["dog", "cat"][i % 2],
                        "segmentation": {
                            "points": [(0.2, 0.2), (0.2, 0.4), (0.4, 0.4)],
                            "height": 256,
                            "width": 256,
                        },
                        "boundingbox": {
                            "x": 0.2,
                            "y": 0.2,
                            "w": 0.3,
                            "h": 0.3,
                        },
                        "keypoints": {
                            "keypoints": [(0.25, 0.25, 2), (0.3, 0.3, 1)]
                        },
                        "metadata": {
                            "breed": "labrador",
                        },
                    },
                }
            else:
                yield {"file": img}

    dataset = create_dataset(
        dataset_name, generator(n_samples), splits={"train": 1.0}
    )
    loader = LuxonisLoader(
        dataset, augmentation_config=config, height=256, width=256
    )
    for _ in loader:
        pass
