from pathlib import Path
from typing import Literal

import pytest

from luxonis_ml.data.loaders.luxonis_loader import LuxonisLoader
from tests.test_data.utils import create_dataset, create_image


@pytest.mark.parametrize("n_samples", [0, 1, 2])
def test_empty(dataset_name: str, tempdir: Path, n_samples: Literal[0, 1, 2]):
    config = [
        {
            "name": "Defocus",
            "params": {"p": 1.0},
        },
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        },
    ]

    def generator(keep_samples: Literal[0, 1, 2]):
        for i in range(20):
            img = create_image(i, tempdir)
            if i < keep_samples:
                yield {
                    "file": img,
                    "annotation": {
                        "class": ["dog", "cat"][i],
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
