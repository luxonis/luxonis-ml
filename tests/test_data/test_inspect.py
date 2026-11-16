from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.data import __main__ as data_cli
from luxonis_ml.data.datasets.base_dataset import DatasetIterator

from .utils import create_dataset, create_image


def test_inspect_with_generated_background(
    dataset_name: str, tempdir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The inspector draws every channel using the loader's class IDs."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:256] = 1

    def generator() -> DatasetIterator:
        for i, class_name in enumerate(["cat", "dog"]):
            yield {
                "file": create_image(i, tempdir),
                "annotation": {
                    "class": class_name,
                    "segmentation": {"mask": mask},
                },
            }

    dataset = create_dataset(dataset_name, generator(), splits={"train": 1.0})
    displayed: list[np.ndarray] = []
    monkeypatch.setattr(data_cli.cv2, "namedWindow", lambda *_: None)
    monkeypatch.setattr(data_cli.cv2, "resizeWindow", lambda *_: None)
    monkeypatch.setattr(
        data_cli.cv2, "imshow", lambda _, image: displayed.append(image)
    )
    monkeypatch.setattr(data_cli.cv2, "waitKey", lambda: -1)

    data_cli.inspect(dataset_name, print_sample_metadata=False)

    assert len(displayed) == 2
    assert all(image.size for image in displayed)
    assert dataset.get_classes() == {"": {"cat": 0, "dog": 1}}
