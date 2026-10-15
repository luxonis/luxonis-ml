"""Opening a dataset written before the LDF 3.0 bump."""

from pathlib import Path

import pytest

from luxonis_ml.data import LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.utils.constants import LDF_VERSION

from .utils import create_dataset, create_image, set_ldf_version


@pytest.mark.parametrize("version", ["2.0.0", "2.1.0"])
def test_2_x_dataset_loads(dataset_name: str, tempdir: Path, version: str):
    """A 2.x dataset takes the current stamp, and its labels load."""

    def generator() -> DatasetIterator:
        yield {
            "file": create_image(0, tempdir),
            "task_name": "vehicles",
            "annotation": {
                "class": "car",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            },
        }

    dataset = create_dataset(dataset_name, generator(), splits={"train": 1.0})
    reopened = set_ldf_version(dataset, version)

    assert reopened.version == LDF_VERSION
    assert reopened.get_classes() == {"vehicles": {"car": 0}}

    labels = LuxonisLoader(reopened, view="train")[0].labels
    assert labels["vehicles/boundingbox"].shape == (1, 5)
