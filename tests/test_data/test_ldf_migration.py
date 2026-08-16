"""Opening a dataset written before the LDF 3.0 bump."""

import json
from pathlib import Path

from semver.version import Version

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.utils.constants import LDF_VERSION

from .utils import create_dataset, create_image


def test_a_2_x_dataset_still_loads(dataset_name: str, tempdir: Path):
    """A major bump sends every older dataset through a migration.

    The only one that existed was written for LDF 1.0, and it renames
    columns a 2.x dataset does not have. Without a version to dispatch on,
    the bump would stop every stored dataset from opening.
    """

    def generator() -> DatasetIterator:
        yield {
            "media": create_image(0, tempdir),
            "task_name": "vehicles",
            "annotation": {
                "class": "car",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            },
        }

    dataset = create_dataset(dataset_name, generator(), splits={"train": 1.0})
    metadata_path = dataset._metadata_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["ldf_version"] = "2.1.0"
    metadata_path.write_text(json.dumps(metadata))

    reopened = LuxonisDataset(dataset_name)

    assert reopened.version == Version.parse("2.1.0")
    assert reopened.version.major != LDF_VERSION.major
    assert reopened.get_classes() == {"vehicles": {"car": 0}}

    labels = LuxonisLoader(reopened, view="train")[0].labels

    assert labels["vehicles/boundingbox"].shape == (1, 5)
