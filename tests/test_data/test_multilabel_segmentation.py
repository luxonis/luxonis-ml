"""A dataset with multi-label semantic masks."""

from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.ldf import SCHEMA_METADATA_KEY, DatasetSchema
from luxonis_ml.typing import LoaderOutput

from .utils import create_dataset, create_image, export_and_import

# `create_image` writes 512x512 images. `road` covers rows 0-299 and `wet`
# rows 200-511, so the two masks share rows 200-299.
ROAD = np.zeros((512, 512), dtype=np.uint8)
ROAD[:300] = 1
WET = np.zeros((512, 512), dtype=np.uint8)
WET[200:] = 1


@pytest.fixture
def dataset(dataset_name: str, tempdir: Path) -> LuxonisDataset:
    """Build a dataset with the flagged ``wet`` mask in ``train`` only."""
    flagged, plain = create_image(0, tempdir), create_image(1, tempdir)

    def generator() -> DatasetIterator:
        yield {
            "media": flagged,
            "annotation": {
                "scene": [
                    {"class": "road", "segmentation": {"mask": ROAD}},
                    {
                        "class": "wet",
                        "segmentation": {"mask": WET, "multilabel": True},
                    },
                ]
            },
        }
        yield {
            "media": plain,
            "annotation": {
                "scene": [
                    {"class": "road", "segmentation": {"mask": ROAD}},
                    {"class": "wet", "segmentation": {"mask": WET}},
                ]
            },
        }

    dataset = create_dataset(dataset_name, generator(), splits=False)
    dataset.make_splits({"train": [flagged], "val": [plain]})
    return dataset


def schema_of(sample: LoaderOutput) -> DatasetSchema:
    return DatasetSchema.model_validate(sample.metadata[SCHEMA_METADATA_KEY])


def classes_of_row(sample: LoaderOutput, row: int) -> set[str]:
    """Return the classes of the first pixel of a row."""
    mask = sample.labels["scene/segmentation"]
    classes = schema_of(sample).classes["scene"]
    return {name for name, i in classes.items() if mask[i, row, 0]}


def test_only_a_flagged_mask_keeps_its_overlap(dataset: LuxonisDataset):
    flagged = LuxonisLoader(dataset, view="train")[0]
    plain = LuxonisLoader(dataset, view="val")[0]

    assert classes_of_row(flagged, 250) == {"road", "wet"}
    assert classes_of_row(plain, 250) == {"road"}
    assert classes_of_row(plain, 400) == {"wet"}


def test_every_view_knows_the_task_is_multilabel(dataset: LuxonisDataset):
    """The ``val`` view has no flagged mask, but the same task layout."""
    for view in ("train", "val"):
        sample = LuxonisLoader(dataset, view=view)[0]
        assert schema_of(sample).multilabel == {"scene"}


def test_a_native_export_keeps_the_flag(
    dataset: LuxonisDataset, tempdir: Path
):
    imported = export_and_import(dataset, tempdir)
    imported.make_splits((1, 0, 0), replace_old_splits=True)

    samples = list(LuxonisLoader(imported))

    assert schema_of(samples[0]).multilabel == {"scene"}
    assert {"road", "wet"} in [classes_of_row(s, 250) for s in samples]


def test_a_dataset_of_negatives_only_loads(dataset_name: str, tempdir: Path):
    """Such a dataset stores its annotation column without a type."""
    negative = {"media": create_image(0, tempdir), "annotation": {"scene": []}}
    dataset = create_dataset(dataset_name, iter([negative]), splits=(1, 0, 0))

    assert schema_of(LuxonisLoader(dataset)[0]).multilabel == set()
