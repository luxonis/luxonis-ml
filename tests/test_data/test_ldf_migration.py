"""Opening a dataset written before the LDF 3.0 bump."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.ldf import SCHEMA_METADATA_KEY, DatasetSchema

from .utils import create_dataset, create_image, set_ldf_version


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

    # The migration ran, so the dataset now reports the version it was
    # migrated to. A dataset left on its old stamp would be migrated again
    # on every open, and could never merge with one this version wrote.
    assert reopened.version == LDF_VERSION
    assert reopened.get_classes() == {"vehicles": {"car": 0}}

    labels = LuxonisLoader(reopened, view="train")[0].labels

    assert labels["vehicles/boundingbox"].shape == (1, 5)


def test_a_2_x_row_without_an_instance_id_keeps_its_own_class(
    dataset_name: str, tempdir: Path
):
    """LDF 2.x stored -1 for a detection without an ID.

    The migration leaves those rows as they are. They were paired by their
    position among the rows of one task type, a number every type shares,
    so the box and the road mask both joined the class-only label.
    """
    image = create_image(0, tempdir)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:100] = 1

    def generator() -> DatasetIterator:
        for annotation in [
            {"class": "indoor"},
            {
                "class": "car",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            },
            {"class": "road", "segmentation": {"mask": mask}},
        ]:
            yield {
                "media": image,
                "task_name": "scene",
                "annotation": annotation,
            }

    dataset = create_dataset(dataset_name, generator(), splits={"train": 1.0})
    df = dataset._load_df_offline(raise_when_empty=True)
    for parquet_file in dataset._annotations_path.glob("*.parquet"):
        parquet_file.unlink()
    df.with_columns(
        pl.when(pl.col("instance_id").is_not_null())
        .then(-1)
        .cast(df.schema["instance_id"])
        .alias("instance_id")
    ).write_parquet(dataset._annotations_path / "0000000000.parquet")
    legacy = set_ldf_version(dataset, "2.2.0")

    sample = LuxonisLoader(legacy, view="train")[0]
    schema = DatasetSchema.model_validate(sample.metadata[SCHEMA_METADATA_KEY])
    segmentation = sample.labels["scene/segmentation"]

    assert sample.labels["scene/boundingbox"][:, 0].tolist() == [
        schema.class_id("scene", "car")
    ]
    assert segmentation[schema.class_id("scene", "road")].sum() == mask.sum()
    assert not segmentation[schema.class_id("scene", "indoor")].any()
