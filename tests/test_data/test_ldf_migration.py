"""Opening a dataset written before the LDF 3.0 bump."""

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.utils.constants import LDF_VERSION

from .utils import (
    create_dataset,
    create_image,
    read_dataset_metadata,
    set_ldf_version,
    write_dataset_metadata,
    write_dataset_rows,
)


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


def as_ldf_2_1(
    dataset: LuxonisDataset, skeletons: dict[str, Any]
) -> LuxonisDataset:
    """Rewrite the dataset as LDF 2.1 stored it, and open it again.

    LDF 2.1 stores the keypoints as lists and the names under
    ``skeletons``.
    """
    df = dataset._load_df_offline(raise_when_empty=True)
    annotations = [
        json.dumps(
            {"keypoints": list(json.loads(annotation)["keypoints"].values())}
        )
        if task_type == "keypoints"
        else annotation
        for task_type, annotation in df.select(
            "task_type", "annotation"
        ).iter_rows()
    ]
    write_dataset_rows(
        dataset,
        df.with_columns(pl.Series("annotation", annotations, dtype=pl.String)),
    )
    stored = read_dataset_metadata(dataset)
    del stored["keypoint_metadata"]
    return write_dataset_metadata(
        dataset, {**stored, "skeletons": skeletons, "ldf_version": "2.1.0"}
    )


def pose_dataset(dataset_name: str, tempdir: Path) -> LuxonisDataset:
    """Create a dataset with one record of three keypoints by position."""

    def generator() -> DatasetIterator:
        yield {
            "file": create_image(0, tempdir),
            "task_name": "pose",
            "annotation": {
                "class": "person",
                "keypoints": {
                    "keypoints": [(0.5, 0.3), (0.4, 0.2, 1), (0.6, 0.2)],
                },
            },
        }

    return create_dataset(dataset_name, generator(), splits={"train": 1.0})


def test_2_x_keypoints_get_the_names_of_the_skeleton(
    dataset_name: str, tempdir: Path
):
    legacy = as_ldf_2_1(
        pose_dataset(dataset_name, tempdir),
        {
            "pose": {
                "labels": ["nose", "left_eye", "right_eye"],
                "edges": [[0, 1], [0, 2]],
            }
        },
    )

    metadata = legacy.get_keypoint_metadata()["pose"]
    stored_rows = pl.read_parquet(legacy._annotations_path / "*.parquet")
    keypoint_row = stored_rows.filter(pl.col("task_type") == "keypoints")

    assert metadata.labels == ["nose", "left_eye", "right_eye"]
    assert metadata.edges == [(0, 1), (0, 2)]
    assert json.loads(keypoint_row["annotation"].item())["keypoints"] == {
        "nose": [0.5, 0.3, 2],
        "left_eye": [0.4, 0.2, 1],
        "right_eye": [0.6, 0.2, 2],
    }
    assert read_dataset_metadata(legacy)["ldf_version"] == "2.1.0"
    labels = LuxonisLoader(legacy, view="train")[0].labels
    assert labels["pose/keypoints"].tolist() == [
        [0.5, 0.3, 2, 0.4, 0.2, 1, 0.6, 0.2, 2]
    ]


def test_a_migrated_2_x_dataset_opens_again_before_a_write(
    dataset_name: str, tempdir: Path
):
    """The second open reads named rows under the old version stamp."""
    legacy = as_ldf_2_1(
        pose_dataset(dataset_name, tempdir),
        {"pose": {"labels": ["nose"], "edges": [[0, 1]]}},
    )

    reopened = LuxonisDataset(legacy.identifier)

    assert read_dataset_metadata(reopened)["ldf_version"] == "2.1.0"
    assert reopened.get_keypoint_metadata() == legacy.get_keypoint_metadata()
    assert reopened.get_keypoint_metadata()["pose"].labels == [
        "nose",
        "1",
        "2",
    ]


@pytest.mark.parametrize(
    ("skeleton", "labels", "warning"),
    [
        (
            {"labels": ["pt", "pt"], "edges": [[0, 1]]},
            ["0", "1", "2"],
            "repeats keypoint names",
        ),
        (
            {"labels": ["nose"], "edges": [[0, 7]]},
            ["nose", "1", "2"],
            "The edge is dropped",
        ),
    ],
)
def test_the_migration_repairs_a_2_x_skeleton(
    dataset_name: str,
    tempdir: Path,
    warnings_log: list[str],
    skeleton: dict[str, Any],
    labels: list[str],
    warning: str,
):
    """Names by position replace repeated names, and a bad edge goes.

    A row with more keypoints than the labels names the others by
    position.
    """
    legacy = as_ldf_2_1(
        pose_dataset(dataset_name, tempdir), {"pose": skeleton}
    )

    assert legacy.get_keypoint_metadata()["pose"].labels == labels
    assert any(warning in message for message in warnings_log)
