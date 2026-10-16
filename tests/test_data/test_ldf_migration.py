"""Opening a dataset written before the LDF 3.0 bump."""

import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.data.utils.ldf_equivalence import ldf_equivalent
from luxonis_ml.data.utils.parquet import ParquetFileManager
from luxonis_ml.utils import environ

from .utils import (
    create_dataset,
    create_image,
    read_dataset_metadata,
    set_ldf_version,
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


def store_as_ldf_2_1(
    dataset: LuxonisDataset, skeletons: dict[str, Any]
) -> None:
    """Rewrite the stored dataset as LDF 2.1 stored it.

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
    (dataset._metadata_path / "metadata.json").write_text(
        json.dumps({**stored, "skeletons": skeletons, "ldf_version": "2.1.0"})
    )


def as_ldf_2_1(
    dataset: LuxonisDataset, skeletons: dict[str, Any]
) -> LuxonisDataset:
    """Store the dataset as LDF 2.1, and open it again."""
    store_as_ldf_2_1(dataset, skeletons)
    return LuxonisDataset(dataset.identifier)


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


FACE = {"pose": {"labels": ["nose", "left_eye", "right_eye"], "edges": []}}
FACE_ROW = [0.5, 0.3, 2, 0.4, 0.2, 1, 0.6, 0.2, 2]


def stored_keypoints(dataset_path: Path) -> list[Any]:
    """Return the stored keypoints of each keypoint row."""
    rows = pl.read_parquet(dataset_path / "annotations" / "*.parquet")
    return [
        json.loads(annotation)["keypoints"]
        for annotation in rows.filter(pl.col("task_type") == "keypoints")[
            "annotation"
        ]
    ]


def test_each_download_names_the_2_x_rows_of_the_bucket(
    dataset_name: str, tempdir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The open and the pulls change only the local copy.

    A local directory stands in for the bucket. The bucket gets the
    named rows with the next `add`.
    """
    dataset = pose_dataset(dataset_name, tempdir)
    store_as_ldf_2_1(dataset, FACE)
    bucket = tempdir / "bucket"
    shutil.copytree(dataset._local_path, bucket)
    shutil.rmtree(dataset._local_path)
    monkeypatch.setattr(environ, "LUXONISML_BUCKET", "bucket")
    monkeypatch.setattr(
        LuxonisDataset, "_construct_url", staticmethod(lambda *_: str(bucket))
    )

    remote = LuxonisDataset(dataset_name, bucket_storage=BucketStorage.GCS)
    labels = LuxonisLoader(remote, view="train")[0].labels

    assert labels["pose/keypoints"].tolist() == [FACE_ROW]
    assert stored_keypoints(bucket) == [
        [[0.5, 0.3, 2], [0.4, 0.2, 1], [0.6, 0.2, 2]]
    ]

    remote.add(
        iter(
            [
                {
                    "file": create_image(1, tempdir),
                    "task_name": "pose",
                    "annotation": {
                        "class": "person",
                        "keypoints": {"keypoints": {"nose": (0.1, 0.1)}},
                    },
                }
            ]
        )
    )

    assert stored_keypoints(bucket) == [
        {
            "nose": [0.5, 0.3, 2],
            "left_eye": [0.4, 0.2, 1],
            "right_eye": [0.6, 0.2, 2],
        },
        {"nose": [0.1, 0.1, 2]},
    ]


def test_an_interrupted_open_keeps_the_2_x_rows(
    dataset_name: str, tempdir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The old shards stay until the rewrite of the rows ends."""
    store_as_ldf_2_1(pose_dataset(dataset_name, tempdir), FACE)

    def interrupt(*_: Any) -> None:
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(ParquetFileManager, "write", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            LuxonisDataset(dataset_name)

    labels = LuxonisLoader(LuxonisDataset(dataset_name), view="train")[
        0
    ].labels
    assert labels["pose/keypoints"].tolist() == [FACE_ROW]


def test_a_2_x_dataset_compares_by_name(dataset_name: str, tempdir: Path):
    """The comparison names the stored list rows, as the open does."""
    current = pose_dataset(f"{dataset_name}_current", tempdir)
    store_as_ldf_2_1(pose_dataset(dataset_name, tempdir), FACE)

    assert ldf_equivalent(dataset_name, current.identifier)




