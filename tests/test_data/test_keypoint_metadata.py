"""Keypoint metadata that `LuxonisDataset.add` builds from the records."""

import json
from pathlib import Path
from typing import Any

import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.enums import DatasetType

from .utils import create_dataset, create_image, export_and_import


def pose_records(
    tempdir: Path, *keypoints: dict[str, Any] | list[Any], start: int = 0
) -> DatasetIterator:
    """Yield one record of the task ``pose`` for each set of keypoints."""
    for i, named in enumerate(keypoints, start=start):
        yield {
            "file": create_image(i, tempdir),
            "task_name": "pose",
            "annotation": {
                "class": "person",
                "instance_id": 0,
                "keypoints": {"keypoints": named},
            },
        }


def pose_rows(dataset: LuxonisDataset) -> list[list[float]]:
    """Return the keypoint rows of every sample, sorted."""
    return sorted(
        row
        for _, labels in LuxonisLoader(dataset)
        for row in labels["pose/keypoints"].tolist()
    )


NOSE_AND_EYE = {
    "nose": {
        "x": 0.5,
        "y": 0.3,
        "sigma": 0.026,
    },
    "left_eye": {
        "x": 0.4,
        "y": 0.2,
        "state": "occluded",
        "sigma": 0.025,
        "connected_to": "nose",
    },
}


def test_add_builds_the_keypoint_metadata_from_the_records(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(
        dataset_name, pose_records(tempdir, NOSE_AND_EYE), splits=(1, 0, 0)
    )

    metadata = dataset.get_keypoint_metadata()["pose"]

    assert metadata.labels == ["nose", "left_eye"]
    assert metadata.edges == [(0, 1)]
    assert metadata.sigmas == [0.026, 0.025]
    assert dataset.get_n_keypoints() == {"pose": 2}
    assert pose_rows(dataset) == [[0.5, 0.3, 2, 0.4, 0.2, 1]]


def test_records_can_name_different_keypoints(
    dataset_name: str, tempdir: Path
):
    """A keypoint that a record does not name loads as (0, 0, 0)."""
    dataset = create_dataset(
        dataset_name,
        pose_records(
            tempdir,
            {"nose": (0.5, 0.3)},
            {"right_eye": (0.6, 0.2), "nose": (0.4, 0.3)},
        ),
        splits=(1, 0, 0),
    )

    assert dataset.get_keypoint_metadata()["pose"].labels == [
        "nose",
        "right_eye",
    ]
    assert pose_rows(dataset) == [
        [0.4, 0.3, 2, 0.6, 0.2, 2],
        [0.5, 0.3, 2, 0, 0, 0],
    ]


def test_a_later_add_appends_its_new_names(dataset_name: str, tempdir: Path):
    dataset = create_dataset(
        dataset_name, pose_records(tempdir, NOSE_AND_EYE), splits=False
    )

    dataset.add(pose_records(tempdir, {"tail": (0.1, 0.9)}, start=1))
    dataset.make_splits((1, 0, 0))

    assert dataset.get_keypoint_metadata()["pose"].labels == [
        "nose",
        "left_eye",
        "tail",
    ]
    assert pose_rows(dataset) == [
        [0, 0, 0, 0, 0, 0, 0.1, 0.9, 2],
        [0.5, 0.3, 2, 0.4, 0.2, 1, 0, 0, 0],
    ]


def test_a_list_of_keypoints_gets_positional_names(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(
        dataset_name,
        pose_records(tempdir, [(0.5, 0.3), (0.4, 0.2, 1)]),
        splits=(1, 0, 0),
    )

    metadata = dataset.get_keypoint_metadata()["pose"]
    assert metadata.labels == ["0", "1"]
    assert metadata.edges == []


@pytest.mark.parametrize("batch_size", [1, 10])
def test_a_different_sigma_fails_the_add(
    dataset_name: str, tempdir: Path, batch_size: int
):
    dataset = create_dataset(
        dataset_name, pose_records(tempdir, NOSE_AND_EYE), splits=False
    )
    stored = (dataset._metadata_path / "metadata.json").read_text()
    other = {
        "nose": {
            "x": 0.5,
            "y": 0.3,
            "sigma": 0.1,
        }
    }

    with pytest.raises(ValueError, match=r"Task 'pose'.*'nose'"):
        dataset.add(pose_records(tempdir, other, start=1), batch_size)

    assert (dataset._metadata_path / "metadata.json").read_text() == stored


def test_merge_unites_the_keypoint_names(dataset_name: str, tempdir: Path):
    target = create_dataset(
        f"{dataset_name}_target",
        pose_records(tempdir, NOSE_AND_EYE),
        splits=(1, 0, 0),
    )
    other = create_dataset(
        f"{dataset_name}_other",
        pose_records(
            tempdir, {"tail": (0.1, 0.9), "nose": (0.2, 0.2)}, start=1
        ),
        splits=(1, 0, 0),
    )

    merged = target.merge_with(
        other, inplace=False, new_dataset_name=f"{dataset_name}_merged"
    )

    assert merged.get_keypoint_metadata()["pose"].labels == [
        "nose",
        "left_eye",
        "tail",
    ]
    assert pose_rows(merged) == [
        [0.2, 0.2, 2, 0, 0, 0, 0.1, 0.9, 2],
        [0.5, 0.3, 2, 0.4, 0.2, 1, 0, 0, 0],
    ]


def test_a_native_import_keeps_each_keypoint_by_name(
    dataset_name: str, tempdir: Path
):
    """The target orders the names differently, and a row lacks one."""
    source = create_dataset(
        dataset_name,
        pose_records(
            tempdir,
            {"right_eye": (0.6, 0.2), "left_eye": (0.4, 0.2)},
            {"right_eye": (0.7, 0.3)},
        ),
        splits=(1, 0, 0),
    )
    target_name = f"{dataset_name}_target"
    create_dataset(
        target_name,
        pose_records(
            tempdir, {"left_eye": (0.1, 0.1), "right_eye": (0.2, 0.2)}, start=2
        ),
        splits=(1, 0, 0),
    )
    exported = source.export(tempdir / "exported", DatasetType.NATIVE)
    assert isinstance(exported, Path)

    target = LuxonisParser(
        str(exported / source.identifier),
        dataset_type=DatasetType.NATIVE,
        dataset_name=target_name,
        save_dir=tempdir,
    ).parse()

    assert target.get_keypoint_metadata()["pose"].labels == [
        "left_eye",
        "right_eye",
    ]
    assert pose_rows(target) == [
        [0, 0, 0, 0.7, 0.3, 2],
        [0.1, 0.1, 2, 0.2, 0.2, 2],
        [0.4, 0.2, 2, 0.6, 0.2, 2],
    ]


def test_native_export_gives_each_record_every_keypoint_of_the_task(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(
        dataset_name,
        pose_records(tempdir, NOSE_AND_EYE, {"right_eye": (0.6, 0.2)}),
        splits=(1, 0, 0),
    )

    exported = dataset.export(tempdir / "exported", DatasetType.NATIVE)
    assert isinstance(exported, Path)
    records = [
        record
        for path in (exported / dataset.identifier).rglob("annotations.json")
        for record in json.loads(path.read_text())
        if "keypoints" in (record.get("annotation") or {})
    ]

    assert {
        json.dumps(record["annotation"]["keypoints"], sort_keys=True)
        for record in records
    } == {
        json.dumps(
            {
                "keypoints": {
                    "nose": {
                        "x": 0.5,
                        "y": 0.3,
                        "sigma": 0.026,
                    },
                    "left_eye": {
                        "x": 0.4,
                        "y": 0.2,
                        "state": 1,
                        "sigma": 0.025,
                        "connected_to": ["nose"],
                    },
                    "right_eye": {"x": 0.0, "y": 0.0, "state": 0},
                }
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "keypoints": {
                    "nose": {
                        "x": 0.0,
                        "y": 0.0,
                        "state": 0,
                        "sigma": 0.026,
                    },
                    "left_eye": {
                        "x": 0.0,
                        "y": 0.0,
                        "state": 0,
                        "sigma": 0.025,
                        "connected_to": ["nose"],
                    },
                    "right_eye": {"x": 0.6, "y": 0.2},
                }
            },
            sort_keys=True,
        ),
    }


@pytest.mark.parametrize(
    "dataset_type", [DatasetType.NATIVE, DatasetType.COCO]
)
def test_export_and_import_keep_the_keypoint_metadata(
    dataset_name: str, tempdir: Path, dataset_type: DatasetType
):
    dataset = create_dataset(
        dataset_name, pose_records(tempdir, NOSE_AND_EYE), splits=(1, 0, 0)
    )

    imported = export_and_import(dataset, tempdir, dataset_type)

    (metadata,) = imported.get_keypoint_metadata().values()
    original = dataset.get_keypoint_metadata()["pose"]
    assert metadata.labels == original.labels
    assert metadata.edges == original.edges
    assert metadata.sigmas == original.sigmas


def test_a_native_round_trip_keeps_a_task_that_no_row_fills(
    dataset_name: str, tempdir: Path
):
    """Each row names only some keypoints of the task."""
    dataset = create_dataset(
        dataset_name,
        pose_records(
            tempdir,
            {"nose": {"x": 0.5, "y": 0.3, "sigma": 0.026}},
            {"left_eye": {"x": 0.4, "y": 0.2, "sigma": 0.025}},
        ),
        splits=(1, 0, 0),
    )

    imported = export_and_import(dataset, tempdir)

    assert imported.get_keypoint_metadata() == dataset.get_keypoint_metadata()
