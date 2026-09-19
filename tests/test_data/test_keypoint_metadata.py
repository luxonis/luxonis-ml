"""Dataset-level tests for keypoint metadata.

`LuxonisDataset.add` moves the task fields of an annotation into the
dataset metadata. These tests cover that move and the compatibility that
it must keep.
"""

import inspect
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.data.datasets.base_dataset import (
    BaseDataset,
    DatasetIterator,
    KeypointPair,
)
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.enums import DatasetType
from luxonis_ml.ldf import KeypointMetadata

from .utils import create_dataset, create_image, set_ldf_version

LABELS = ["nose", "left_eye", "right_eye"]
# The names give an inferred flip pair, but a horizontal flip must not swap
# two cameras.
CAMERAS = ["left_cam", "right_cam", "center"]
REPEATED_LABELS = ["point", "point", "tip"]
NAMED_KEYPOINTS = {
    "nose": (0.5, 0.3, 2),
    "left_eye": (0.4, 0.2, 2),
    "right_eye": (0.6, 0.2, 1),
}


def keypoint_generator(
    tempdir: Path,
    keypoints: Any,
    fields: dict[str, Any] | None = None,
    n: int = 4,
    start: int = 0,
) -> DatasetIterator:
    for i in range(start, start + n):
        annotation: dict[str, Any] = {"keypoints": keypoints}
        annotation.update(fields or {})
        yield {
            "file": str(create_image(i, tempdir)),
            "task_name": "pose",
            "annotation": {"class": "person", "keypoints": annotation},
        }


def positional_generator(
    tempdir: Path, counts: list[int], start: int = 0
) -> DatasetIterator:
    for i, n_keypoints in enumerate(counts, start=start):
        yield from keypoint_generator(
            tempdir,
            [(0.1 * j, 0.1 * j, 2) for j in range(n_keypoints)],
            n=1,
            start=i,
        )


def keypoint_and_box_generator(
    tempdir: Path, n_keypoints: int, start: int = 0
) -> DatasetIterator:
    """Yield a record of unnamed keypoints and a record without keypoints.

    The loader gives the second sample an empty keypoint array, which has
    the width of the stored keypoint count.
    """
    yield from positional_generator(tempdir, [n_keypoints], start=start)
    yield {
        "file": str(create_image(start + 1, tempdir)),
        "task_name": "pose",
        "annotation": {
            "class": "person",
            "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.3},
        },
    }


def detection_generator(tempdir: Path) -> DatasetIterator:
    for i in range(4):
        yield {
            "file": str(create_image(i, tempdir)),
            "task_name": "detection",
            "annotation": {
                "class": "person",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.3},
            },
        }


def named_dataset(
    dataset_name: str, tempdir: Path, **kwargs: Any
) -> LuxonisDataset:
    return create_dataset(
        dataset_name, keypoint_generator(tempdir, NAMED_KEYPOINTS, **kwargs)
    )


def keypoint_payloads(dataset: LuxonisDataset) -> list[str]:
    df = dataset._load_df_offline(raise_when_empty=True)
    return df.filter(df["task_type"] == "keypoints")["annotation"].to_list()


def exported_detections(annotations_path: Path) -> list[dict[str, Any]]:
    """Return every detection of an exported ``annotations.json``.

    LDF 3.0 groups the detections of a record by task name. An export to
    an older version is flat, so the record holds its one detection
    directly under ``annotation``.
    """
    detections = []
    for record in json.loads(annotations_path.read_text()):
        annotation = record.get("annotation") or {}
        if "task_name" in record:
            detections.append(annotation)
        else:
            for task_detections in annotation.values():
                detections.extend(task_detections)
    return detections


def loaded_keypoint_shapes(dataset: LuxonisDataset) -> list[tuple[int, ...]]:
    return sorted(
        labels["pose/keypoints"].shape for _, labels in LuxonisLoader(dataset)
    )


def read_dataset_metadata(dataset: LuxonisDataset) -> dict[str, Any]:
    return json.loads((dataset._metadata_path / "metadata.json").read_text())


def legacy_dataset(
    dataset: LuxonisDataset,
    skeletons: dict[str, dict[str, list[str] | list[list[int]]]],
) -> LuxonisDataset:
    """Open the dataset again as an older luxonis-ml wrote it.

    That version wrote LDF 2.1, which has no flip pairs and no sigmas. It
    stored the keypoint metadata under ``skeletons``.
    """
    metadata_path = dataset._metadata_path / "metadata.json"
    dataset_metadata = json.loads(metadata_path.read_text())
    del dataset_metadata["keypoint_metadata"]
    dataset_metadata["ldf_version"] = "2.1.0"
    dataset_metadata["skeletons"] = skeletons
    metadata_path.write_text(json.dumps(dataset_metadata))
    return LuxonisDataset(dataset.identifier)


def repeated_names_dataset(dataset_name: str, tempdir: Path) -> LuxonisDataset:
    """Open a dataset that an older luxonis-ml wrote.

    That version stored the keypoint names without a check, so a name
    can repeat.
    """
    dataset = create_dataset(
        dataset_name,
        positional_generator(tempdir, [3, 3, 3, 3]),
        splits=(1, 0, 0),
    )
    return legacy_dataset(
        dataset, {"pose": {"labels": REPEATED_LABELS, "edges": [[0, 1]]}}
    )


def dataset_without_flip_pairs(dataset_name: str) -> LuxonisDataset:
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(
        labels=CAMERAS, task="pose", infer_flip_pairs=False
    )
    return dataset


def test_names_are_promoted_to_the_task_metadata(
    dataset_name: str, tempdir: Path
):
    dataset = named_dataset(
        dataset_name,
        tempdir,
        fields={
            "edges": [("nose", "left_eye"), ("nose", "right_eye")],
            "sigmas": [0.026, 0.025, 0.025],
        },
    )

    assert dataset.get_keypoint_metadata() == {
        "pose": KeypointMetadata(
            labels=LABELS,
            edges=[(0, 1), (0, 2)],
            flip_pairs=[(1, 2)],
            sigmas=[0.026, 0.025, 0.025],
        )
    }
    assert dataset.get_n_keypoints() == {"pose": 3}


def test_flip_pairs_are_inferred_from_the_names(
    dataset_name: str, tempdir: Path
):
    dataset = named_dataset(dataset_name, tempdir)

    assert dataset.get_keypoint_metadata()["pose"].flip_pairs == [(1, 2)]


def test_sub_detections_get_their_own_metadata(
    dataset_name: str, tempdir: Path
):
    def generator() -> DatasetIterator:
        for i in range(4):
            yield {
                "file": str(create_image(i, tempdir)),
                "task_name": "person",
                "annotation": {
                    "class": "person",
                    "sub_detections": {
                        "face": {
                            "class": "face",
                            "keypoints": {
                                "keypoints": {
                                    "left_eye": (0.4, 0.2, 2),
                                    "right_eye": (0.6, 0.2, 2),
                                }
                            },
                        }
                    },
                },
            }

    dataset = create_dataset(dataset_name, generator())

    task_keypoints = dataset.get_keypoint_metadata()["person/face"]
    assert task_keypoints.labels == ["left_eye", "right_eye"]
    assert task_keypoints.flip_pairs == [(0, 1)]


def test_disagreeing_records_are_rejected(dataset_name: str, tempdir: Path):
    def generator() -> DatasetIterator:
        yield from keypoint_generator(
            tempdir, {"nose": (0.5, 0.3, 2), "left_eye": (0.4, 0.2, 2)}, n=1
        )
        yield from keypoint_generator(
            tempdir, {"nose": (0.5, 0.3, 2), "right_eye": (0.6, 0.2, 2)}, n=1
        )

    with pytest.raises(ValueError, match="Conflicting keypoint metadata"):
        create_dataset(dataset_name, generator())


def test_records_in_another_key_order_agree_on_the_task_fields(
    dataset_name: str, tempdir: Path
):
    """`add` compared the edges and sigmas of two records by index.

    The second record keys its keypoints in another order, so the same
    edge and the same sigmas have other indices there. The whole `add`
    failed with a conflict.
    """

    def generator() -> DatasetIterator:
        yield from keypoint_generator(
            tempdir,
            {
                "nose": (0.5, 0.3, 2),
                "left_eye": (0.4, 0.2, 2),
                "right_eye": (0.6, 0.2, 1),
            },
            {"edges": [("nose", "left_eye")], "sigmas": [0.026, 0.025, 0.035]},
            n=1,
        )
        yield from keypoint_generator(
            tempdir,
            {
                "right_eye": (0.6, 0.2, 1),
                "left_eye": (0.4, 0.2, 2),
                "nose": (0.5, 0.3, 2),
            },
            {"edges": [("left_eye", "nose")], "sigmas": [0.035, 0.025, 0.026]},
            n=1,
        )

    dataset = create_dataset(dataset_name, generator())

    task_keypoints = dataset.get_keypoint_metadata()["pose"]
    assert task_keypoints.labels == LABELS
    assert task_keypoints.edges == [(0, 1)]
    assert task_keypoints.sigmas == [0.026, 0.025, 0.035]


def test_an_unknown_keypoint_name_is_rejected(
    dataset_name: str, tempdir: Path
):
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(labels=LABELS, task="pose")

    with pytest.raises(ValueError, match="not part of the task"):
        dataset.add(keypoint_generator(tempdir, {"noze": (0.5, 0.3, 2)}))


def test_add_does_not_clobber_explicit_metadata(
    dataset_name: str, tempdir: Path
):
    """`add` used to overwrite every entry with ``"0"``, ``"1"``, ...

    Adding unnamed keypoints to a dataset whose names were set by hand has
    to leave those names alone.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(
        labels=LABELS, edges=[(0, 1), (0, 2)], task="pose"
    )

    dataset.add(
        keypoint_generator(
            tempdir, [(0.5, 0.3, 2), (0.4, 0.2, 2), (0.6, 0.2, 1)]
        )
    )

    task_keypoints = dataset.get_keypoint_metadata()["pose"]
    assert task_keypoints.labels == LABELS
    assert task_keypoints.edges == [(0, 1), (0, 2)]


def test_a_later_add_cannot_reorder_the_stored_labels(
    dataset_name: str, tempdir: Path
):
    """The labels of a second `add` used to replace the stored ones.

    The rows of the second `add` are written in the stored order, so the
    new order renamed every column. The payload and the flip pair prove
    it: the payload keeps the nose in column 0, and the flip pair holds
    indices, so it must still join the two eyes.
    """
    dataset = named_dataset(dataset_name, tempdir)

    dataset.add(
        keypoint_generator(
            tempdir,
            {
                "right_eye": (0.6, 0.2, 1),
                "left_eye": (0.4, 0.2, 2),
                "nose": (0.5, 0.3, 2),
            },
            n=1,
        )
    )

    task_keypoints = dataset.get_keypoint_metadata()["pose"]
    assert task_keypoints.labels == LABELS
    assert task_keypoints.flip_pairs == [(1, 2)]
    assert set(keypoint_payloads(dataset)) == {
        '{"keypoints":[[0.5,0.3,2],[0.4,0.2,2],[0.6,0.2,1]]}'
    }


@pytest.mark.parametrize(
    ("keypoints", "fields", "expected"),
    [
        pytest.param(
            {
                "right_eye": (0.6, 0.2, 1),
                "left_eye": (0.4, 0.2, 2),
                "nose": (0.5, 0.3, 2),
            },
            {"edges": [("nose", "left_eye")], "sigmas": [0.035, 0.025, 0.026]},
            {"edges": [(0, 1)], "sigmas": [0.026, 0.025, 0.035]},
            id="other-order",
        ),
        pytest.param(
            {"left_eye": (0.4, 0.2, 2), "right_eye": (0.6, 0.2, 1)},
            {"edges": [("left_eye", "right_eye")]},
            {"edges": [(1, 2)]},
            id="subset",
        ),
    ],
)
def test_a_later_add_moves_the_task_fields_to_the_stored_order(
    dataset_name: str,
    tempdir: Path,
    keypoints: dict[str, tuple[float, float, int]],
    fields: dict[str, list[tuple[str, str]] | list[float]],
    expected: dict[str, list[tuple[int, int]] | list[float]],
):
    """`add` kept the indices of a record for the stored names.

    The indices of a record point into its own keys. The stored names
    have another order, so the edge joined other keypoints, and each
    keypoint got the sigma of another keypoint.
    """
    dataset = named_dataset(dataset_name, tempdir)

    dataset.add(keypoint_generator(tempdir, keypoints, fields, n=1, start=4))

    assert dataset.get_keypoint_metadata()[
        "pose"
    ] == KeypointMetadata.model_validate(
        {"labels": LABELS, "flip_pairs": [(1, 2)], **expected}
    )


def test_a_later_add_rejects_sigmas_for_a_subset(
    dataset_name: str, tempdir: Path
):
    """`add` stored the sigmas of a record that names a subset.

    The stored entry then had three labels and two sigmas, and a native
    export of the dataset did not import.
    """
    dataset = named_dataset(dataset_name, tempdir)

    with pytest.raises(ValueError, match="2 of the 3 keypoints"):
        dataset.add(
            keypoint_generator(
                tempdir,
                {"nose": (0.5, 0.3, 2), "right_eye": (0.6, 0.2, 1)},
                {"sigmas": [0.1, 0.2]},
                n=1,
                start=4,
            )
        )

    assert dataset.get_keypoint_metadata()["pose"].sigmas == []


def test_records_can_name_different_subsets_of_the_stored_names(
    dataset_name: str, tempdir: Path, warnings_log: list[str]
):
    """`add` merged two records before it used the stored names.

    Each record names a subset of the stored names, and the write pads
    both of them. The two subsets differ, so the whole `add` failed with
    a conflict. The padded rows all have the stored width, so the task
    does not mix rows of different widths either.
    """
    dataset = named_dataset(dataset_name, tempdir)

    def generator() -> DatasetIterator:
        yield from keypoint_generator(
            tempdir, {"left_eye": (0.4, 0.2, 2)}, n=1, start=4
        )
        yield from keypoint_generator(
            tempdir,
            {"nose": (0.5, 0.3, 2), "right_eye": (0.6, 0.2, 1)},
            n=1,
            start=5,
        )

    dataset.add(generator())

    assert dataset.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=LABELS, flip_pairs=[(1, 2)]
    )
    assert {
        '{"keypoints":[[0.0,0.0,0],[0.4,0.2,2],[0.0,0.0,0]]}',
        '{"keypoints":[[0.5,0.3,2],[0.0,0.0,0],[0.6,0.2,1]]}',
    } <= set(keypoint_payloads(dataset))
    assert not any("mixes annotations" in m for m in warnings_log)


def test_a_subset_of_the_stored_names_can_join_a_record_without_names(
    dataset_name: str, tempdir: Path
):
    """`add` checked the declared names before it merged the stored names.

    The record names one of the three stored keypoints, and the other
    record has three keypoints without names. The check compared one name
    with three keypoints and failed after `add` wrote both rows.
    """
    dataset = named_dataset(dataset_name, tempdir)

    def generator() -> DatasetIterator:
        yield from keypoint_generator(
            tempdir, {"left_eye": (0.4, 0.2, 2)}, n=1, start=4
        )
        yield from positional_generator(tempdir, [3], start=5)

    dataset.add(generator())

    assert dataset.get_keypoint_metadata()["pose"].labels == LABELS
    assert {
        '{"keypoints":[[0.0,0.0,0],[0.4,0.2,2],[0.0,0.0,0]]}',
        '{"keypoints":[[0.0,0.0,2],[0.1,0.1,2],[0.2,0.2,2]]}',
    } <= set(keypoint_payloads(dataset))


@pytest.mark.parametrize(
    ("batch_size", "n_written"),
    [
        pytest.param(1_000_000, 0, id="one-batch"),
        pytest.param(1, 1, id="batch-per-record"),
    ],
)
def test_add_checks_the_keypoint_metadata_before_a_batch_is_written(
    dataset_name: str, tempdir: Path, batch_size: int, n_written: int
):
    """`add` checked the keypoint metadata after it wrote every row.

    The check failed, but the rows and the classes were already on disk.
    The tasks and the keypoint metadata were not. A check cannot undo an
    earlier batch. Only the third record breaks the sigmas, so a batch of
    one record keeps the first record.
    """

    def generator() -> DatasetIterator:
        yield from keypoint_generator(
            tempdir, [(0.1, 0.1, 2)] * 3, {"sigmas": [0.1] * 3}, n=1
        )
        yield from positional_generator(tempdir, [3, 4], start=1)

    dataset = LuxonisDataset(dataset_name, delete_local=True)

    with pytest.raises(ValueError, match="3 sigmas"):
        dataset.add(generator(), batch_size=batch_size)

    assert len(dataset) == n_written
    assert not (dataset._metadata_path / "metadata.json").exists()


@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(1_000_000, id="one-batch"),
        pytest.param(4, id="batch-of-every-record"),
    ],
)
def test_add_checks_the_stored_fields_before_the_last_batch_is_written(
    dataset_name: str, tempdir: Path, batch_size: int
):
    """`add` wrote a full batch before it checked the stored fields.

    The stored sigmas cover five keypoints, and each of the four records
    has three. With a batch of four records, the loop wrote every row
    before the check failed. The classes and the tasks stayed as they
    were, next to rows that they do not describe.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(sigmas=[0.1] * 5, task="pose")
    before = read_dataset_metadata(dataset)

    with pytest.raises(ValueError, match="5 sigmas"):
        dataset.add(
            positional_generator(tempdir, [3, 3, 3, 3]), batch_size=batch_size
        )

    assert len(dataset) == 0
    assert read_dataset_metadata(dataset) == before


def test_add_checks_the_declared_fields_against_the_stored_names(
    dataset_name: str, tempdir: Path
):
    """The check before a batch ignored the number of stored names.

    `add` pads each record to the five stored names, so no later record
    can make three sigmas fit. The check before each batch compared the
    sigmas with the three keypoints of the records. A batch of one record
    thus wrote the rows before the last check failed.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(
        labels=[*LABELS, "left_ear", "right_ear"], task="pose"
    )
    before = read_dataset_metadata(dataset)

    with pytest.raises(ValueError, match="3 sigmas, but the annotations"):
        dataset.add(
            keypoint_generator(
                tempdir, [(0.1, 0.1, 2)] * 3, {"sigmas": [0.1] * 3}, n=3
            ),
            batch_size=1,
        )

    assert len(dataset) == 0
    assert read_dataset_metadata(dataset) == before


@pytest.mark.parametrize(
    ("open_dataset", "fields"),
    [
        pytest.param(named_dataset, {}, id="names"),
        pytest.param(
            named_dataset, {"sigmas": [0.1] * 4}, id="names-and-sigmas"
        ),
        pytest.param(repeated_names_dataset, {}, id="repeated-names"),
    ],
)
def test_add_checks_a_record_against_the_stored_names_before_a_batch(
    dataset_name: str,
    tempdir: Path,
    open_dataset: Callable[[str, Path], LuxonisDataset],
    fields: dict[str, list[float]],
):
    """`add` wrote the batch in front of a record wider than the stored names.

    The last record has four keypoints, and the task has three stored
    names. `add` does not change stored names, so no later record can make
    the record fit. The check before a batch ignored a task without a
    declaration, and declared sigmas for four keypoints also passed it.
    `add` thus wrote the first two records before the alignment failed.
    """
    dataset = open_dataset(dataset_name, tempdir)
    before = read_dataset_metadata(dataset)

    def generator() -> DatasetIterator:
        yield from positional_generator(tempdir, [3, 3], start=4)
        yield from keypoint_generator(
            tempdir, [(0.1, 0.1, 2)] * 4, fields, n=1, start=6
        )

    with pytest.raises(ValueError, match="task defines only 3"):
        dataset.add(generator(), batch_size=2)

    assert len(dataset) == 4
    assert read_dataset_metadata(dataset) == before


def test_add_checks_the_names_against_the_rows_of_an_earlier_batch(
    dataset_name: str, tempdir: Path
):
    """The names of a task must cover the widest record of `add`.

    The first batch writes two rows of five keypoints without names. The
    last record names three keypoints. `add` must reject the names before
    it writes the last batch.
    """

    def generator() -> DatasetIterator:
        yield from positional_generator(tempdir, [5, 5, 5])
        yield from keypoint_generator(tempdir, NAMED_KEYPOINTS, n=1, start=3)

    dataset = LuxonisDataset(dataset_name, delete_local=True)

    with pytest.raises(ValueError, match="3 labels, but the annotations"):
        dataset.add(generator(), batch_size=2)

    assert len(dataset) == 2


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(None, id="placeholder-names"),
        pytest.param([], id="no-labels"),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(1_000_000, id="one-batch"),
        pytest.param(1, id="batch-per-record"),
    ],
)
def test_add_checks_the_names_against_the_rows_of_an_earlier_add(
    dataset_name: str,
    tempdir: Path,
    batch_size: int,
    labels: list[str] | None,
):
    """`add` accepted names for fewer keypoints than an earlier `add` wrote.

    The first `add` writes rows of five keypoints without names. The
    second `add` stored three names for the task, and the loader then gave
    the samples keypoint arrays of different widths. The same records in
    one `add` fail, so the second `add` must fail too. A batch of one
    record must fail before it writes a row. Cleared labels do not change
    the rows, so the names must fail there too.
    """
    dataset = create_dataset(
        dataset_name, positional_generator(tempdir, [5, 5])
    )
    dataset.set_keypoint_metadata(labels=labels, edges=[(0, 1)], task="pose")
    before = read_dataset_metadata(dataset)

    with pytest.raises(ValueError, match="3 labels, but the annotations"):
        dataset.add(
            keypoint_generator(tempdir, NAMED_KEYPOINTS, n=2, start=2),
            batch_size=batch_size,
        )

    assert len(dataset) == 2
    assert read_dataset_metadata(dataset) == before


def test_a_small_batch_accepts_what_a_later_record_completes(
    dataset_name: str, tempdir: Path
):
    """A check before a batch sees only the records up to that batch.

    The stored sigmas cover five keypoints, but the first record has only
    three. The second record completes the task. A batch of one record
    must thus accept the same `add` as one batch does.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(sigmas=[0.1] * 5, task="pose")

    dataset.add(positional_generator(tempdir, [3, 5]), batch_size=1)

    assert dataset.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=["0", "1", "2", "3", "4"], sigmas=[0.1] * 5
    )


def test_a_changed_task_field_warns_once_for_each_add(
    dataset_name: str, tempdir: Path, warnings_log: list[str]
):
    """`add` checks the keypoint metadata before each batch.

    Only the last step of `add` warns about a changed field, so a small
    batch size does not repeat the warning.
    """
    dataset = named_dataset(
        dataset_name, tempdir, fields={"sigmas": [0.1, 0.2, 0.3]}
    )

    dataset.add(
        keypoint_generator(
            tempdir, NAMED_KEYPOINTS, {"sigmas": [0.3, 0.2, 0.1]}, start=4
        ),
        batch_size=1,
    )

    assert dataset.get_keypoint_metadata()["pose"].sigmas == [0.3, 0.2, 0.1]
    assert sum("different `sigmas`" in m for m in warnings_log) == 1


def test_a_failed_add_keeps_the_rows_of_re_added_files(
    dataset_name: str, tempdir: Path
):
    """`add` removed the old rows of a file before it built the new rows.

    The second record has more keypoints than the task, so the alignment
    of its keypoints raised. The old rows of both files were already gone.
    `add` still wrote the rows of the first record and the bounding box of
    the second record. A failed batch must change no row.
    """
    dataset = named_dataset(dataset_name, tempdir)
    before = dataset._load_df_offline(raise_when_empty=True)

    def generator() -> DatasetIterator:
        yield from positional_generator(tempdir, [3])
        yield {
            "file": str(create_image(1, tempdir)),
            "task_name": "pose",
            "annotation": {
                "class": "person",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.3},
                "keypoints": {"keypoints": [(0.1, 0.1, 2)] * 5},
            },
        }

    with pytest.raises(ValueError, match="task defines only 3"):
        dataset.add(generator())

    assert dataset._load_df_offline(raise_when_empty=True).equals(before)


def test_a_later_add_updates_the_placeholder_count(
    dataset_name: str, tempdir: Path
):
    """The stored count used to freeze at what the first `add` saw.

    A record of unnamed keypoints declares nothing, so the guard that
    protects an explicit definition also skipped the placeholder that
    `add` wrote itself. The count then contradicts the rows on disk, and
    `LuxonisLoader` sizes an empty keypoint label to the old width.
    """
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, [(0.1, 0.1, 2), (0.2, 0.2, 2)], n=1),
    )
    assert dataset.get_n_keypoints() == {"pose": 2}

    dataset.add(
        keypoint_generator(
            tempdir, [(0.1, 0.1, 2), (0.2, 0.2, 2), (0.3, 0.3, 2)], n=1
        )
    )

    assert dataset.get_n_keypoints() == {"pose": 3}
    assert dataset.get_keypoint_metadata()["pose"].labels == ["0", "1", "2"]


def test_an_edges_only_declaration_still_names_the_keypoints(
    dataset_name: str, tempdir: Path
):
    """A declaration of edges alone used to store ``labels=[]``.

    The stored entry then held no count, so `get_n_keypoints` read it off
    the highest edge index. A task with five keypoints reported two, and
    the loader padded a keypoint-free sample to that width. The names
    keep the count, so the assertion on them guards the count too.
    """
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(
            tempdir,
            [(0.1 * i, 0.2, 2) for i in range(5)],
            fields={"edges": [(0, 1)]},
        ),
    )

    task_keypoints = dataset.get_keypoint_metadata()["pose"]
    assert task_keypoints.labels == ["0", "1", "2", "3", "4"]
    assert task_keypoints.edges == [(0, 1)]
    assert dataset.get_n_keypoints() == {"pose": 5}


@pytest.mark.parametrize(
    ("edges", "sigmas"),
    [
        pytest.param([(0, 1)], None, id="edges"),
        pytest.param(None, [0.1] * 5, id="sigmas"),
    ],
)
def test_a_stored_entry_without_labels_gets_the_keypoint_count(
    dataset_name: str,
    tempdir: Path,
    edges: list[KeypointPair] | None,
    sigmas: list[float] | None,
):
    """`add` skipped a stored entry that had no labels.

    Only the labels carry the keypoint count. `get_n_keypoints` then read
    the count off the edges, or found no count, and `LuxonisLoader` gave
    a sample without keypoints a narrower array than the other samples.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(edges=edges, sigmas=sigmas, task="pose")

    dataset.add(keypoint_and_box_generator(tempdir, 5))
    dataset.make_splits((1, 0, 0))

    assert dataset.get_keypoint_metadata()[
        "pose"
    ] == KeypointMetadata.model_validate(
        {
            "labels": ["0", "1", "2", "3", "4"],
            "edges": edges or [],
            "sigmas": sigmas or [],
        }
    )
    assert loaded_keypoint_shapes(dataset) == [(0, 15), (1, 15)]


def test_an_empty_stored_entry_gets_the_placeholder(
    dataset_name: str, tempdir: Path
):
    """`add` kept the empty edges of an empty stored entry.

    An empty entry describes no keypoints, and `_write_metadata` can leave
    it out of the file. A reopened dataset thus got the chain edges, but
    the same handle did not.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(edges=[], task="pose")

    dataset.add(positional_generator(tempdir, [3]))

    assert dataset.get_keypoint_metadata() == {
        "pose": KeypointMetadata(
            labels=["0", "1", "2"], edges=[(0, 1), (1, 2)]
        )
    }


@pytest.mark.parametrize(
    ("edges", "expected_edges"),
    [
        pytest.param([(0, 1)], [(0, 1)], id="edges"),
        pytest.param([], [(i, i + 1) for i in range(6)], id="empty-entry"),
    ],
)
def test_a_task_without_labels_keeps_the_width_of_its_rows(
    dataset_name: str,
    tempdir: Path,
    edges: list[KeypointPair],
    expected_edges: list[tuple[int, int]],
):
    """`add` took the keypoint count of a task without labels from the records.

    The rows of the task have seven keypoints, but the task has no labels.
    An older luxonis-ml also stored such a task. The records of the next
    `add` have five keypoints, and the task got five labels. The loader
    thus gave a sample without keypoints a narrower array than the stored
    rows.
    """
    dataset = create_dataset(
        dataset_name, keypoint_and_box_generator(tempdir, 7), splits=False
    )
    dataset.set_keypoint_metadata(labels=[], edges=edges, task="pose")

    dataset.add(keypoint_and_box_generator(tempdir, 5, start=2))
    dataset.make_splits((1, 0, 0))

    assert dataset.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=[str(i) for i in range(7)], edges=expected_edges
    )
    assert loaded_keypoint_shapes(dataset) == [(0, 21)] * 2 + [(1, 21)] * 2


@pytest.mark.parametrize(
    ("edges", "flip_pairs", "expected_edges"),
    [
        pytest.param([(0, 1)], None, [(0, 1)], id="edges"),
        pytest.param(
            None,
            [(3, 4)],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
            id="flip-pairs",
        ),
    ],
)
def test_a_wider_add_widens_placeholder_names_with_a_task_field(
    dataset_name: str,
    tempdir: Path,
    edges: list[KeypointPair] | None,
    flip_pairs: list[KeypointPair] | None,
    expected_edges: list[tuple[int, int]],
):
    """`add` widened only a stored entry that held nothing but placeholders.

    Here the entry has placeholder names and a field that the user set.
    The second `add` has seven keypoints, but the entry kept five names.
    `get_n_keypoints` thus returned 5, and `LuxonisLoader` gave a sample
    without keypoints a narrower array than the other samples. The chain
    edges that `add` generated have to grow with the names.
    """
    dataset = create_dataset(
        dataset_name, keypoint_and_box_generator(tempdir, 5), splits=False
    )
    dataset.set_keypoint_metadata(
        edges=edges, flip_pairs=flip_pairs, task="pose"
    )
    dataset.add(keypoint_and_box_generator(tempdir, 7, start=2))
    dataset.make_splits((1, 0, 0))

    assert dataset.get_keypoint_metadata()[
        "pose"
    ] == KeypointMetadata.model_validate(
        {
            "labels": ["0", "1", "2", "3", "4", "5", "6"],
            "edges": expected_edges,
            "flip_pairs": flip_pairs or [],
        }
    )
    assert dataset.get_n_keypoints() == {"pose": 7}
    assert loaded_keypoint_shapes(dataset) == [(0, 21)] * 2 + [(1, 21)] * 2


def test_a_wider_add_checks_the_stored_sigmas(
    dataset_name: str, tempdir: Path
):
    """`add` did not check the sigmas of placeholder names.

    The sigmas cover five keypoints. A first `add` of seven keypoints
    raised, but the same records in a second `add` passed. The entry then
    held five sigmas for rows of seven keypoints.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(sigmas=[0.1] * 5, task="pose")
    dataset.add(positional_generator(tempdir, [5]))

    with pytest.raises(ValueError, match="5 sigmas"):
        dataset.add(positional_generator(tempdir, [7], start=1))

    assert len(dataset) == 1
    assert dataset.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=["0", "1", "2", "3", "4"], sigmas=[0.1] * 5
    )


@pytest.mark.parametrize(
    "keypoints",
    [
        pytest.param(list(NAMED_KEYPOINTS.values()), id="list"),
        pytest.param(NAMED_KEYPOINTS, id="names"),
    ],
)
def test_stored_sigmas_must_match_the_keypoint_count(
    dataset_name: str,
    tempdir: Path,
    keypoints: list[tuple[float, float, int]]
    | dict[str, tuple[float, float, int]],
):
    """`add` kept stored sigmas for another number of keypoints.

    The stored entry then held five sigmas for three keypoints, and
    nothing reported the difference.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(sigmas=[0.1] * 5, task="pose")

    with pytest.raises(ValueError, match="5 sigmas"):
        dataset.add(keypoint_generator(tempdir, keypoints))


def test_new_names_drop_the_placeholder_edges(
    dataset_name: str, tempdir: Path
):
    """`add` gives keypoints without names invented chain edges.

    New names kept those edges. A COCO export then wrote them as the
    skeleton, and a visualization drew lines between unrelated
    keypoints.
    """
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, list(NAMED_KEYPOINTS.values())),
    )

    dataset.set_keypoint_metadata(labels=LABELS, task="pose")

    assert dataset.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=LABELS, flip_pairs=[(1, 2)]
    )


@pytest.mark.parametrize("n_placeholders", [2, 3])
def test_a_named_add_drops_the_placeholder_edges(
    dataset_name: str,
    tempdir: Path,
    warnings_log: list[str],
    n_placeholders: int,
):
    """A named `add` copied the invented chain edges into its entry.

    The check for placeholder values used the keypoint count of the new
    `add`, and not the count of the stored entry. After two unnamed
    keypoints, the stored entry did not match the placeholder of three
    keypoints. The entry thus kept the chain edge, and a warning reported
    the placeholder labels as a conflict.
    """
    dataset = create_dataset(
        dataset_name, positional_generator(tempdir, [n_placeholders])
    )

    dataset.add(keypoint_generator(tempdir, NAMED_KEYPOINTS, n=1, start=1))

    task_keypoints = dataset.get_keypoint_metadata()["pose"]
    assert task_keypoints.labels == LABELS
    assert task_keypoints.edges == []
    assert not any("describe a different" in m for m in warnings_log)


@pytest.mark.parametrize(
    ("first", "second", "fields"),
    [
        pytest.param(3, 5, {"sigmas": [0.1] * 5}, id="wider-sigmas"),
        pytest.param(3, 5, {"flip_pairs": [(3, 4)]}, id="wider-flip-pairs"),
        pytest.param(5, 3, {"flip_pairs": [(0, 1)]}, id="narrower"),
    ],
)
def test_an_add_with_task_fields_sizes_the_placeholder_to_the_widest_row(
    dataset_name: str,
    tempdir: Path,
    first: int,
    second: int,
    fields: dict[str, list[float] | list[tuple[int, int]]],
):
    """`add` sized only the placeholder labels to the widest row.

    The records of the second `add` give a task field. After a wider
    `add`, the entry got five labels, but the chain edges of three
    keypoints. Those edges are not the chain of five keypoints, so new
    names kept them. After a narrower `add`, the entry must keep five
    labels, because the rows of the first `add` have five keypoints.
    """
    dataset = create_dataset(
        dataset_name, positional_generator(tempdir, [first])
    )

    dataset.add(
        keypoint_generator(
            tempdir, [(0.1, 0.1, 2)] * second, fields, n=1, start=1
        )
    )

    assert dataset.get_keypoint_metadata()[
        "pose"
    ] == KeypointMetadata.model_validate(
        {
            "labels": ["0", "1", "2", "3", "4"],
            "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
            **fields,
        }
    )

    dataset.set_keypoint_metadata(
        labels=[*LABELS, "left_ear", "right_ear"], task="pose"
    )

    assert dataset.get_keypoint_metadata()["pose"].edges == []


def test_a_named_add_keeps_the_edges_of_stored_names(
    dataset_name: str, tempdir: Path
):
    """Only keypoints without names have generated edges.

    The stored edges join named keypoints, and they have the form of a
    chain. A named `add` without edges must keep them.
    """
    dataset = named_dataset(
        dataset_name,
        tempdir,
        fields={"edges": [("nose", "left_eye"), ("left_eye", "right_eye")]},
    )

    dataset.add(keypoint_generator(tempdir, NAMED_KEYPOINTS, n=1, start=4))

    assert dataset.get_keypoint_metadata()["pose"].edges == [(0, 1), (1, 2)]


@pytest.mark.parametrize(
    ("fields", "edges", "expected"),
    [
        pytest.param({"edges": [(0, 2)]}, None, [(0, 2)], id="stored"),
        pytest.param({}, [(0, 1), (1, 2)], [(0, 1), (1, 2)], id="given"),
    ],
)
def test_new_names_keep_the_edges_that_add_did_not_generate(
    dataset_name: str,
    tempdir: Path,
    fields: dict[str, list[tuple[int, int]]],
    edges: list[KeypointPair] | None,
    expected: list[tuple[int, int]],
):
    """New names drop only the chain edges that `add` generated.

    The records can give edges for keypoints without names. Those edges
    describe the keypoints, so new names keep them. A chain that the call
    gives with the names stays too.
    """
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, list(NAMED_KEYPOINTS.values()), fields),
    )

    dataset.set_keypoint_metadata(labels=LABELS, edges=edges, task="pose")

    assert dataset.get_keypoint_metadata()["pose"].edges == expected


def test_positional_names_do_not_clash_with_real_ones(
    dataset_name: str, tempdir: Path
):
    """``"0"``, ``"1"``, ... are a fallback, not something the record chose."""
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(
            tempdir, [(0.5, 0.3, 2), (0.4, 0.2, 2), (0.6, 0.2, 1)]
        ),
    )
    assert dataset.get_keypoint_metadata()["pose"].labels == ["0", "1", "2"]

    dataset.set_keypoint_metadata(labels=LABELS, task="pose")
    dataset.add(
        keypoint_generator(
            tempdir, [(0.1, 0.1, 2), (0.2, 0.2, 2), (0.3, 0.3, 2)]
        )
    )

    assert dataset.get_keypoint_metadata()["pose"].labels == LABELS


def test_placeholders_are_still_generated(dataset_name: str, tempdir: Path):
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, [(0.5, 0.3, 2), (0.4, 0.2, 2)]),
    )

    assert dataset.get_keypoint_metadata() == {
        "pose": KeypointMetadata(labels=["0", "1"], edges=[(0, 1)])
    }


def test_the_task_fields_are_not_repeated_on_every_row(
    dataset_name: str, tempdir: Path
):
    dataset = named_dataset(
        dataset_name,
        tempdir,
        fields={"edges": [("nose", "left_eye")], "sigmas": [0.1, 0.2, 0.3]},
    )

    payloads = keypoint_payloads(dataset)

    assert payloads
    for payload in payloads:
        assert json.loads(payload) == {
            "keypoints": [[0.5, 0.3, 2], [0.4, 0.2, 2], [0.6, 0.2, 1]]
        }


def test_records_are_stored_in_task_order(dataset_name: str, tempdir: Path):
    """Column position is keypoint identity across the whole task."""

    def generator() -> DatasetIterator:
        yield from keypoint_generator(
            tempdir,
            {"nose": (0.1, 0.1, 2), "left_eye": (0.2, 0.2, 2)},
            n=1,
        )
        yield from keypoint_generator(
            tempdir,
            {"left_eye": (0.2, 0.2, 2), "nose": (0.1, 0.1, 2)},
            n=1,
        )

    dataset = create_dataset(dataset_name, generator())

    assert set(keypoint_payloads(dataset)) == {
        '{"keypoints":[[0.1,0.1,2],[0.2,0.2,2]]}'
    }


def test_omitted_keypoints_are_padded_on_disk(
    dataset_name: str, tempdir: Path
):
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_keypoint_metadata(labels=LABELS, task="pose")
    dataset.add(keypoint_generator(tempdir, {"left_eye": (0.4, 0.2, 2)}))

    assert json.loads(keypoint_payloads(dataset)[0]) == {
        "keypoints": [[0.0, 0.0, 0], [0.4, 0.2, 2], [0.0, 0.0, 0]]
    }


def test_opening_a_dataset_does_not_materialize_flip_pairs(
    dataset_name: str, tempdir: Path
):
    """A reopen must leave the stored file byte for byte alone.

    Every open revalidates `Metadata`. Inference there would add flip
    pairs that an older ``luxonis-ml`` cannot read.
    """
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, [(0.5, 0.3, 2), (0.4, 0.2, 2)]),
    )
    metadata_path = dataset._metadata_path / "metadata.json"
    before = metadata_path.read_text()
    assert json.loads(before)["keypoint_metadata"]["pose"]["flip_pairs"] == []

    reopened = LuxonisDataset(dataset_name)

    assert reopened.get_keypoint_metadata()["pose"].flip_pairs == []
    assert metadata_path.read_text() == before


def test_the_stored_metadata_holds_every_field(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, [(0.5, 0.3, 2), (0.4, 0.2, 2)]),
    )

    assert read_dataset_metadata(dataset)["keypoint_metadata"] == {
        "pose": {
            "labels": ["0", "1"],
            "edges": [[0, 1]],
            "flip_pairs": [],
            "sigmas": [],
        }
    }


def test_a_legacy_dataset_still_loads(dataset_name: str, tempdir: Path):
    """Written under the old key, and before flip pairs and sigmas."""
    dataset = named_dataset(dataset_name, tempdir)
    metadata_path = dataset._metadata_path / "metadata.json"
    dataset_metadata = json.loads(metadata_path.read_text())
    dataset_metadata["skeletons"] = dataset_metadata.pop("keypoint_metadata")
    for entry in dataset_metadata["skeletons"].values():
        entry.pop("flip_pairs", None)
        entry.pop("sigmas", None)
    metadata_path.write_text(json.dumps(dataset_metadata))

    reopened = LuxonisDataset(dataset_name)

    assert reopened.get_keypoint_metadata()["pose"].labels == LABELS
    assert reopened.get_keypoint_metadata()["pose"].flip_pairs == []
    _, labels = LuxonisLoader(reopened)[0]
    assert labels["pose/keypoints"].shape[1] == 9


@pytest.mark.parametrize(
    ("labels", "keypoints"),
    [
        pytest.param(LABELS, {"left_eye": (0.4, 0.2, 2)}, id="names"),
        pytest.param([], list(NAMED_KEYPOINTS.values()), id="no-names"),
    ],
)
def test_a_legacy_edge_out_of_range_does_not_stop_an_add(
    dataset_name: str,
    tempdir: Path,
    labels: list[str],
    keypoints: list[tuple[float, float, int]]
    | dict[str, tuple[float, float, int]],
):
    """`add` checked the stored edges against the keypoint count.

    An older luxonis-ml stored the edges without a check, and the dataset
    still opens. The next `add` failed, and the error blamed the
    annotations. The drawing code skips an edge out of range.
    """
    dataset = legacy_dataset(
        named_dataset(dataset_name, tempdir),
        {"pose": {"labels": labels, "edges": [[-1, 0], [1, 3]]}},
    )

    dataset.add(keypoint_generator(tempdir, keypoints, n=1, start=4))

    assert dataset.get_keypoint_metadata()["pose"].edges == [(-1, 0), (1, 3)]
    assert dataset.get_n_keypoints() == {"pose": 3}


def test_the_loader_names_the_keypoints(dataset_name: str, tempdir: Path):
    dataset = named_dataset(dataset_name, tempdir)
    loader = LuxonisLoader(dataset)

    _, labels = loader[0]

    assert labels["pose/keypoints"].shape == (1, 9)
    assert loader.get_keypoint_metadata()["pose"].labels == LABELS


def test_the_loader_builds_no_keypoint_metadata_for_each_row(
    dataset_name: str, tempdir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The loader keyed each stored row by the names of the task.

    Each loaded row thus built a `KeypointMetadata` from the names and
    checked it against the row. A stored row has no edges, no flip pairs
    and no sigmas, so the check found nothing. The build took almost all of
    the extra load time, so the test counts the builds.
    """
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, NAMED_KEYPOINTS),
        splits=(1, 0, 0),
    )
    loader = LuxonisLoader(dataset)
    built: list[KeypointMetadata] = []

    def spy(self: KeypointMetadata, **data: object) -> None:
        BaseModel.__init__(self, **data)
        built.append(self)

    monkeypatch.setattr(KeypointMetadata, "__init__", spy)

    assert [labels["pose/keypoints"].tolist() for _, labels in loader] == [
        [[0.5, 0.3, 2.0, 0.4, 0.2, 2.0, 0.6, 0.2, 1.0]]
    ] * 4
    assert built == []


def test_set_keypoint_metadata_updates_only_what_it_is_given(
    dataset_name: str, tempdir: Path
):
    """It used to replace the whole entry, so one field wiped the rest.

    It now has four fields, which makes that unacceptable.
    """
    dataset = named_dataset(dataset_name, tempdir)

    dataset.set_keypoint_metadata(sigmas=[0.1, 0.2, 0.3], task="pose")

    task_keypoints = dataset.get_keypoint_metadata()["pose"]
    assert task_keypoints.labels == LABELS
    assert task_keypoints.sigmas == [0.1, 0.2, 0.3]
    assert task_keypoints.flip_pairs == [(1, 2)]


def test_new_labels_drop_the_indices_they_invalidate(
    dataset_name: str, tempdir: Path
):
    """A relabel used to keep the stored edges, flip pairs and sigmas.

    All three address a keypoint by its position. New labels put a
    different keypoint at each position, so the stored values then
    describe the wrong keypoints. Nothing raises, because every index
    stays in range. The kept flip pair ``(1, 2)`` flips ``right_eye``
    onto ``nose``.
    """
    dataset = named_dataset(
        dataset_name,
        tempdir,
        fields={
            "edges": [("nose", "left_eye")],
            "sigmas": [0.026, 0.025, 0.025],
        },
    )
    assert dataset.get_keypoint_metadata()["pose"].flip_pairs == [(1, 2)]

    dataset.set_keypoint_metadata(
        labels=["left_eye", "right_eye", "nose"], task="pose"
    )

    task_keypoints = dataset.get_keypoint_metadata()["pose"]
    assert task_keypoints.flip_pairs == [(0, 1)]
    assert task_keypoints.edges == []
    assert task_keypoints.sigmas == []


def test_a_keypointless_dataset_omits_the_new_key(
    dataset_name: str, tempdir: Path
):
    """LDF 2.2 renamed the stored ``skeletons`` to ``keypoint_metadata``.

    `Metadata` forbids extra fields, so the new key alone stops an older
    ``luxonis-ml`` from opening the dataset. A dataset without keypoints
    holds nothing that LDF 2.2 added, so it must not carry the key. Every
    write path rewrote it, even for a plain detection dataset.
    """
    dataset = create_dataset(dataset_name, detection_generator(tempdir))

    assert "keypoint_metadata" not in read_dataset_metadata(dataset)
    assert LuxonisDataset(dataset_name).get_keypoint_metadata() == {}


def test_empty_keypoint_metadata_omits_the_new_key(
    dataset_name: str, tempdir: Path
):
    """An entry without values describes no keypoints.

    Empty lists still stored an entry for the detection task. The key
    was left out only for a dict without entries, so the dataset got the
    key, and an older luxonis-ml refused to open it.
    """
    dataset = create_dataset(dataset_name, detection_generator(tempdir))

    dataset.set_keypoint_metadata(labels=[], edges=[])

    assert "keypoint_metadata" not in read_dataset_metadata(dataset)
    assert LuxonisDataset(dataset_name).get_keypoint_metadata() == {}


def test_the_new_key_stamps_the_current_ldf_version(
    dataset_name: str, tempdir: Path
):
    """No write updated the stored LDF version.

    The `add` gave a dataset of LDF 2.1 the new ``keypoint_metadata`` key,
    which LDF 2.1 cannot read. The dataset still claimed LDF 2.1.
    """
    dataset = set_ldf_version(
        create_dataset(dataset_name, detection_generator(tempdir)), "2.1.0"
    )

    dataset.add(keypoint_generator(tempdir, NAMED_KEYPOINTS, start=4))

    stored = read_dataset_metadata(dataset)
    assert "keypoint_metadata" in stored
    assert stored["ldf_version"] == str(LDF_VERSION)
    assert dataset.version == LDF_VERSION


def test_a_legacy_dataset_merges_with_a_new_dataset(
    dataset_name: str, tempdir: Path
):
    """The merge compared the LDF versions as strings.

    A dataset of LDF 2.1 thus did not merge with a new dataset. The merged
    file stores the keypoint metadata under the new key, so it needs the
    current LDF version.
    """
    old = legacy_dataset(
        named_dataset(f"{dataset_name}_old", tempdir),
        {"pose": {"labels": LABELS}},
    )
    new = named_dataset(
        f"{dataset_name}_new",
        tempdir,
        fields={"sigmas": [0.1, 0.2, 0.3]},
        start=4,
    )

    old.merge_with(new)

    stored = read_dataset_metadata(old)
    assert "skeletons" not in stored
    assert stored["ldf_version"] == str(LDF_VERSION)
    assert len(old) == 8
    assert LuxonisDataset(old.identifier).get_keypoint_metadata() == {
        "pose": KeypointMetadata(
            labels=LABELS, flip_pairs=[(1, 2)], sigmas=[0.1, 0.2, 0.3]
        )
    }


def test_set_keypoint_metadata_accepts_names(dataset_name: str, tempdir: Path):
    dataset = named_dataset(dataset_name, tempdir)

    dataset.set_keypoint_metadata(
        labels=LABELS,
        edges=[("nose", "left_eye")],
        flip_pairs=[("left_eye", "right_eye")],
        task="pose",
    )

    task_keypoints = dataset.get_keypoint_metadata()["pose"]
    assert task_keypoints.edges == [(0, 1)]
    assert task_keypoints.flip_pairs == [(1, 2)]


def test_the_deprecated_skeleton_aliases_still_forward(
    dataset_name: str, tempdir: Path
):
    """Nothing else calls them, so they need a test of their own."""
    dataset = named_dataset(dataset_name, tempdir)

    with pytest.deprecated_call():
        dataset.set_skeletons(sigmas=[0.1, 0.2, 0.3], task="pose")
    with pytest.deprecated_call():
        skeletons = dataset.get_skeletons()

    assert skeletons == dataset.get_keypoint_metadata()
    assert skeletons["pose"].sigmas == [0.1, 0.2, 0.3]


def test_flip_pair_inference_can_be_turned_off(
    dataset_name: str, tempdir: Path
):
    dataset = named_dataset(dataset_name, tempdir)

    dataset.set_keypoint_metadata(
        labels=LABELS, task="pose", infer_flip_pairs=False
    )

    # The `add` already inferred them. A fresh dataset below shows that
    # the flag keeps them away.
    assert dataset.get_keypoint_metadata()["pose"].flip_pairs == [(1, 2)]

    fresh = LuxonisDataset(f"{dataset_name}_fresh", delete_local=True)
    fresh.set_tasks({"pose": ["keypoints"]})
    fresh.set_keypoint_metadata(
        labels=LABELS, task="pose", infer_flip_pairs=False
    )

    assert fresh.get_keypoint_metadata()["pose"].flip_pairs == []


def test_a_later_add_keeps_the_flip_pairs_turned_off(
    dataset_name: str, tempdir: Path
):
    """`add` inferred flip pairs for every entry without flip pairs.

    The stored entry does not record that the inference is off. A later
    `add` of named records thus paired the two cameras, and a horizontal
    flip swapped two keypoints that must not swap.
    """
    dataset = dataset_without_flip_pairs(dataset_name)

    dataset.add(
        keypoint_generator(tempdir, dict.fromkeys(CAMERAS, (0.5, 0.5, 2)))
    )

    assert dataset.get_keypoint_metadata()["pose"].flip_pairs == []


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(None, id="sigmas-only"),
        pytest.param(CAMERAS, id="same-names"),
    ],
)
def test_a_later_set_keeps_the_flip_pairs_turned_off(
    dataset_name: str, labels: list[str] | None
):
    """`set_keypoint_metadata` inferred flip pairs on each call.

    A call that set only the sigmas thus paired the two cameras. A call
    that gave the same names did too. Each parser gives the names again
    after its `add`, so an import into the dataset paired them.
    """
    dataset = dataset_without_flip_pairs(dataset_name)

    dataset.set_keypoint_metadata(
        labels=labels, sigmas=[0.1, 0.2, 0.3], task="pose"
    )

    assert dataset.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=CAMERAS, sigmas=[0.1, 0.2, 0.3]
    )


@pytest.mark.parametrize(
    ("infer_flip_pairs", "flip_pairs"),
    [
        pytest.param(None, [], id="none"),
        pytest.param(True, [(1, 2)], id="true"),
    ],
)
def test_only_an_explicit_flag_infers_flip_pairs_for_stored_names(
    dataset_name: str,
    tempdir: Path,
    infer_flip_pairs: bool | None,
    flip_pairs: list[tuple[int, int]],
):
    """An older luxonis-ml stored the names without flip pairs.

    The empty list can also mean that the inference is off. The default
    ``None`` thus infers flip pairs only for names that are new to the
    task. ``True`` infers them for the stored names too.
    """
    dataset = legacy_dataset(
        named_dataset(dataset_name, tempdir), {"pose": {"labels": LABELS}}
    )

    dataset.set_keypoint_metadata(
        labels=LABELS, task="pose", infer_flip_pairs=infer_flip_pairs
    )

    assert dataset.get_keypoint_metadata()["pose"].flip_pairs == flip_pairs


def test_a_record_turns_the_inference_off_with_an_empty_list(
    dataset_name: str, tempdir: Path
):
    """An empty list looked the same as a record without flip pairs.

    `add` thus inferred flip pairs, and a record could not stop it. The
    native export writes an empty list to keep the inference off.
    """
    dataset = named_dataset(dataset_name, tempdir, fields={"flip_pairs": []})

    assert dataset.get_keypoint_metadata()["pose"].flip_pairs == []


def test_set_keypoint_metadata_keeps_an_empty_list_of_flip_pairs(
    dataset_name: str,
):
    """The method must infer flip pairs only when you omit them.

    The docstring says so, but the method also inferred flip pairs for an
    empty list.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})

    dataset.set_keypoint_metadata(labels=LABELS, flip_pairs=[], task="pose")

    assert dataset.get_keypoint_metadata()["pose"].flip_pairs == []


@pytest.mark.parametrize(
    ("fields", "flip_pairs"),
    [
        pytest.param({}, [(1, 2)], id="omitted"),
        pytest.param({"flip_pairs": []}, [], id="empty"),
    ],
)
def test_box_relative_keypoints_keep_the_fields_that_the_record_gives(
    dataset_name: str,
    tempdir: Path,
    fields: dict[str, list[tuple[int, int]]],
    flip_pairs: list[tuple[int, int]],
):
    """`scale_to_boxes` builds a new keypoint annotation.

    The new annotation must set only the fields that the record gives. A
    copy of every field gives an empty list of flip pairs to each record.
    A copy of only the fields with a value loses the empty list that turns
    the inference off.
    """

    def generator() -> DatasetIterator:
        for i in range(4):
            yield {
                "file": str(create_image(i, tempdir)),
                "task_name": "pose",
                "annotation": {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
                    "keypoints": {"keypoints": NAMED_KEYPOINTS, **fields},
                    "scale_to_boxes": True,
                },
            }

    dataset = create_dataset(dataset_name, generator())

    assert dataset.get_keypoint_metadata()["pose"].flip_pairs == flip_pairs


def test_only_the_skeleton_aliases_are_documented_as_deprecated():
    """A body-level ``.. deprecated::`` block deprecates the whole method.

    The rename left such a block on `set_keypoint_metadata`, which is the
    replacement API. pydoctor then printed the same banner on the
    supported setter as on the aliases that it replaces. The docstring
    must agree with the decorator.
    """
    documented: set[str] = set()
    decorated: set[str] = set()
    for name in (
        "set_keypoint_metadata",
        "get_keypoint_metadata",
        "set_skeletons",
        "get_skeletons",
    ):
        method = getattr(BaseDataset, name)
        docstring = inspect.getdoc(method) or ""
        if any(
            line.startswith(".. deprecated::")
            for line in docstring.splitlines()
        ):
            documented.add(name)
        if hasattr(method, "__deprecated__"):
            decorated.add(name)

    assert documented == decorated == {"set_skeletons", "get_skeletons"}


def test_set_keypoint_metadata_rejects_duplicate_names(
    dataset_name: str, tempdir: Path
):
    """A duplicate name used to pass.

    The stored names then did not identify the keypoints. A record with
    names could not join the task, and an export lost the names. The
    failed call must leave the stored names alone.
    """
    dataset = named_dataset(dataset_name, tempdir)

    with pytest.raises(ValueError, match="Duplicate keypoint names"):
        dataset.set_keypoint_metadata(
            labels=["nose", "left_eye", "left_eye"], task="pose"
        )

    assert dataset.get_keypoint_metadata()["pose"].labels == LABELS


def test_set_keypoint_metadata_changes_no_task_when_one_fails(
    dataset_name: str, tempdir: Path
):
    """Without a task, the method stored each task before the next check.

    The second task has no keypoint names, so the edge names failed there.
    The first task already held the new edge in memory, and the next
    metadata write saved it.
    """

    def generator() -> DatasetIterator:
        yield from keypoint_generator(
            tempdir, NAMED_KEYPOINTS, {"edges": [("nose", "left_eye")]}, n=1
        )
        yield {
            "file": str(create_image(1, tempdir)),
            "task_name": "detection",
            "annotation": {
                "class": "car",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.3},
            },
        }

    dataset = create_dataset(dataset_name, generator(), splits=False)

    with pytest.raises(ValueError, match="Keypoint names are required"):
        dataset.set_keypoint_metadata(edges=[("left_eye", "right_eye")])

    assert dataset.get_keypoint_metadata()["pose"].edges == [(0, 1)]
    dataset.set_classes(["person"], task="pose")
    stored = read_dataset_metadata(dataset)["keypoint_metadata"]["pose"]
    assert stored["edges"] == [[0, 1]]


def test_a_dataset_with_repeated_names_still_opens_and_loads(
    dataset_name: str, tempdir: Path
):
    """The check for repeated names ran each time a dataset opened.

    `Metadata` validates the stored file on open, so a dataset from an
    older luxonis-ml did not open, and no API could rename its keypoints.
    The loader also keyed each row by name, and a repeated name dropped a
    keypoint from the loaded array.
    """
    dataset = repeated_names_dataset(dataset_name, tempdir)

    assert dataset.get_keypoint_metadata()["pose"].labels == REPEATED_LABELS
    assert [
        labels["pose/keypoints"].tolist()
        for _, labels in LuxonisLoader(dataset)
    ] == [[[0.0, 0.0, 2.0, 0.1, 0.1, 2.0, 0.2, 0.2, 2.0]]] * 4


def test_a_later_add_keeps_every_keypoint_of_repeated_names(
    dataset_name: str, tempdir: Path
):
    """`add` aligned the new rows against the stored names.

    The alignment keys the keypoints by name, so the repeated name
    dropped a keypoint from each new row.
    """
    dataset = repeated_names_dataset(dataset_name, tempdir)

    dataset.add(positional_generator(tempdir, [3], start=4))

    assert set(keypoint_payloads(dataset)) == {
        '{"keypoints":[[0.0,0.0,2],[0.1,0.1,2],[0.2,0.2,2]]}'
    }
    assert dataset.get_keypoint_metadata()["pose"].labels == REPEATED_LABELS


def test_repeated_names_still_set_the_keypoint_count(
    dataset_name: str, tempdir: Path
):
    """`add` did not align a record against repeated names.

    A record with four keypoints thus joined a task with three. The loader
    gave that sample a wider keypoint array than the other samples. A task
    with unique names rejects the same record.
    """
    dataset = repeated_names_dataset(dataset_name, tempdir)

    with pytest.raises(ValueError, match="task defines only 3"):
        dataset.add(positional_generator(tempdir, [4], start=4))

    assert len(dataset) == 4
    assert dataset.get_n_keypoints() == {"pose": 3}


@pytest.mark.parametrize(
    "keypoints",
    [
        pytest.param(
            {"tip": (0.9, 0.9, 2), "a": (0.1, 0.1, 2), "b": (0.2, 0.2, 2)},
            id="every-keypoint",
        ),
        pytest.param({"tip": (0.9, 0.9, 2)}, id="subset"),
    ],
)
def test_a_record_with_names_cannot_join_repeated_names(
    dataset_name: str,
    tempdir: Path,
    keypoints: dict[str, tuple[float, float, int]],
):
    """`add` let the names of a record replace repeated names.

    The stored rows kept their order, so a stored keypoint got the name of
    another keypoint. A subset of the names also made the task smaller
    than its rows. The names of a record cannot match repeated names, so
    `add` must reject the record before it writes a row.
    """
    dataset = repeated_names_dataset(dataset_name, tempdir)

    with pytest.raises(ValueError, match="repeats the keypoint names point"):
        dataset.add(
            keypoint_generator(tempdir, keypoints, n=2, start=4), batch_size=1
        )

    assert len(dataset) == 4
    assert dataset.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=REPEATED_LABELS, edges=[(0, 1)]
    )


def test_set_keypoint_metadata_keeps_the_repeated_names_it_is_not_given(
    dataset_name: str, tempdir: Path
):
    """The check for repeated names ran on every call.

    A call that sets only the sigmas thus failed on a dataset from an
    older luxonis-ml. `add` stored the same sigmas without an error.
    """
    dataset = repeated_names_dataset(dataset_name, tempdir)

    dataset.set_keypoint_metadata(sigmas=[0.1, 0.2, 0.3], task="pose")

    reopened = LuxonisDataset(dataset_name)
    assert reopened.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=REPEATED_LABELS, edges=[(0, 1)], sigmas=[0.1, 0.2, 0.3]
    )


@pytest.mark.parametrize(
    "dataset_type", [DatasetType.NATIVE, DatasetType.COCO]
)
def test_an_export_warns_that_it_loses_repeated_names(
    dataset_name: str,
    tempdir: Path,
    warnings_log: list[str],
    dataset_type: DatasetType,
):
    """No export keeps repeated names, and no export warned.

    The native import numbers the keypoints, and the COCO import rejects
    the names.
    """
    dataset = repeated_names_dataset(dataset_name, tempdir)

    dataset.export(tempdir / "exported", dataset_type)

    assert any("repeats the keypoint names point" in m for m in warnings_log)


def test_a_coco_export_without_keypoints_does_not_warn_about_them(
    dataset_name: str, tempdir: Path, warnings_log: list[str]
):
    """The COCO exporter read empty keypoint metadata as many tasks.

    A dataset without keypoints thus got a warning that the export skips
    its keypoint annotations.
    """
    dataset = create_dataset(dataset_name, detection_generator(tempdir))

    dataset.export(tempdir / "exported_coco", DatasetType.COCO)

    assert not any("single keypoint export class" in m for m in warnings_log)


def test_new_names_replace_repeated_names(dataset_name: str, tempdir: Path):
    """Repeated names do not identify the keypoints.

    New names thus only name the keypoints that are already there, as
    they do for placeholder names. A rename of named keypoints drops the
    stored edge instead, because the edge can then join other keypoints.
    """
    dataset = repeated_names_dataset(dataset_name, tempdir)

    dataset.set_keypoint_metadata(
        labels=["left_point", "right_point", "tip"], task="pose"
    )

    reopened = LuxonisDataset(dataset_name)
    assert reopened.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=["left_point", "right_point", "tip"],
        edges=[(0, 1)],
        flip_pairs=[(0, 1)],
    )


def test_set_keypoint_metadata_needs_something_to_set(
    dataset_name: str, tempdir: Path
):
    dataset = named_dataset(dataset_name, tempdir)

    with pytest.raises(ValueError, match="Must provide either"):
        dataset.set_keypoint_metadata()


def test_native_export_round_trips_the_metadata(
    dataset_name: str, tempdir: Path
):
    """Native export used to drop the keypoint metadata entirely.

    `test_export` covers the round-trip against downloaded fixtures; this
    keeps it verifiable without them.
    """
    dataset = named_dataset(
        dataset_name,
        tempdir,
        fields={
            "edges": [("nose", "left_eye"), ("nose", "right_eye")],
            "sigmas": [0.026, 0.025, 0.025],
        },
    )
    exported = dataset.export(tempdir / "exported", DatasetType.NATIVE)
    assert isinstance(exported, Path)

    imported = LuxonisParser(
        str(exported / dataset_name),
        dataset_type=DatasetType.NATIVE,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert imported.get_keypoint_metadata() == dataset.get_keypoint_metadata()


def test_native_export_keeps_the_flip_pairs_turned_off(
    dataset_name: str, tempdir: Path
):
    """The export left out an empty list of flip pairs.

    The import then saw names without flip pairs and inferred them. The
    imported dataset paired the two cameras, and the source dataset did
    not.
    """
    dataset = create_dataset(
        dataset_name,
        positional_generator(tempdir, [3, 3, 3, 3]),
        splits=(1, 0, 0),
    )
    dataset.set_keypoint_metadata(
        labels=CAMERAS, task="pose", infer_flip_pairs=False
    )
    exported = dataset.export(tempdir / "exported", DatasetType.NATIVE)
    assert isinstance(exported, Path)

    imported = LuxonisParser(
        str(exported / dataset_name),
        dataset_type=DatasetType.NATIVE,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert imported.get_keypoint_metadata() == {
        "pose": KeypointMetadata(labels=CAMERAS)
    }


def test_a_coco_import_into_the_dataset_keeps_the_flip_pairs_turned_off(
    dataset_name: str, tempdir: Path
):
    """The parser gives the names of the source again after its `add`.

    The task already had these names, but the call inferred flip pairs
    for them. The import thus paired the two cameras of the dataset.
    """
    dataset = create_dataset(
        dataset_name,
        positional_generator(tempdir, [3, 3, 3, 3]),
        splits=(1, 0, 0),
    )
    dataset.set_keypoint_metadata(
        labels=CAMERAS, task="pose", infer_flip_pairs=False
    )
    exported = dataset.export(tempdir / "exported_coco", DatasetType.COCO)
    assert isinstance(exported, Path)

    LuxonisParser(
        str(exported / dataset_name),
        dataset_type=DatasetType.COCO,
        dataset_name=dataset_name,
        task_name="pose",
        save_dir=tempdir,
    ).parse()

    assert LuxonisDataset(dataset_name).get_keypoint_metadata() == {
        "pose": KeypointMetadata(labels=CAMERAS)
    }


@pytest.mark.parametrize(
    ("labels", "ldf_version", "imported_edges"),
    [
        pytest.param(None, None, [(0, 2)], id="no-names"),
        pytest.param(LABELS, "2.1", [(0, 1), (1, 2)], id="ldf-2.1"),
        pytest.param(LABELS, "2.0", [(0, 1), (1, 2)], id="ldf-2.0"),
    ],
)
def test_a_shorter_record_without_names_does_not_get_the_task_fields(
    dataset_name: str,
    tempdir: Path,
    labels: list[str] | None,
    ldf_version: str | None,
    imported_edges: list[tuple[int, int]],
):
    """The export used to put the task fields on a short first record.

    A task can hold records with different numbers of keypoints. The
    edges and the sigmas describe the full set, but the import checks
    every record against its own keypoints. The short record referred to
    a keypoint that it does not have, so the export was unreadable.

    The export then padded the short record. The import pads the other
    short records only against names. A task without names and LDF 2.1
    or older have no names, so the imported row had another width than
    the source row. LDF 2.1 and older have no edges either, so their
    import chains the keypoints.
    """
    dataset = create_dataset(
        dataset_name, positional_generator(tempdir, [2, 3, 3, 3]), splits=False
    )
    dataset.set_keypoint_metadata(labels=labels, edges=[(0, 2)], task="pose")
    # The short record is alone in its split, so the export meets it
    # first there.
    dataset.make_splits(
        {
            "val": [str(create_image(0, tempdir))],
            "train": [str(create_image(i, tempdir)) for i in (1, 2, 3)],
        }
    )
    exported = dataset.export(
        tempdir / "exported_mixed", DatasetType.NATIVE, ldf_version=ldf_version
    )
    assert isinstance(exported, Path)

    imported = LuxonisParser(
        str(exported / dataset_name),
        dataset_type=DatasetType.NATIVE,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert sorted(keypoint_payloads(imported)) == sorted(
        keypoint_payloads(dataset)
    )
    val_path = exported / dataset_name / "val" / "annotations.json"
    assert [
        detection["keypoints"]
        for detection in exported_detections(val_path)
        if "keypoints" in detection
    ] == [{"keypoints": [[0.0, 0.0, 2], [0.1, 0.1, 2]]}]
    assert imported.get_keypoint_metadata()["pose"].edges == imported_edges


def test_a_named_task_imports_a_record_with_fewer_keypoints(
    dataset_name: str, tempdir: Path
):
    """The export names the keypoints on one full record of the split.

    The import applies those names to every record of the split. The
    keys ``"0"``, ``"1"`` of the short record were then unknown names,
    so the export of the task did not import.
    """
    dataset = create_dataset(
        dataset_name,
        positional_generator(tempdir, [3, 3, 2, 3]),
        splits=(1, 0, 0),
    )
    dataset.set_keypoint_metadata(labels=LABELS, task="pose")
    exported = dataset.export(tempdir / "exported", DatasetType.NATIVE)
    assert isinstance(exported, Path)

    imported = LuxonisParser(
        str(exported / dataset_name),
        dataset_type=DatasetType.NATIVE,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert imported.get_keypoint_metadata()["pose"].labels == LABELS
    assert sorted(keypoint_payloads(imported)) == [
        '{"keypoints":[[0.0,0.0,2],[0.1,0.1,2],[0.0,0.0,0]]}',
        *['{"keypoints":[[0.0,0.0,2],[0.1,0.1,2],[0.2,0.2,2]]}'] * 3,
    ]


def test_a_task_of_short_records_exports_its_names(
    dataset_name: str, tempdir: Path
):
    """The export put the task fields only on a record with every keypoint.

    Every record here has fewer keypoints than the names, so no record
    carried the fields. The import gave the task the names ``"0"`` and
    ``"1"``, and it lost the edge, the flip pair and the sigmas. The
    imported rows also kept two keypoints, but the source loader gives
    three.
    """
    dataset = create_dataset(
        dataset_name,
        positional_generator(tempdir, [2, 2, 2, 2]),
        splits=(1, 0, 0),
    )
    dataset.set_keypoint_metadata(
        labels=LABELS,
        edges=[("nose", "right_eye")],
        sigmas=[0.1, 0.2, 0.3],
        task="pose",
    )
    exported = dataset.export(tempdir / "exported_short", DatasetType.NATIVE)
    assert isinstance(exported, Path)

    imported = LuxonisParser(
        str(exported / dataset_name),
        dataset_type=DatasetType.NATIVE,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert imported.get_keypoint_metadata() == dataset.get_keypoint_metadata()
    assert set(keypoint_payloads(imported)) == {
        '{"keypoints":[[0.0,0.0,2],[0.1,0.1,2],[0.0,0.0,0]]}'
    }


def test_the_native_export_keeps_every_keypoint_of_repeated_names(
    dataset_name: str, tempdir: Path
):
    """The export keyed the keypoints of one record by name.

    A mapping holds one keypoint for each name, so that record lost a
    keypoint. The import took its two names for the task, and every
    other record of the split then had more keypoints than the task.
    """
    dataset = repeated_names_dataset(dataset_name, tempdir)
    exported = dataset.export(
        tempdir / "exported_repeated", DatasetType.NATIVE
    )
    assert isinstance(exported, Path)

    imported = LuxonisParser(
        str(exported / dataset_name),
        dataset_type=DatasetType.NATIVE,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert imported.get_keypoint_metadata()["pose"].edges == [(0, 1)]
    assert set(keypoint_payloads(imported)) == {
        '{"keypoints":[[0.0,0.0,2],[0.1,0.1,2],[0.2,0.2,2]]}'
    }


def test_the_native_export_leaves_out_names_for_fewer_keypoints_than_a_row(
    dataset_name: str, tempdir: Path, warnings_log: list[str]
):
    """New names do not change the rows that a task already has.

    The rows of the first `add` have five keypoints, and the names cover
    three. The export named the keypoints of the val split. The import
    then rejected the three names, because the train split has rows of
    five keypoints.
    """
    dataset = create_dataset(
        dataset_name, positional_generator(tempdir, [5, 5]), splits=False
    )
    dataset.set_keypoint_metadata(labels=LABELS, task="pose")
    dataset.add(keypoint_generator(tempdir, NAMED_KEYPOINTS, n=1, start=2))
    dataset.make_splits(
        {
            "train": [str(create_image(i, tempdir)) for i in (0, 1)],
            "val": [str(create_image(2, tempdir))],
        }
    )
    exported = dataset.export(tempdir / "exported_wider", DatasetType.NATIVE)
    assert isinstance(exported, Path)

    imported = LuxonisParser(
        str(exported / dataset_name),
        dataset_type=DatasetType.NATIVE,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert any("names 3 keypoints, but a row has 5" in m for m in warnings_log)
    assert imported.get_keypoint_metadata()["pose"].labels == [
        str(i) for i in range(5)
    ]
    assert sorted(keypoint_payloads(imported)) == sorted(
        keypoint_payloads(dataset)
    )


def test_a_later_add_pads_a_record_with_fewer_keypoints(
    dataset_name: str, tempdir: Path
):
    """A record without names holds the leading keypoints of the task."""
    dataset = named_dataset(dataset_name, tempdir)

    dataset.add(positional_generator(tempdir, [2], start=4))

    assert '{"keypoints":[[0.0,0.0,2],[0.1,0.1,2],[0.0,0.0,0]]}' in set(
        keypoint_payloads(dataset)
    )
    assert dataset.get_keypoint_metadata()["pose"].labels == LABELS


def test_the_batch_size_does_not_change_a_short_record(
    dataset_name: str, tempdir: Path
):
    """A small batch writes the short records before the names are known.

    The default batch saw the names first and rejected the records. A
    batch of one accepted them and stored rows of two keypoints and of
    one keypoint. The loader, the equality of two datasets and the
    exporters all read the stored rows, so the rows must not depend on
    the batch size.
    """

    def generator() -> DatasetIterator:
        yield from positional_generator(tempdir, [2, 1])
        yield from keypoint_generator(tempdir, NAMED_KEYPOINTS, n=1, start=2)

    stored = []
    for batch_size in (1, 2, 1_000_000):
        dataset = LuxonisDataset(
            f"{dataset_name}_{batch_size}", delete_local=True
        ).add(generator(), batch_size=batch_size)
        assert dataset.get_keypoint_metadata()["pose"].labels == LABELS
        stored.append(sorted(keypoint_payloads(dataset)))

    padded = [
        '{"keypoints":[[0.0,0.0,2],[0.0,0.0,0],[0.0,0.0,0]]}',
        '{"keypoints":[[0.0,0.0,2],[0.1,0.1,2],[0.0,0.0,0]]}',
        '{"keypoints":[[0.5,0.3,2],[0.4,0.2,2],[0.6,0.2,1]]}',
    ]
    assert stored == [padded] * 3


def test_the_batch_size_does_not_change_a_row_of_an_earlier_add(
    dataset_name: str, tempdir: Path
):
    """`add` found the short rows of its earlier batches by their UUID.

    The UUID comes from the bytes of the image. The copy of an image from
    an earlier `add` has another path, but the same UUID. A small batch
    thus also padded the row of the earlier `add`, and one batch did not.
    """
    copy = tempdir / "copy_of_img_0.jpg"
    shutil.copy(create_image(0, tempdir), copy)

    def generator() -> DatasetIterator:
        yield {
            "file": str(copy),
            "task_name": "pose",
            "annotation": {
                "class": "person",
                "keypoints": {"keypoints": [(0.9, 0.9, 2)]},
            },
        }
        # A batch of one record writes the copy before the names are known.
        yield from positional_generator(tempdir, [3], start=1)
        yield from keypoint_generator(tempdir, NAMED_KEYPOINTS, n=1, start=2)

    stored = []
    for batch_size in (1, 1_000_000):
        dataset = create_dataset(
            f"{dataset_name}_{batch_size}",
            positional_generator(tempdir, [2]),
            splits=False,
        ).add(generator(), batch_size=batch_size)
        stored.append(sorted(keypoint_payloads(dataset)))

    expected = sorted(
        [
            '{"keypoints":[[0.0,0.0,2],[0.1,0.1,2]]}',
            '{"keypoints":[[0.9,0.9,2],[0.0,0.0,0],[0.0,0.0,0]]}',
            '{"keypoints":[[0.0,0.0,2],[0.1,0.1,2],[0.2,0.2,2]]}',
            '{"keypoints":[[0.5,0.3,2],[0.4,0.2,2],[0.6,0.2,1]]}',
        ]
    )
    assert stored == [expected] * 2


@pytest.mark.parametrize(
    ("keypoints", "warns"),
    [
        pytest.param(NAMED_KEYPOINTS, False, id="names"),
        pytest.param(list(NAMED_KEYPOINTS.values()), True, id="list"),
    ],
)
def test_only_a_task_without_names_warns_about_mixed_widths(
    dataset_name: str,
    tempdir: Path,
    warnings_log: list[str],
    keypoints: list[tuple[float, float, int]]
    | dict[str, tuple[float, float, int]],
    warns: bool,
):
    """`add` warned about mixed widths after it padded every row.

    The names of a task set the width, and `add` pads the short record to
    it. Without names, the rows keep their own widths.
    """

    def generator() -> DatasetIterator:
        yield from positional_generator(tempdir, [2])
        yield from keypoint_generator(tempdir, keypoints, n=1, start=1)

    LuxonisDataset(dataset_name, delete_local=True).add(generator())

    assert any("mixes annotations" in m for m in warnings_log) == warns


def test_the_loader_pads_a_short_row_to_the_names(
    dataset_name: str, tempdir: Path
):
    """New names do not change the rows that a task already has.

    The loader read a short row under positional keys, so the sample of
    that row got a narrower keypoint array than the other sample.
    """
    dataset = create_dataset(
        dataset_name, positional_generator(tempdir, [2, 3]), splits=(1, 0, 0)
    )
    dataset.set_keypoint_metadata(labels=LABELS, task="pose")

    assert sorted(
        labels["pose/keypoints"].tolist()
        for _, labels in LuxonisLoader(dataset)
    ) == [
        [[0.0, 0.0, 2.0, 0.1, 0.1, 2.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 2.0, 0.1, 0.1, 2.0, 0.2, 0.2, 2.0]],
    ]


def test_edges_alone_do_not_widen_a_loaded_row(
    dataset_name: str, tempdir: Path
):
    """Edges do not set the width of a loaded row.

    Without labels, `get_n_keypoints` reads the keypoint count off the
    highest edge index. An older luxonis-ml wrote the same edges to every
    task, so the hand task has the edges of the pose task. The loader must
    not give each hand three keypoints that it does not have.
    """

    def generator() -> DatasetIterator:
        yield from positional_generator(tempdir, [5])
        yield {
            "file": str(create_image(0, tempdir)),
            "task_name": "hand",
            "annotation": {
                "class": "hand",
                "keypoints": {"keypoints": [(0.3, 0.3, 2), (0.4, 0.4, 1)]},
            },
        }

    dataset = legacy_dataset(
        create_dataset(dataset_name, generator(), splits=(1, 0, 0)),
        {
            task: {"labels": [], "edges": [[0, 1], [3, 4]]}
            for task in ("pose", "hand")
        },
    )

    _, labels = LuxonisLoader(dataset)[0]

    assert labels["pose/keypoints"].shape == (1, 15)
    assert labels["hand/keypoints"].tolist() == [
        [0.3, 0.3, 2.0, 0.4, 0.4, 1.0]
    ]


def test_coco_export_round_trips_the_sigmas(dataset_name: str, tempdir: Path):
    """The COCO exporter writes the sigmas, but the parser dropped them.

    OKS scoring reads the sigmas. A silent fallback to the defaults
    changes the metric, so the round trip has to keep the exact values.
    """
    sigmas = [0.026, 0.025, 0.025]
    dataset = named_dataset(dataset_name, tempdir, fields={"sigmas": sigmas})
    exported = dataset.export(tempdir / "exported_coco", DatasetType.COCO)
    assert isinstance(exported, Path)

    imported = LuxonisParser(
        str(exported / dataset_name),
        dataset_type=DatasetType.COCO,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    task_keypoints = next(iter(imported.get_keypoint_metadata().values()))
    assert task_keypoints.labels == LABELS
    assert task_keypoints.sigmas == sigmas


def test_the_exported_names_are_written_once_per_task(
    dataset_name: str, tempdir: Path
):
    """Naming every record would balloon ``annotations.json``.

    One record per task and split carries the names as a mapping. Every
    other record stays a positional list. Each split gets its own file, so
    each one needs its own named record.
    """
    dataset = named_dataset(dataset_name, tempdir, n=8)
    exported = dataset.export(tempdir / "exported_once", DatasetType.NATIVE)
    assert isinstance(exported, Path)

    counts = []
    for path in (exported / dataset_name).rglob("annotations.json"):
        keypoints = [
            detection["keypoints"]["keypoints"]
            for record in json.loads(path.read_text())
            for detections in record.get("annotation", {}).values()
            for detection in detections
            # Every detection also emits a classification record.
            if "keypoints" in detection
        ]
        if keypoints:
            named = sum(isinstance(k, dict) for k in keypoints)
            counts.append((len(keypoints), named))

    assert counts
    assert all(n_named == 1 for _, n_named in counts)
    assert any(n_keypoints > 1 for n_keypoints, _ in counts)
