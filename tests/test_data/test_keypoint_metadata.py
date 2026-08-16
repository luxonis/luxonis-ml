"""Dataset-level tests for keypoint metadata.

`LuxonisDataset.add` moves the task fields of an annotation into the
dataset metadata. These tests cover that move and the compatibility that
it must keep.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.enums import DatasetType
from luxonis_ml.ldf import KeypointMetadata

from .utils import create_dataset, create_image

LABELS = ["nose", "left_eye", "right_eye"]


def keypoint_generator(
    tempdir: Path,
    keypoints: Any,
    fields: dict[str, Any] | None = None,
    n: int = 4,
) -> DatasetIterator:
    for i in range(n):
        annotation: dict[str, Any] = {"keypoints": keypoints}
        annotation.update(fields or {})
        yield {
            "file": str(create_image(i, tempdir)),
            "task_name": "pose",
            "annotation": {"class": "person", "keypoints": annotation},
        }


def named_dataset(
    dataset_name: str, tempdir: Path, **kwargs: Any
) -> LuxonisDataset:
    return create_dataset(
        dataset_name,
        keypoint_generator(
            tempdir,
            {
                "nose": (0.5, 0.3, 2),
                "left_eye": (0.4, 0.2, 2),
                "right_eye": (0.6, 0.2, 1),
            },
            **kwargs,
        ),
    )


def keypoint_payloads(dataset: LuxonisDataset) -> list[str]:
    df = dataset._load_df_offline(raise_when_empty=True)
    return df.filter(df["task_type"] == "keypoints")["annotation"].to_list()


def read_dataset_metadata(dataset: LuxonisDataset) -> dict[str, Any]:
    return json.loads((dataset._metadata_path / "metadata.json").read_text())


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
    """No task fields declared at all; the names alone are enough."""
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
        assert "edges" not in payload
        assert "sigmas" not in payload
        assert "nose" not in payload


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


def test_the_loader_names_the_keypoints(dataset_name: str, tempdir: Path):
    dataset = named_dataset(dataset_name, tempdir)
    loader = LuxonisLoader(dataset)

    _, labels = loader[0]

    assert labels["pose/keypoints"].shape == (1, 9)
    assert loader.get_keypoint_metadata()["pose"].labels == LABELS


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


def test_set_keypoint_metadata_rejects_duplicate_names(
    dataset_name: str, tempdir: Path
):
    """A duplicate name used to pass, and it destroyed a keypoint.

    Both the stored payload and the loader output key the keypoints by
    name, so the second ``left_eye`` overwrote the first one.
    `get_n_keypoints` still reported three, and nothing warned. The shape
    assertion is what a duplicate name silently changed to ``(1, 6)``.
    """
    dataset = named_dataset(dataset_name, tempdir)

    with pytest.raises(ValueError, match="Duplicate keypoint names"):
        dataset.set_keypoint_metadata(
            labels=["nose", "left_eye", "left_eye"], task="pose"
        )

    assert dataset.get_keypoint_metadata()["pose"].labels == LABELS
    _, labels = LuxonisLoader(dataset)[0]
    assert labels["pose/keypoints"].shape == (1, 9)


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
            record["annotation"]["keypoints"]["keypoints"]
            for record in json.loads(path.read_text())
            # Every detection also emits a classification record.
            if "keypoints" in record.get("annotation", {})
        ]
        if keypoints:
            named = sum(isinstance(k, dict) for k in keypoints)
            counts.append((len(keypoints), named))

    assert counts
    assert all(n_named == 1 for _, n_named in counts)
    assert any(n_keypoints > 1 for n_keypoints, _ in counts)
