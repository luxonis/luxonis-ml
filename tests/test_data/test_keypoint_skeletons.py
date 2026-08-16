"""Dataset-level tests for keypoint skeletons.

`LuxonisDataset.add` moves the skeleton of an annotation into the dataset
metadata. These tests cover that move and the compatibility that it must
keep.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.enums import DatasetType
from luxonis_ml.ldf import Skeleton

from .utils import create_dataset, create_image

LABELS = ["nose", "left_eye", "right_eye"]


def keypoint_generator(
    tempdir: Path,
    keypoints: Any,
    skeleton: dict[str, Any] | None = None,
    n: int = 4,
) -> DatasetIterator:
    for i in range(n):
        annotation: dict[str, Any] = {"keypoints": keypoints}
        if skeleton is not None:
            annotation["skeleton"] = skeleton
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


def read_metadata(dataset: LuxonisDataset) -> dict[str, Any]:
    return json.loads((dataset._metadata_path / "metadata.json").read_text())


def test_names_are_promoted_to_the_task_skeleton(
    dataset_name: str, tempdir: Path
):
    dataset = named_dataset(
        dataset_name,
        tempdir,
        skeleton={
            "edges": [("nose", "left_eye"), ("nose", "right_eye")],
            "sigmas": [0.026, 0.025, 0.025],
        },
    )

    assert dataset.get_skeletons() == {
        "pose": Skeleton(
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
    """No skeleton declared at all; the names alone are enough."""
    dataset = named_dataset(dataset_name, tempdir)

    assert dataset.get_skeletons()["pose"].flip_pairs == [(1, 2)]


def test_sub_detections_get_their_own_skeleton(
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

    skeleton = dataset.get_skeletons()["person/face"]
    assert skeleton.labels == ["left_eye", "right_eye"]
    assert skeleton.flip_pairs == [(0, 1)]


def test_disagreeing_records_are_rejected(dataset_name: str, tempdir: Path):
    def generator() -> DatasetIterator:
        yield from keypoint_generator(
            tempdir, {"nose": (0.5, 0.3, 2), "left_eye": (0.4, 0.2, 2)}, n=1
        )
        yield from keypoint_generator(
            tempdir, {"nose": (0.5, 0.3, 2), "right_eye": (0.6, 0.2, 2)}, n=1
        )

    with pytest.raises(ValueError, match="Conflicting keypoint skeletons"):
        create_dataset(dataset_name, generator())


def test_an_unknown_keypoint_name_is_rejected(
    dataset_name: str, tempdir: Path
):
    with pytest.raises(ValueError, match="not part of the skeleton"):
        create_dataset(
            dataset_name,
            keypoint_generator(
                tempdir,
                {"noze": (0.5, 0.3, 2)},
                skeleton={"labels": LABELS},
            ),
        )


def test_add_does_not_clobber_explicit_skeletons(
    dataset_name: str, tempdir: Path
):
    """`add` used to overwrite every skeleton with ``"0"``, ``"1"``, ...

    Adding unnamed keypoints to a dataset whose skeleton was set by hand
    has to leave that skeleton alone.
    """
    dataset = LuxonisDataset(dataset_name, delete_local=True)
    dataset.set_tasks({"pose": ["keypoints"]})
    dataset.set_skeletons(labels=LABELS, edges=[(0, 1), (0, 2)], task="pose")

    dataset.add(
        keypoint_generator(
            tempdir, [(0.5, 0.3, 2), (0.4, 0.2, 2), (0.6, 0.2, 1)]
        )
    )

    skeleton = dataset.get_skeletons()["pose"]
    assert skeleton.labels == LABELS
    assert skeleton.edges == [(0, 1), (0, 2)]


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
    assert dataset.get_skeletons()["pose"].labels == ["0", "1", "2"]

    dataset.set_skeletons(labels=LABELS, task="pose")
    dataset.add(
        keypoint_generator(
            tempdir, [(0.1, 0.1, 2), (0.2, 0.2, 2), (0.3, 0.3, 2)]
        )
    )

    assert dataset.get_skeletons()["pose"].labels == LABELS


def test_placeholders_are_still_generated(dataset_name: str, tempdir: Path):
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, [(0.5, 0.3, 2), (0.4, 0.2, 2)]),
    )

    assert dataset.get_skeletons() == {
        "pose": Skeleton(labels=["0", "1"], edges=[(0, 1)])
    }


def test_the_skeleton_is_not_repeated_on_every_row(
    dataset_name: str, tempdir: Path
):
    dataset = named_dataset(
        dataset_name,
        tempdir,
        skeleton={"edges": [("nose", "left_eye")], "sigmas": [0.1, 0.2, 0.3]},
    )

    payloads = keypoint_payloads(dataset)

    assert payloads
    for payload in payloads:
        assert json.loads(payload) == {
            "keypoints": [[0.5, 0.3, 2], [0.4, 0.2, 2], [0.6, 0.2, 1]]
        }
        assert "skeleton" not in payload
        assert "nose" not in payload


def test_records_are_stored_in_skeleton_order(
    dataset_name: str, tempdir: Path
):
    """Column position is keypoint identity across the whole task."""

    def generator() -> DatasetIterator:
        yield from keypoint_generator(
            tempdir,
            {"nose": (0.1, 0.1, 2), "left_eye": (0.2, 0.2, 2)},
            skeleton={"labels": ["nose", "left_eye"]},
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
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(
            tempdir,
            {"left_eye": (0.4, 0.2, 2)},
            skeleton={"labels": LABELS},
        ),
    )

    assert json.loads(keypoint_payloads(dataset)[0]) == {
        "keypoints": [[0.0, 0.0, 0], [0.4, 0.2, 2], [0.0, 0.0, 0]]
    }


def test_opening_a_dataset_does_not_materialize_flip_pairs(
    dataset_name: str, tempdir: Path
):
    """Inference belongs to the write paths only.

    Every open of a dataset revalidates `Metadata`. Inference there would
    give flip pairs to a dataset that never asked for them. An older
    ``luxonis-ml`` cannot read a skeleton that has them.
    """
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, [(0.5, 0.3, 2), (0.4, 0.2, 2)]),
    )
    metadata_path = dataset._metadata_path / "metadata.json"
    before = metadata_path.read_text()
    assert "flip_pairs" not in before

    reopened = LuxonisDataset(dataset_name)

    assert reopened.get_skeletons()["pose"].flip_pairs == []
    assert metadata_path.read_text() == before


def test_unused_fields_stay_out_of_the_metadata(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(
        dataset_name,
        keypoint_generator(tempdir, [(0.5, 0.3, 2), (0.4, 0.2, 2)]),
    )

    assert read_metadata(dataset)["skeletons"] == {
        "pose": {"labels": ["0", "1"], "edges": [[0, 1]]}
    }


def test_a_legacy_dataset_still_loads(dataset_name: str, tempdir: Path):
    """Datasets written before flip pairs and sigmas existed."""
    dataset = named_dataset(dataset_name, tempdir)
    metadata_path = dataset._metadata_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for skeleton in metadata["skeletons"].values():
        skeleton.pop("flip_pairs", None)
        skeleton.pop("sigmas", None)
    metadata_path.write_text(json.dumps(metadata))

    reopened = LuxonisDataset(dataset_name)

    assert reopened.get_skeletons()["pose"].labels == LABELS
    assert reopened.get_skeletons()["pose"].flip_pairs == []
    _, labels = LuxonisLoader(reopened)[0]
    assert labels["pose/keypoints"].shape[1] == 9


def test_the_loader_names_the_keypoints(dataset_name: str, tempdir: Path):
    dataset = named_dataset(dataset_name, tempdir)
    loader = LuxonisLoader(dataset)

    _, labels = loader[0]

    assert labels["pose/keypoints"].shape == (1, 9)
    assert loader.get_skeletons()["pose"].labels == LABELS


def test_set_skeletons_updates_only_what_it_is_given(
    dataset_name: str, tempdir: Path
):
    """It used to replace the whole entry, so one field wiped the rest.

    A skeleton now has four fields, which makes that unacceptable.
    """
    dataset = named_dataset(dataset_name, tempdir)

    dataset.set_skeletons(sigmas=[0.1, 0.2, 0.3], task="pose")

    skeleton = dataset.get_skeletons()["pose"]
    assert skeleton.labels == LABELS
    assert skeleton.sigmas == [0.1, 0.2, 0.3]
    assert skeleton.flip_pairs == [(1, 2)]


def test_set_skeletons_accepts_names(dataset_name: str, tempdir: Path):
    dataset = named_dataset(dataset_name, tempdir)

    dataset.set_skeletons(
        labels=LABELS,
        edges=[("nose", "left_eye")],
        flip_pairs=[("left_eye", "right_eye")],
        task="pose",
    )

    skeleton = dataset.get_skeletons()["pose"]
    assert skeleton.edges == [(0, 1)]
    assert skeleton.flip_pairs == [(1, 2)]


def test_flip_pair_inference_can_be_turned_off(
    dataset_name: str, tempdir: Path
):
    dataset = named_dataset(dataset_name, tempdir)

    dataset.set_skeletons(labels=LABELS, task="pose", infer_flip_pairs=False)

    # The `add` already inferred them. A fresh dataset below shows that
    # the flag keeps them away.
    assert dataset.get_skeletons()["pose"].flip_pairs == [(1, 2)]

    fresh = LuxonisDataset(f"{dataset_name}_fresh", delete_local=True)
    fresh.set_tasks({"pose": ["keypoints"]})
    fresh.set_skeletons(labels=LABELS, task="pose", infer_flip_pairs=False)

    assert fresh.get_skeletons()["pose"].flip_pairs == []


def test_set_skeletons_needs_something_to_set(
    dataset_name: str, tempdir: Path
):
    dataset = named_dataset(dataset_name, tempdir)

    with pytest.raises(ValueError, match="Must provide either"):
        dataset.set_skeletons()


def test_native_export_round_trips_the_skeleton(
    dataset_name: str, tempdir: Path
):
    """Native export used to drop skeletons entirely.

    `test_export` covers the round-trip against downloaded fixtures; this
    keeps it verifiable without them.
    """
    dataset = named_dataset(
        dataset_name,
        tempdir,
        skeleton={
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

    assert imported.get_skeletons() == dataset.get_skeletons()


def test_the_exported_skeleton_is_written_once_per_task(
    dataset_name: str, tempdir: Path
):
    """Writing it on every record would balloon ``annotations.json``.

    Each split gets its own file, so each has to carry the skeleton once.
    """
    dataset = named_dataset(dataset_name, tempdir, n=8)
    exported = dataset.export(tempdir / "exported_once", DatasetType.NATIVE)
    assert isinstance(exported, Path)

    counts = []
    for path in (exported / dataset_name).rglob("annotations.json"):
        keypoints = [
            record["annotation"]["keypoints"]
            for record in json.loads(path.read_text())
            # Every detection also emits a classification record.
            if "keypoints" in record.get("annotation", {})
        ]
        if keypoints:
            counts.append(
                (len(keypoints), sum("skeleton" in k for k in keypoints))
            )

    assert counts
    assert all(n_skeletons == 1 for _, n_skeletons in counts)
    assert any(n_keypoints > 1 for n_keypoints, _ in counts)
