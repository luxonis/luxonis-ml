"""Exporting the native format at an older LDF version.

`DatasetRecord` forbids extra fields, so LDF 2.1 adding ``sample_metadata``
made every 2.1 export unreadable by an older luxonis-ml. An annotation
forbids them too, and LDF 2.2 added the keypoint task fields.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger
from semver.version import Version

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.exporters.ldf_downgrade import (
    _ADDED_ANNOTATION_FIELDS,
    _ADDED_FIELDS,
    LDFDowngrader,
    resolve_export_version,
)
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.enums.enums import DatasetType
from luxonis_ml.ldf import DatasetRecord, KeypointAnnotation

from .utils import create_dataset, create_image

#: Fields a pre-2.1 `DatasetRecord` accepts. Anything else trips
#: ``extra="forbid"`` on an older install.
LDF_2_0_RECORD_FIELDS = {"file", "files", "task_name", "annotation"}

#: The same for a keypoint annotation, which gained the task fields
#: in 2.2.
LDF_2_0_KEYPOINT_FIELDS = {"keypoints"}


@pytest.fixture
def warnings_log() -> Iterator[list[str]]:
    """Collect loguru warnings, which pytest's caplog does not see."""
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler)


def _generator(tempdir: Path, with_metadata: bool = True) -> DatasetIterator:
    for i in range(2):
        record = {
            "file": create_image(i, tempdir),
            "annotation": {
                "class": "person",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                "instance_id": i,
            },
        }
        if with_metadata:
            record["sample_metadata"] = {"record_id": i, "origin": "test"}
        yield record


def _keypoint_generator(tempdir: Path) -> DatasetIterator:
    """Yield named keypoints, which `add` promotes to the keypoint metadata."""
    for i in range(2):
        yield {
            "file": create_image(i, tempdir),
            "task_name": "pose",
            "sample_metadata": {"record_id": i, "origin": "test"},
            "annotation": {
                "class": "person",
                "keypoints": {
                    "keypoints": {
                        "nose": (0.5, 0.3, 2),
                        "left_eye": (0.4, 0.2, 2),
                        "right_eye": (0.6, 0.2, 1),
                    }
                },
            },
        }


def _read_records(export_root: Path) -> list[dict]:
    """Read every annotation record an export directory holds."""
    return [
        record
        for path in sorted(export_root.rglob("annotations.json"))
        for record in json.loads(path.read_text())
    ]


def _read_stamp(export_root: Path) -> str:
    return json.loads((export_root / "metadata.json").read_text())[
        "ldf_version"
    ]


def _export(
    dataset: LuxonisDataset, tempdir: Path, name: str, **kwargs
) -> Path:
    """Export to a fresh subdirectory and return the dataset root."""
    dataset.export(tempdir / name, **kwargs)
    return tempdir / name / dataset.identifier


@pytest.mark.parametrize(
    ("version", "expected"),
    [("2.0", "2.0.0"), ("2.0.0", "2.0.0"), ("2.1", "2.1.0")],
)
def test_resolve_accepts_short_and_full_versions(version: str, expected: str):
    assert resolve_export_version(version) == Version.parse(expected)


def test_resolve_defaults_to_current_version():
    assert resolve_export_version(None) == LDF_VERSION


@pytest.mark.parametrize("version", ["banana", "", "v2.0", "2.0.0.0"])
def test_resolve_rejects_malformed_versions(version: str):
    with pytest.raises(ValueError, match="Invalid LDF version"):
        resolve_export_version(version)


@pytest.mark.parametrize("version", ["3.0", "2.9"])
def test_resolve_rejects_newer_than_installed(version: str):
    with pytest.raises(ValueError, match="at the newest"):
        resolve_export_version(version)


@pytest.mark.parametrize("version", ["1.0", "2.0.1"])
def test_resolve_rejects_unsupported_versions(version: str):
    with pytest.raises(ValueError, match="Supported versions"):
        resolve_export_version(version)


def test_every_record_field_has_a_known_ldf_version():
    """Adding a `DatasetRecord` field must register the version it arrived in.

    Otherwise an export targeting an older version silently keeps the new
    field, and the install it was made for rejects the whole record.
    """
    known = LDF_2_0_RECORD_FIELDS | set(_ADDED_FIELDS)
    assert set(DatasetRecord.model_fields) <= known


def test_every_keypoint_field_has_a_known_ldf_version():
    """An annotation forbids extra fields just as a record does."""
    known = LDF_2_0_KEYPOINT_FIELDS | {
        field
        for task_type, field in _ADDED_ANNOTATION_FIELDS
        if task_type == "keypoints"
    }
    assert set(KeypointAnnotation.model_fields) <= known


def test_downgrade_removes_the_key_rather_than_emptying_it():
    """``sample_metadata: {}`` still fails ``extra="forbid"``."""
    downgraded = LDFDowngrader(Version.parse("2.0.0"))(
        {"file": "a.jpg", "task_name": "t", "sample_metadata": {"x": 1}}
    )
    assert "sample_metadata" not in downgraded


def test_downgrade_to_current_version_is_a_passthrough():
    record = {"file": "a.jpg", "sample_metadata": {"x": 1}}
    assert LDFDowngrader(LDF_VERSION)(dict(record)) == record


def test_downgrade_removes_the_keypoint_task_fields():
    """They sit inside the annotation, not on the record.

    The names are the keys of the payload, so the downgrade also has to
    turn a named mapping back into a positional list.
    """
    keypoints = {
        "keypoints": {"nose": [0.5, 0.3, 2], "left_eye": [0.4, 0.2, 2]},
        "edges": [[0, 1]],
        "flip_pairs": [[0, 1]],
        "sigmas": [0.026, 0.025],
    }
    downgraded = LDFDowngrader(Version.parse("2.0.0"))(
        {"file": "a.jpg", "annotation": {"keypoints": keypoints}}
    )

    assert downgraded["annotation"]["keypoints"] == {
        "keypoints": [[0.5, 0.3, 2], [0.4, 0.2, 2]]
    }


def test_export_2_0_omits_sample_metadata(dataset_name: str, tempdir: Path):
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(
        dataset,
        tempdir,
        "v20",
        dataset_type=DatasetType.NATIVE,
        ldf_version="2.0",
    )

    records = _read_records(root)
    assert records
    for record in records:
        assert set(record) <= LDF_2_0_RECORD_FIELDS
    assert _read_stamp(root) == "2.0.0"


def test_export_2_0_drops_the_keypoint_task_fields(
    dataset_name: str, tempdir: Path, warnings_log: list[str]
):
    """The keypoint task fields arrived in LDF 2.2.

    `test_keypoint_metadata` covers the export writing them at the current
    version, so this only has to show that 2.0 does not.
    """
    dataset = create_dataset(
        dataset_name, _keypoint_generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(dataset, tempdir, "keypoints20", ldf_version="2.0")

    keypoints = [
        record["annotation"]["keypoints"]
        for record in _read_records(root)
        # Every keypoint detection also emits a classification record.
        if "keypoints" in record.get("annotation", {})
    ]
    assert keypoints
    assert all(isinstance(k["keypoints"], list) for k in keypoints)
    assert all("edges" not in k for k in keypoints)
    assert [msg for msg in warnings_log if "keypoints.names" in msg]


def test_export_2_1_keeps_sample_metadata_but_drops_the_task_fields(
    dataset_name: str, tempdir: Path
):
    """LDF 2.1 knows ``sample_metadata``. Only the rest is newer."""
    dataset = create_dataset(
        dataset_name, _keypoint_generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(dataset, tempdir, "keypoints21", ldf_version="2.1")

    records = _read_records(root)
    assert records
    assert all("sample_metadata" in record for record in records)
    assert not [
        record
        for record in records
        if "edges" in record.get("annotation", {}).get("keypoints", {})
    ]
    assert _read_stamp(root) == "2.1.0"


def test_an_export_without_the_task_fields_still_imports(
    dataset_name: str, tempdir: Path
):
    """A dropped field must not make the export unusable.

    The stored keypoints keep the task order, so the import names them by
    position.
    """
    dataset = create_dataset(
        dataset_name, _keypoint_generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(dataset, tempdir, "keypoints20_import", ldf_version="2.0")

    imported = LuxonisParser(
        str(root),
        dataset_type=DatasetType.NATIVE,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()
    imported.make_splits((1, 0, 0), replace_old_splits=True)

    _, labels = LuxonisLoader(imported)[0]
    assert labels["pose/keypoints"].shape == (1, 9)
    assert imported.get_keypoint_metadata()["pose"].labels == ["0", "1", "2"]


def test_export_defaults_to_current_version(dataset_name: str, tempdir: Path):
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(dataset, tempdir, "default")

    records = _read_records(root)
    assert records
    assert all("sample_metadata" in record for record in records)
    assert _read_stamp(root) == str(LDF_VERSION)


def test_export_2_0_round_trips(dataset_name: str, tempdir: Path):
    """A 2.0 export re-imports cleanly, minus the dropped sample metadata."""
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(dataset, tempdir, "v20", ldf_version="2.0")

    imported = LuxonisParser(
        str(root),
        dataset_type=DatasetType.NATIVE,
        dataset_name=f"{dataset_name}_imported",
        delete_local=True,
        save_dir=tempdir,
    ).parse()
    imported.make_splits((1, 0, 0), replace_old_splits=True)

    outputs = list(LuxonisLoader(imported))
    assert len(outputs) == 2
    # The loader always autopopulates `filenames`, so the record-level
    # metadata is never empty; what must be gone are the exported keys.
    for output in outputs:
        assert "record_id" not in output.metadata
        assert "origin" not in output.metadata


def test_export_2_0_warns_about_dropped_sample_metadata(
    dataset_name: str, tempdir: Path, warnings_log: list[str]
):
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(dataset, tempdir, "v20", ldf_version="2.0")

    # One record per annotation row, so the count is not the image count.
    n_records = len(_read_records(root))
    dropped = [msg for msg in warnings_log if "sample_metadata" in msg]
    assert len(dropped) == 1
    assert f"{n_records} of {n_records} records" in dropped[0]


def test_export_2_0_is_quiet_when_nothing_is_dropped(
    dataset_name: str, tempdir: Path, warnings_log: list[str]
):
    """An empty `sample_metadata` is no loss, so it must not warn."""
    dataset = create_dataset(
        dataset_name,
        _generator(tempdir, with_metadata=False),
        splits=(1, 0, 0),
    )
    _export(dataset, tempdir, "v20", ldf_version="2.0")

    assert not [msg for msg in warnings_log if "sample_metadata" in msg]


def test_export_2_0_stamps_every_partition(dataset_name: str, tempdir: Path):
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    out = tempdir / "parts"
    dataset.export(out, max_partition_size_gb=1e-7, ldf_version="2.0")

    parts = sorted(out.glob(f"{dataset.identifier}_part*"))
    assert len(parts) > 1
    for part in parts:
        assert _read_stamp(part) == "2.0.0"
    for record in _read_records(out):
        assert "sample_metadata" not in record


def _stamp(export_root: Path, version: str) -> None:
    """Overwrite an export's version stamp, faking another install."""
    (export_root / "metadata.json").write_text(
        json.dumps({"ldf_version": version})
    )


def _import(export_root: Path, dataset_name: str, tempdir: Path) -> None:
    LuxonisParser(
        str(export_root),
        dataset_type=DatasetType.NATIVE,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()


def _newer_export(dataset_name: str, tempdir: Path, version: str) -> Path:
    """Export a dataset and restamp it as written by a newer install."""
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(dataset, tempdir, f"from_{version}")
    _stamp(root, version)
    return root


@pytest.mark.parametrize(
    "version",
    [
        # A newer major, and -- the case a `.major`-only comparison
        # misses -- a newer minor. `sample_metadata` arrived in exactly
        # such a bump, so a same-major export is just as unreadable.
        f"{LDF_VERSION.major + 1}.0.0",
        str(LDF_VERSION.bump_minor()),
        str(LDF_VERSION.bump_patch()),
    ],
)
def test_import_warns_once_on_newer_export(
    dataset_name: str, tempdir: Path, warnings_log: list[str], version: str
):
    """A newer stamp warns exactly once, not once per split.

    `NativeParser.from_dir` parses train, val and test, and the stamp sits
    above all three.
    """
    root = _newer_export(dataset_name, tempdir, version)
    _import(root, f"{dataset_name}_imported", tempdir)

    warned = [msg for msg in warnings_log if "declares LDF" in msg]
    assert len(warned) == 1
    assert version in warned[0]


@pytest.mark.parametrize("version", ["2.0.0", None])
def test_import_is_quiet_for_current_or_older_exports(
    dataset_name: str,
    tempdir: Path,
    warnings_log: list[str],
    version: str | None,
):
    """An older export is a strict subset of what this version accepts."""
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(dataset, tempdir, "quiet", ldf_version=version)
    _import(root, f"{dataset_name}_imported", tempdir)

    assert not [msg for msg in warnings_log if "declares LDF" in msg]


# ``None`` removes the stamp altogether, standing for an export made
# before the stamp existed.
@pytest.mark.parametrize(
    "stamp", ['{"ldf_version": "banana"}', "{}", "{", None]
)
def test_damaged_or_missing_stamp_does_not_break_the_import(
    dataset_name: str,
    tempdir: Path,
    warnings_log: list[str],
    stamp: str | None,
):
    """The stamp is best-effort, so neither case may fail the parse."""
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    root = _export(dataset, tempdir, "damaged")
    if stamp is None:
        (root / "metadata.json").unlink()
    else:
        (root / "metadata.json").write_text(stamp)

    _import(root, f"{dataset_name}_imported", tempdir)

    # Nothing readable says the export is newer, so nothing to report.
    assert not [msg for msg in warnings_log if "declares LDF" in msg]


# `DatasetType` mixes in `str`, so the bare string is a real calling
# convention and must reach the same error as the enum.
@pytest.mark.parametrize("dataset_type", [DatasetType.COCO, "coco"])
def test_ldf_version_rejected_for_non_native(
    dataset_name: str, tempdir: Path, dataset_type: DatasetType
):
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    out = tempdir / "coco"
    with pytest.raises(ValueError, match="only applies to the native format"):
        dataset.export(out, dataset_type=dataset_type, ldf_version="2.0")
    assert not out.exists(), "must fail before creating the output directory"


def test_invalid_version_leaves_no_output_directory(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(
        dataset_name, _generator(tempdir), splits=(1, 0, 0)
    )
    out = tempdir / "bad"
    with pytest.raises(ValueError, match="Invalid LDF version"):
        dataset.export(out, ldf_version="banana")
    assert not out.exists(), "must fail before creating the output directory"
