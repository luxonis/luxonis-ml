"""Exporting the native format at an older LDF version.

`DatasetRecord` forbids extra fields, so LDF 2.1 adding ``sample_metadata``
made every 2.1 export unreadable by an older luxonis-ml.
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
    _ADDED_FIELDS,
    LDFDowngrader,
    resolve_export_version,
)
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.enums.enums import DatasetType
from luxonis_ml.ldf import DatasetRecord

from .utils import create_dataset, create_image

#: Fields a pre-2.1 `DatasetRecord` accepts. Anything else trips
#: ``extra="forbid"`` on an older install.
LDF_2_0_RECORD_FIELDS = {"file", "files", "task_name", "annotation"}


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


@pytest.mark.parametrize("version", ["2.0", "2.0.0"])
def test_resolve_accepts_short_and_full_versions(version: str):
    assert resolve_export_version(version) == Version.parse("2.0.0")


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


def test_downgrade_removes_the_key_rather_than_emptying_it():
    """``sample_metadata: {}`` still fails ``extra="forbid"``."""
    downgraded = LDFDowngrader(Version.parse("2.0.0"))(
        {"file": "a.jpg", "task_name": "t", "sample_metadata": {"x": 1}}
    )
    assert "sample_metadata" not in downgraded


def test_downgrade_to_current_version_is_a_passthrough():
    record = {"file": "a.jpg", "sample_metadata": {"x": 1}}
    assert LDFDowngrader(LDF_VERSION)(dict(record)) == record


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
    """A 2.0 export re-imports cleanly, minus the dropped metadata."""
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


def test_export_2_0_warns_about_dropped_metadata(
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
