"""Exact-output digests used to gate parser optimizations.

An optimization may make a parser faster but must not change what it
produces. `digest` reduces a parse to a canonical, path-independent
structure that can be captured before a change and compared after it.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from luxonis_ml.data.datasets.annotation import DatasetRecord
from luxonis_ml.data.datasets.base_dataset import _record_files
from luxonis_ml.data.parsers.parser_plugin import (
    ParseIssueCollector,
    get_parser_plugin,
)

from .harness import BenchmarkCase


def _root_of(case: BenchmarkCase) -> Path:
    """Return the directory paths in a digest are relative to.

    A source may be a single file (an NDJSON export), in which case the
    dataset lives beside it.
    """
    return case.source if case.source.is_dir() else case.source.parent


def _relative(value: Any, root: Path) -> Any:
    """Rewrite absolute paths under ``root`` to root-relative ones."""
    if isinstance(value, Path) or (
        isinstance(value, str) and ("/" in value or "\\" in value)
    ):
        path = Path(value)
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.name
    return value


def _canonical(value: Any, root: Path) -> Any:
    if isinstance(value, DatasetRecord):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _canonical(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item, root) for item in value]
    if isinstance(value, float):
        # Parsers differ in the last bits depending on how a coordinate
        # is computed; anything coarser than this would hide real
        # changes, anything finer reports noise.
        return round(value, 9)
    return _relative(value, root)


def digest(case: BenchmarkCase) -> dict[str, Any]:
    """Return a canonical description of what parsing ``case`` yields."""
    plugin_type, _, layout = get_parser_plugin(case.source, case.dataset_type)
    issues = ParseIssueCollector()
    plugin = plugin_type(issues)
    result = plugin.parse(case.source, layout, **case.parse_kwargs)

    root = _root_of(case)
    files: dict[Any, None] = {}
    split_files: dict[str, dict[Any, None]] = {}
    record_hashes = []
    for split_name, record in result.records:
        for file in _record_files(record):
            relative = _relative(file, root)
            files[relative] = None
            if split_name is not None:
                split_files.setdefault(split_name, {})[relative] = None
        payload = json.dumps(
            _canonical(record, root), sort_keys=True, default=str
        )
        record_hashes.append(hashlib.sha256(payload.encode()).hexdigest()[:16])

    # The old contract reported every split of a split-based source, empty
    # ones included, and reported `None` for a source parsed as a whole.
    # Reproducing that keeps digests taken before the API change valid.
    splits = (
        {
            name: list(split_files.get(name, {}))
            for name in ("train", "val", "test")
        }
        if layout.split_names
        else None
    )

    return {
        "dataset_type": case.dataset_type,
        "files": list(files),
        "splits": splits,
        "skeletons": _canonical(result.skeletons, root),
        "records": record_hashes,
        "issues": sorted(
            json.dumps(
                {
                    "issue": message.parser_issue.value,
                    "reason": message.reason,
                    "source": _relative(message.source, root),
                    "image": _relative(message.image, root),
                    "annotation_id": message.annotation_id,
                },
                sort_keys=True,
            )
            for message in issues.messages
        ),
    }


def _unique(files: Any) -> Any:
    """Deduplicate a file list while keeping its order.

    A digest identifies a file by its path relative to the dataset root,
    falling back to its name when it lies outside - which is what happens
    for a source given as a relative path. Two files of different splits
    can then share an entry, so the lists are compared as the ordered
    unique sequences the contract actually specifies.
    """
    if isinstance(files, list):
        return list(dict.fromkeys(files))
    if isinstance(files, dict):
        return {name: _unique(entries) for name, entries in files.items()}
    return files


def compare(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Return human-readable differences between two digests."""
    differences = []
    for key in ("files", "splits"):
        if _unique(before[key]) != _unique(after[key]):
            differences.append(f"{key} changed")
    for key in ("skeletons", "issues"):
        if before[key] != after[key]:
            differences.append(f"{key} changed")

    if before["records"] != after["records"]:
        if len(before["records"]) != len(after["records"]):
            differences.append(
                f"record count {len(before['records'])} -> "
                f"{len(after['records'])}"
            )
        else:
            changed = sum(
                1
                for old, new in zip(
                    before["records"], after["records"], strict=True
                )
                if old != new
            )
            differences.append(f"{changed} records changed")
    return differences
