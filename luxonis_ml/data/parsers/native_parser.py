"""Parser for datasets in the native LDF format."""

import json
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger
from semver.version import Version
from typing_extensions import override

from luxonis_ml.data import DatasetIterator, ParseIssueCollector
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.typing import PathType
from luxonis_ml.utils.path import resolve_manifest_path

from .parser_plugin import SplitParserPlugin

# Annotation fields holding a path to a companion file, as ``(field, key)``.
# These are written relative to ``annotations.json`` so a dataset directory
# stays portable, and have to be resolved before the record is validated.
_ANNOTATION_PATHS: tuple[tuple[str, str], ...] = (
    ("segmentation", "mask"),
    ("instance_segmentation", "mask"),
    ("array", "path"),
)


class NativeParser(SplitParserPlugin):
    """Parse a directory with native LDF annotations.

    Expected format::

        dataset_dir/
        ├── metadata.json  (optional; names the LDF version written)
        ├── train/
        │   └── annotations.json
        ├── val/
        └── test/

    ``metadata.json`` is a version stamp written by `NativeExporter`. It is
    never required, and is read only to warn about exports from a *newer*
    LDF version.

    The annotations are stored in a single JSON file as a list of dictionaries
    in the same format as the output of the generator function used by
    `BaseDataset.add`.

    ``sample_metadata`` is read as **record-level metadata** and preserved on
    the resulting `DatasetRecord`. It is distinct from
    ``annotation["metadata"]``, which creates metadata label tasks.

    Example ``annotations.json`` entry:

        .. code-block:: json

            {
              "file": "images/0.jpg",
              "task_name": "detection",

              "sample_metadata": {
                "record_id": 123,
                "camera": "left",
                "tags": ["night", "warehouse"]
              },

              "annotation": {
                "class": "person",
                "boundingbox": {
                  "x": 0.1,
                  "y": 0.2,
                  "w": 0.3,
                  "h": 0.4
                }
              }
            }

    """

    dataset_types = ("native",)
    split_names = ("train", "val", "test")

    def __init__(self, issues: ParseIssueCollector) -> None:
        """Create the parser.

        Args:
            issues: Collector for non-fatal parser issues.

        """
        super().__init__(issues)
        self._checked_stamps: set[Path] = set()

    @staticmethod
    @override
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        annotation_path = split_path / "annotations.json"
        if not annotation_path.exists():
            return None
        return {"annotation_path": annotation_path}

    @override
    def _split_records(self, annotation_path: Path) -> DatasetIterator:
        """Stream native LDF annotations with their paths resolved.

        The records are the JSON document itself, so a record costs no
        more than rewriting the media and mask references it names into
        resolved paths.

        `_split_files` is deliberately left unimplemented: listing a
        split's files means reading and resolving the whole document,
        which is the parse, and the importer's fallback already does that.

        Args:
            annotation_path: JSON file with annotations.

        Yields:
            Annotation records, with every path they name resolved.

        """
        self._warn_on_newer_export(annotation_path)
        base_dir = annotation_path.parent
        resolved: dict[str, Path] = {}

        def resolve(value: PathType) -> Path:
            # A split names the same media once per annotation, and for a
            # fixed base directory the result depends only on the spelling.
            key = str(value)
            path = resolved.get(key)
            if path is None:
                path = resolved[key] = resolve_manifest_path(base_dir, key)
            return path

        for record in json.loads(annotation_path.read_text(encoding="utf-8")):
            _resolve_record_paths(record, resolve)
            yield record

    def _warn_on_newer_export(self, annotation_path: Path) -> None:
        """Warn when the export was written by a newer LDF version.

        An older export is a strict subset of what this version accepts,
        so only a newer one is worth reporting. Comparing the whole
        version matters: ``sample_metadata`` arrived in a minor bump, so
        a same-major export can be just as unreadable.

        The stamp is best-effort: exports predating it have none, and a
        damaged one must not break the parse.
        """
        stamp = annotation_path.parent.parent / "metadata.json"
        if stamp in self._checked_stamps:
            return
        self._checked_stamps.add(stamp)
        with suppress(OSError, ValueError, TypeError, KeyError):
            version = Version.parse(
                json.loads(stamp.read_text(encoding="utf-8"))["ldf_version"],
                optional_minor_and_patch=True,
            )
            if version > LDF_VERSION:
                logger.warning(
                    f"'{stamp}' declares LDF {version}, but this luxonis-ml "
                    f"reads LDF {LDF_VERSION}. Upgrade it if parsing fails."
                )


def _resolve_record_paths(
    record: dict[str, Any],
    resolve: Callable[[PathType], Path],
) -> None:
    """Rewrite the file paths of one record in place.

    The image path or paths come first, then the companion files of each
    detection.

    Args:
        record: Record read from the native manifest.
        resolve: Cached resolver for paths relative to the manifest.

    """
    with suppress(KeyError):
        if "file" in record:
            record["file"] = resolve(record["file"])
        elif "files" in record:
            for key, value in record["files"].items():
                if isinstance(value, PathType):
                    record["files"][key] = resolve(value)

    annotation = record.get("annotation")
    detections = annotation if isinstance(annotation, list) else [annotation]
    for detection in detections:
        if isinstance(detection, dict):
            _resolve_annotation_paths(detection, resolve)


def _resolve_annotation_paths(
    annotation: dict[str, Any],
    resolve: Callable[[PathType], Path],
) -> None:
    """Rewrite one detection's companion-file paths in place.

    Args:
        annotation: A detection read from the native manifest.
        resolve: Cached resolver for paths relative to the manifest.

    """
    for field, key in _ANNOTATION_PATHS:
        value = annotation.get(field)
        if isinstance(value, dict) and isinstance(value.get(key), PathType):
            value[key] = resolve(value[key])
    sub_detections = annotation.get("sub_detections")
    if isinstance(sub_detections, dict):
        for sub_detection in sub_detections.values():
            if isinstance(sub_detection, dict):
                _resolve_annotation_paths(sub_detection, resolve)
