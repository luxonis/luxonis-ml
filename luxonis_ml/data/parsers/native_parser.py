import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger
from semver.version import Version

from luxonis_ml.data import DatasetIterator
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.typing import PathType
from luxonis_ml.utils.path import resolve_manifest_path

from .base_parser import BaseParser, ParserOutput

# Annotation fields holding a path to a companion file, as ``(field, key)``.
# These are written relative to ``annotations.json`` so a dataset directory
# stays portable, and have to be resolved before the record is validated.
_ANNOTATION_PATHS: tuple[tuple[str, str], ...] = (
    ("segmentation", "mask"),
    ("instance_segmentation", "mask"),
    ("array", "path"),
)


class NativeParser(BaseParser):
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

    _SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checked_stamps: set[Path] = set()

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        annotation_path = split_path / "annotations.json"
        if not annotation_path.exists():
            return None
        return {"annotation_path": annotation_path}

    def from_dir(
        self, dataset_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
        added_train_imgs = self._parse_split(
            annotation_path=dataset_dir / "train" / "annotations.json",
        )
        added_val_imgs = self._parse_split(
            annotation_path=dataset_dir / "val" / "annotations.json",
        )
        added_test_imgs = self._parse_split(
            annotation_path=dataset_dir / "test" / "annotations.json",
        )
        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(self, annotation_path: Path) -> ParserOutput:
        """Parse native LDF annotations.

        Args:
            annotation_path: JSON file with annotations.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        self._warn_on_newer_export(annotation_path)
        data = json.loads(annotation_path.read_text())

        def generator() -> DatasetIterator:
            for record in data:
                with suppress(KeyError):
                    # An older manifest names the media with the keys a
                    # record now accepts only as deprecated ones.
                    for key in ("media", "file", "files"):
                        if key in record:
                            record["media"] = _resolve_media(
                                record.pop(key), annotation_path.parent
                            )
                            break
                annotation = record.get("annotation")
                for detection in (
                    annotation
                    if isinstance(annotation, list)
                    else [annotation]
                ):
                    if isinstance(detection, dict):
                        _resolve_annotation_paths(
                            detection, annotation_path.parent
                        )
                yield record

        added_images = self._get_added_images(generator())

        return generator(), {}, added_images

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
                json.loads(stamp.read_text())["ldf_version"],
                optional_minor_and_patch=True,
            )
            if version > LDF_VERSION:
                logger.warning(
                    f"'{stamp}' declares LDF {version}, but this luxonis-ml "
                    f"reads LDF {LDF_VERSION}. Upgrade it if parsing fails."
                )


def _resolve_media(media: object, base_dir: Path) -> object:
    """Resolve one path, or a mapping of source names to paths.

    Anything else is returned untouched, so the record model reports it.
    """
    if isinstance(media, dict):
        return {
            source: resolve_manifest_path(base_dir, path)
            if isinstance(path, PathType)
            else path
            for source, path in media.items()
        }
    if isinstance(media, PathType):
        return resolve_manifest_path(base_dir, media)
    return media


def _resolve_annotation_paths(
    annotation: dict[str, Any], base_dir: Path
) -> None:
    """Rewrite one detection's companion-file paths in place.

    Args:
        annotation: A detection, as read from the manifest.
        base_dir: Directory that relative paths are resolved against.

    """
    for field, key in _ANNOTATION_PATHS:
        value = annotation.get(field)
        if isinstance(value, dict) and isinstance(value.get(key), PathType):
            value[key] = resolve_manifest_path(base_dir, value[key])
            if field == "array":
                # The manifest keeps the stored key, which the annotation
                # now accepts only as a deprecated name.
                value["data"] = value.pop(key)
    sub_detections = annotation.get("sub_detections")
    if isinstance(sub_detections, dict):
        for sub_detection in sub_detections.values():
            if isinstance(sub_detection, dict):
                _resolve_annotation_paths(sub_detection, base_dir)
