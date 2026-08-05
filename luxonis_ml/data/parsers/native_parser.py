import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator
from luxonis_ml.typing import PathType
from luxonis_ml.utils.path import resolve_manifest_path

from .base_parser import BaseParser, ParserOutput

#: Annotation fields holding a path to a companion file, as ``(field, key)``.
#: These are written relative to ``annotations.json`` so a dataset directory
#: stays portable, and have to be resolved before the record is validated.
_ANNOTATION_PATHS: tuple[tuple[str, str], ...] = (
    ("segmentation", "mask"),
    ("instance_segmentation", "mask"),
    ("array", "path"),
)


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
    sub_detections = annotation.get("sub_detections")
    if isinstance(sub_detections, dict):
        for sub_detection in sub_detections.values():
            if isinstance(sub_detection, dict):
                _resolve_annotation_paths(sub_detection, base_dir)


class NativeParser(BaseParser):
    """Parse a directory with native LDF annotations.

    Expected format::

        dataset_dir/
        ├── train/
        │   └── annotations.json
        ├── val/
        └── test/

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
        data = json.loads(annotation_path.read_text())

        def generator() -> DatasetIterator:
            for record in data:
                with suppress(KeyError):
                    if "file" in record:
                        record["file"] = resolve_manifest_path(
                            annotation_path.parent, record["file"]
                        )
                    elif "files" in record:
                        for key, value in record["files"].items():
                            if isinstance(value, PathType):
                                record["files"][key] = resolve_manifest_path(
                                    annotation_path.parent, value
                                )
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
