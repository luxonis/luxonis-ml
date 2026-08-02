import json
from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator
from luxonis_ml.typing import PathType
from luxonis_ml.utils.path import resolve_manifest_path

from .parser_plugin import SplitParserPlugin

_MASK_KEYS = ("segmentation", "instance_segmentation")
"""Annotation keys whose ``mask`` may hold a path to a mask file."""


class NativeParser(SplitParserPlugin):
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

    dataset_types = ("native",)
    split_names = ("train", "val", "test")

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        annotation_path = split_path / "annotations.json"
        if not annotation_path.exists():
            return None
        return {"annotation_path": annotation_path}

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

        for record in json.loads(annotation_path.read_text()):
            if "file" in record:
                record["file"] = resolve(record["file"])
            elif "files" in record:
                media = record["files"]
                for key, value in media.items():
                    if isinstance(value, PathType):
                        media[key] = resolve(value)

            annotation = record.get("annotation")
            if isinstance(annotation, dict):
                for mask_key in _MASK_KEYS:
                    try:
                        segmentation = annotation[mask_key]
                        mask = segmentation["mask"]
                    except KeyError:
                        continue
                    if isinstance(mask, PathType):
                        segmentation["mask"] = resolve(mask)

            yield record
