import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator
from luxonis_ml.typing import PathType
from luxonis_ml.utils.path import resolve_manifest_path

from .parser_plugin import ParsedDataset, SplitParserPlugin


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

    def _parse_split(self, annotation_path: Path) -> ParsedDataset:
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
                for mask_type in ["segmentation", "instance_segmentation"]:
                    with suppress(KeyError):
                        mask = record["annotation"][mask_type]["mask"]
                        if isinstance(mask, PathType):
                            record["annotation"][mask_type]["mask"] = (
                                resolve_manifest_path(
                                    annotation_path.parent, mask
                                )
                            )
                yield record

        added_images = self._get_added_images(generator())

        return ParsedDataset(generator(), {}, added_images)
