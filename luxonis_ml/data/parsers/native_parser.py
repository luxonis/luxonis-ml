import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator
from luxonis_ml.typing import PathType
from luxonis_ml.utils.path import resolve_manifest_path

from .base_parser import BaseParser, ParserOutput


class NativeParser(BaseParser):
    SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
    """Parse a directory with native LDF annotations.

    Expected format::

        dataset_dir/
        ├── train/
        │   └── annotations.json
        ├── valid/
        └── test/

    The annotations are stored in a single JSON file as a list of dictionaries
    in the same format as the output of the generator function used
    by ``BaseDataset.add``.
    """

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

        return generator(), {}, added_images
