import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator
from luxonis_ml.typing import PathType

from .base_parser import BaseParser, ParserOutput


class NativeParser(BaseParser):
    SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
    """Parses directory with native LDF annotations.

    Expected format::

        dataset_dir/
        ├── train/
        │   └── annotations.json
        ├── valid/
        └── test/

    The annotations are stored in a single JSON file as a list of dictionaries
    in the same format as the output of the generator function used
    in L{BaseDataset.add} method.
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
        """Parses annotations from LDF Format.

        @type annotation_path: C{Path}
        @param annotation_dir: Path to the JSON file with annotations.
        @rtype: L{ParserOutput}
        @return: Annotation generator, list of classes names, skeleton
            dictionary for keypoints and list of added images.
        """

        data = json.loads(annotation_path.read_text())

        def generator() -> DatasetIterator:
            for record in data:
                with suppress(KeyError):
                    if "file" in record:
                        record["file"] = (
                            annotation_path.parent / record["file"]
                        ).absolute()
                    elif "files" in record:
                        for key, value in record["files"].items():
                            if isinstance(value, PathType):
                                record["files"][key] = (
                                    annotation_path.parent / value
                                ).absolute()
                for mask_type in ["segmentation", "instance_segmentation"]:
                    with suppress(KeyError):
                        mask = record["annotation"][mask_type]["mask"]
                        if isinstance(mask, PathType):
                            record["annotation"][mask_type]["mask"] = (
                                annotation_path.parent / mask
                            ).absolute()
                yield record

        added_images = self._get_added_images(generator())

        return generator(), {}, added_images
