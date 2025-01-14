import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class NativeParser(BaseParser):
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
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        annotation_path = split_path / "annotations.json"
        if not annotation_path.exists():
            return None
        return {"annotation_path": annotation_path}

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if NativeParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(
        self, dataset_dir: Path
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        added_train_imgs = self._parse_split(
            image_dir=dataset_dir / "train",
            annotation_dir=dataset_dir / "train",
        )
        added_val_imgs = self._parse_split(
            image_dir=dataset_dir / "valid",
            annotation_dir=dataset_dir / "valid",
        )
        added_test_imgs = self._parse_split(
            image_dir=dataset_dir / "test",
            annotation_dir=dataset_dir / "test",
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
        os.chdir(annotation_path.parent)

        def generator() -> DatasetIterator:
            yield from data

        added_images = self._get_added_images(generator())

        return generator(), {}, added_images
