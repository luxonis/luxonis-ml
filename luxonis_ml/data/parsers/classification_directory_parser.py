from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class ClassificationDirectoryParser(BaseParser):
    """Parses directory with ClassificationDirectory annotations to LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── class1/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   ├── class2/
        │   └── ...
        ├── valid/
        └── test/

    This is one of the formats that can be generated by
    U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        if not split_path.exists():
            return None
        classes = [
            d
            for d in split_path.iterdir()
            if d.is_dir() and d.name not in ["train", "valid", "test"]
        ]
        if not classes:
            return None
        fnames = [f for f in split_path.iterdir() if f.is_file()]
        if fnames:
            return None
        return {"class_dir": split_path}

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if ClassificationDirectoryParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(self, dataset_dir: Path) -> Tuple[List[str], List[str], List[str]]:
        added_train_imgs = self._parse_split(class_dir=dataset_dir / "train")
        added_val_imgs = self._parse_split(class_dir=dataset_dir / "valid")
        added_test_imgs = self._parse_split(class_dir=dataset_dir / "test")
        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(self, class_dir: Path) -> ParserOutput:
        """Parses annotations from classification directory format to LDF. Annotations
        include classification.

        @type class_dir: Path
        @param class_dir: Path to top level directory
        @rtype: L{ParserOutput}
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        class_names = [d.name for d in class_dir.iterdir() if d.is_dir()]

        def generator() -> DatasetIterator:
            for class_name in class_names:
                for img_path in (class_dir / class_name).iterdir():
                    yield {
                        "file": str(img_path.absolute()),
                        "annotation": {
                            "class": class_name,
                        },
                    }

        added_images = self._get_added_images(generator())

        return generator(), class_names, {}, added_images
