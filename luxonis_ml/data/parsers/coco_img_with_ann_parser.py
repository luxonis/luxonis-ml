from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_parser import BaseParser
from .coco_parser import COCOParser


class COCOImgWithAnnParser(COCOParser):
    """Parses directory with COCO annotations to LDF. Unlike COCOParser, images and
    their annotations are in same directory.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        │   └── _annotations.coco.json
        ├── valid/
        └── test/

    This is one of the formats that can be generated by
    U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        if not split_path.exists():
            return None
        if not (split_path / "_annotations.coco.json").exists():
            return None
        if not BaseParser._list_images(split_path):
            return None
        return {
            "image_dir": split_path,
            "annotation_path": split_path / "_annotations.coco.json",
        }

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if COCOImgWithAnnParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(
        self,
        dataset_dir: Path,
    ) -> Tuple[List[str], List[str], List[str]]:
        train_ann_path = dataset_dir / "train" / "_annotations.coco.json"
        added_train_imgs = self._parse_split(
            image_dir=dataset_dir / "train",
            annotation_path=train_ann_path,
        )

        val_ann_path = dataset_dir / "valid" / "_annotations.coco.json"
        added_val_imgs = self._parse_split(
            image_dir=dataset_dir / "valid",
            annotation_path=val_ann_path,
        )

        test_ann_path = dataset_dir / "test" / "_annotations.coco.json"
        added_test_imgs = self._parse_split(
            image_dir=dataset_dir / "test",
            annotation_path=test_ann_path,
        )

        return added_train_imgs, added_val_imgs, added_test_imgs
