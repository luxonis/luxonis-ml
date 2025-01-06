from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class YoloV6Parser(BaseParser):
    """Parses annotations from YoloV6 annotations to LDF.

    Expected format::

        dataset_dir/
        ├── images/
        │   ├── train/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   ├── valid/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
        │   │   └── ...
        │   ├── valid/
        │   └── test/
        └── data.yaml


    C{data.yaml} contains names of all present classes.

    This is one of the formats that can be generated by
    U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        label_split = split_path.parent.parent / "labels" / split_path.name
        if not split_path.exists():
            return None
        if not label_split.exists():
            return None

        labels = label_split.glob("*.txt")
        images = BaseParser._list_images(split_path)
        if not BaseParser._compare_stem_files(images, labels):
            return None
        data_yaml = split_path.parent.parent / "data.yaml"
        if not data_yaml.exists():
            return None
        return {
            "image_dir": split_path,
            "annotation_dir": label_split,
            "classes_path": data_yaml,
        }

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            img_split = dataset_dir / "images" / split
            if YoloV6Parser.validate_split(img_split) is None:
                return False
        return True

    def from_dir(
        self, dataset_dir: Path
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        classes_path = dataset_dir / "data.yaml"
        added_train_imgs = self._parse_split(
            image_dir=dataset_dir / "images" / "train",
            annotation_dir=dataset_dir / "labels" / "train",
            classes_path=classes_path,
        )
        added_val_imgs = self._parse_split(
            image_dir=dataset_dir / "images" / "valid",
            annotation_dir=dataset_dir / "labels" / "valid",
            classes_path=classes_path,
        )
        added_test_imgs = self._parse_split(
            image_dir=dataset_dir / "images" / "test",
            annotation_dir=dataset_dir / "labels" / "test",
            classes_path=classes_path,
        )

        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(
        self, image_dir: Path, annotation_dir: Path, classes_path: Path
    ) -> ParserOutput:
        """Parses annotations from YoloV6 format to LDF. Annotations
        include classification and object detection.

        @type image_dir: Path
        @param image_dir: Path to directory with images
        @type annotation_dir: Path
        @param annotation_dir: Path to directory with annotations
        @type classes_path: Path
        @param classes_path: Path to yaml file with classes names
        @rtype: L{ParserOutput}
        @return: Annotation generator, list of classes names, skeleton
            dictionary for keypoints and list of added images.
        """
        with open(classes_path) as f:
            classes_data = yaml.safe_load(f)
        class_names = {
            i: class_name for i, class_name in enumerate(classes_data["names"])
        }

        def generator() -> DatasetIterator:
            for img_path in self._list_images(image_dir):
                ann_path = annotation_dir / img_path.with_suffix(".txt").name

                with open(ann_path) as f:
                    annotation_data = f.readlines()

                for ann_line in annotation_data:
                    class_id, x_center, y_center, width, height = [
                        x for x in ann_line.split()
                    ]
                    class_name = class_names[int(class_id)]

                    yield {
                        "file": str(img_path),
                        "annotation": {
                            "class": class_name,
                            "boundingbox": {
                                "x": float(x_center) - float(width) / 2,
                                "y": float(y_center) - float(height) / 2,
                                "w": float(width),
                                "h": float(height),
                            },
                        },
                    }

        added_images = self._get_added_images(generator())

        return generator(), {}, added_images
