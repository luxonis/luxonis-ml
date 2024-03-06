from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from luxonis_ml.data import DatasetGenerator

from .base_parser import BaseParser, ParserOutput


class DarknetParser(BaseParser):
    """Parses directory with DarkNet annotations to LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img1.txt
        │   ├── ...
        │   └── _darknet.labels
        ├── valid/
        └── test/

    This is one of the formats that can be generated by
    U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        if not split_path.exists():
            return None
        if not (split_path / "_darknet.labels").exists():
            return None
        images = BaseParser._list_images(split_path)
        labels = split_path.glob("*.txt")
        if not BaseParser._compare_stem_files(images, labels):
            return None
        return {"image_dir": split_path, "classes_path": split_path / "_darknet.labels"}

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if DarknetParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(self, dataset_dir: Path) -> Tuple[List[str], List[str], List[str]]:
        added_train_imgs = self._parse_split(
            image_dir=dataset_dir / "train",
            classes_path=dataset_dir / "train" / "_darknet.labels",
        )
        added_val_imgs = self._parse_split(
            image_dir=dataset_dir / "valid",
            classes_path=dataset_dir / "valid" / "_darknet.labels",
        )
        added_test_imgs = self._parse_split(
            image_dir=dataset_dir / "test",
            classes_path=dataset_dir / "test" / "_darknet.labels",
        )
        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(self, image_dir: Path, classes_path: Path) -> ParserOutput:
        """Parses annotations from Darknet format to LDF. Annotations include
        classification and object detection.

        @type image_dir: Path
        @param image_dir: Path to directory with images
        @type classes_path: Path
        @param classes_path: Path to file with class names
        @rtype: L{ParserOutput}
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        with open(classes_path) as f:
            class_names = {i: line.rstrip() for i, line in enumerate(f.readlines())}

        def generator() -> DatasetGenerator:
            for img_path in self._list_images(image_dir):
                ann_path = img_path.with_suffix(".txt")
                file = str(img_path)
                with open(ann_path) as f:
                    annotation_data = f.readlines()

                for ann_line in annotation_data:
                    class_id, x_center, y_center, width, height = [
                        x for x in ann_line.split(" ")
                    ]
                    class_name = class_names[int(class_id)]

                    yield {
                        "file": file,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

                    bbox_xywh = (
                        float(x_center) - float(width) / 2,
                        float(y_center) - float(height) / 2,
                        float(width),
                        float(height),
                    )

                    yield {
                        "file": file,
                        "class": class_name,
                        "type": "box",
                        "value": bbox_xywh,
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names.values()), {}, added_images