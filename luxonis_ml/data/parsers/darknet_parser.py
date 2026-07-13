from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class DarknetParser(BaseParser):
    """Parse a directory with Darknet annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img1.txt
        │   ├── ...
        │   └── _darknet.labels
        ├── valid/
        └── test/

    This is one of the formats that Roboflow can generate.
    """

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        if not (split_path / "_darknet.labels").exists():
            return None
        if not BaseParser._list_images(split_path):
            return None
        return {
            "image_dir": split_path,
            "classes_path": split_path / "_darknet.labels",
        }

    def from_dir(
        self, dataset_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
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
        """Parse Darknet annotations into LDF records.

        Annotations include classification and object detection.

        Args:
            image_dir: Directory with images.
            classes_path: File with class names.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        with open(classes_path) as f:
            class_names = {
                i: line.rstrip() for i, line in enumerate(f.readlines())
            }

        def generator() -> DatasetIterator:
            for img_path in self._list_images(image_dir):
                ann_path = img_path.with_suffix(".txt")
                file = str(img_path)
                if ann_path.exists():
                    with open(ann_path) as f:
                        annotation_data = f.readlines()
                else:
                    annotation_data = []

                if not annotation_data:
                    yield {"file": file, "annotation": None}
                    continue

                for ann_line in annotation_data:
                    class_id, x_center, y_center, width, height = list(
                        ann_line.split(" ")
                    )
                    class_name = class_names[int(class_id)]

                    yield {
                        "file": file,
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
