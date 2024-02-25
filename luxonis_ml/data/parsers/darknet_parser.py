import os
import os.path as osp
from pathlib import Path

from luxonis_ml.data import DatasetGenerator

from .luxonis_parser import LuxonisParser, ParserOutput


class DarknetParser(LuxonisParser):
    def validate(self, dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if not (split_path / "_darknet.labels").exists():
                return False
            images = self._list_images(split_path)
            labels = split_path.glob("*.txt")
            if not self._compare_stem_files(images, labels):
                return False
        return True

    def from_dir(self, dataset_dir: str) -> None:
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

        This is the default format returned when using U{Roboflow <https://roboflow.com/>}.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "train"),
            classes_path=osp.join(dataset_dir, "train", "_darknet.labels"),
        )
        added_val_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "valid"),
            classes_path=osp.join(dataset_dir, "valid", "_darknet.labels"),
        )
        added_test_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "test"),
            classes_path=osp.join(dataset_dir, "test", "_darknet.labels"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    def _from_format(self, image_dir: str, classes_path: str) -> ParserOutput:
        """Parses annotations from Darknet format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type classes_path: str
        @param classes_path: Path to file with class names
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        with open(classes_path) as f:
            class_names = {i: line.rstrip() for i, line in enumerate(f.readlines())}

        def generator() -> DatasetGenerator:
            images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
            for img_path in images:
                path = osp.join(osp.abspath(image_dir), img_path)
                ann_path = osp.join(image_dir, img_path.replace(".jpg", ".txt"))
                with open(ann_path) as f:
                    annotation_data = f.readlines()

                for ann_line in annotation_data:
                    class_id, x_center, y_center, width, height = [
                        x for x in ann_line.split(" ")
                    ]
                    class_name = class_names[int(class_id)]

                    yield {
                        "file": path,
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
                        "file": path,
                        "class": class_name,
                        "type": "box",
                        "value": bbox_xywh,
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names.values()), {}, added_images
