import os
import os.path as osp
from pathlib import Path

import yaml

from luxonis_ml.data import DatasetGenerator

from .luxonis_parser import LuxonisParser, ParserOutput


class YoloV6Parser(LuxonisParser):
    def validate(self, dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            img_split = dataset_dir / "images" / split
            label_split = dataset_dir / "labels" / split
            if not img_split.exists():
                return False
            if not label_split.exists():
                return False

            labels = label_split.glob("*.txt")
            images = self._list_images(img_split)
            if not self._compare_stem_files(images, labels):
                return False
        return True

    def from_dir(self, dataset_dir: str) -> None:
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

        This is the default format returned when using U{Roboflow <https://roboflow.com/>}.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        classes_path = osp.join(dataset_dir, "data.yaml")
        added_train_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "images", "train"),
            annotation_dir=osp.join(dataset_dir, "labels", "train"),
            classes_path=classes_path,
        )
        added_val_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "images", "valid"),
            annotation_dir=osp.join(dataset_dir, "labels", "valid"),
            classes_path=classes_path,
        )
        added_test_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "images", "test"),
            annotation_dir=osp.join(dataset_dir, "labels", "test"),
            classes_path=classes_path,
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    def _from_format(
        self, image_dir: str, annotation_dir: str, classes_path: str
    ) -> ParserOutput:
        """Parses annotations from YoloV6 format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_dir: str
        @param annotation_dir: Path to directory with annotations
        @type classes_path: str
        @param classes_path: Path to yaml file with classes names
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        with open(classes_path) as f:
            classes_data = yaml.safe_load(f)
        class_names = {
            i: class_name for i, class_name in enumerate(classes_data["names"])
        }

        def generator() -> DatasetGenerator:
            for ann_file in os.listdir(annotation_dir):
                ann_path = osp.join(osp.abspath(annotation_dir), ann_file)
                path = osp.join(
                    osp.abspath(image_dir), ann_file.replace(".txt", ".jpg")
                )
                if not osp.exists(path):
                    continue

                with open(ann_path) as f:
                    annotation_data = f.readlines()

                for ann_line in annotation_data:
                    class_id, x_center, y_center, width, height = [
                        x for x in ann_line.split()
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
