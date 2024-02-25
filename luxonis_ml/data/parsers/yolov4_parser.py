import os.path as osp
from pathlib import Path

import cv2

from luxonis_ml.data import DatasetGenerator

from .luxonis_parser import LuxonisParser, ParserOutput


class YoloV4Parser(LuxonisParser):
    def validate(self, dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if not split_path.exists():
                return False
            if not (split_path / "_annotations.txt").exists():
                return False
            if not (split_path / "_classes.txt").exists():
                return False
            images = self._list_images(split_path)
            labels = split_path.glob("*.txt")
            if not self._compare_stem_files(images, labels):
                return False
        return True

    def from_dir(self, dataset_dir: str) -> None:
        """Parses directory with YoloV4 annotations to LDF.

        Expected format::

            dataset_dir/
            ├── train/
            │   ├── img1.jpg
            │   ├── img1.txt
            │   ├── ...
            │   ├── _annotations.txt
            │   └── _classes.txt
            ├── valid/
            └── test/


        This is the default format returned when using U{Roboflow <https://roboflow.com/>}.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory.
        """
        added_train_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "train"),
            annotation_path=osp.join(dataset_dir, "train", "_annotations.txt"),
            classes_path=osp.join(dataset_dir, "train", "_classes.txt"),
        )
        added_val_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "valid"),
            annotation_path=osp.join(dataset_dir, "valid", "_annotations.txt"),
            classes_path=osp.join(dataset_dir, "valid", "_classes.txt"),
        )
        added_test_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "test"),
            annotation_path=osp.join(dataset_dir, "test", "_annotations.txt"),
            classes_path=osp.join(dataset_dir, "test", "_classes.txt"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    def _from_format(
        self, image_dir: str, annotation_path: str, classes_path: str
    ) -> ParserOutput:
        """Parses annotations from YoloV4 format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_path: str
        @param annotation_path: Path to annotation file
        @type classes_path: str
        @param classes_path: Path to file with class names
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        with open(classes_path) as f:
            class_names = {i: line.rstrip() for i, line in enumerate(f.readlines())}

        def generator() -> DatasetGenerator:
            with open(annotation_path) as f:
                annotation_data = [line.rstrip() for line in f.readlines()]

            for ann_line in annotation_data:
                data = ann_line.split(" ")
                img_path = data[0]

                path = osp.join(osp.abspath(image_dir), img_path)
                if not osp.exists(path):
                    continue
                else:
                    img = cv2.imread(path)
                    shape = img.shape
                    height, width = shape[0], shape[1]

                for ann_data in data[1:]:
                    curr_ann_data = ann_data.split(",")
                    class_name = class_names[int(curr_ann_data[4])]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

                    bbox_xyxy = [float(i) for i in curr_ann_data[:4]]
                    bbox_xywh = [
                        bbox_xyxy[0] / width,
                        bbox_xyxy[1] / height,
                        (bbox_xyxy[2] - bbox_xyxy[0]) / width,
                        (bbox_xyxy[3] - bbox_xyxy[1]) / height,
                    ]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "box",
                        "value": tuple(bbox_xywh),
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names.values()), {}, added_images
