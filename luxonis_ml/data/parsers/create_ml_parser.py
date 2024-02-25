import json
import os.path as osp
from pathlib import Path

import cv2

from luxonis_ml.data import DatasetGenerator

from .luxonis_parser import LuxonisParser, ParserOutput


class CreateMLParser(LuxonisParser):
    def validate(self, dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if not split_path.exists():
                return False
            if not (split_path / "_annotations.createml.json").exists():
                return False
            if not self._list_images(split_path):
                return False
        return True

    def from_dir(self, dataset_dir: str) -> None:
        """Parses directory with CreateML annotations to LDF.

        Expected format::
            dataset_dir/
            ├── train/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            │   └── _annotations.createml.json
            ├── valid/
            └── test/

        This is the default format returned when using U{Roboflow <https://roboflow.com/>}.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """

        added_train_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "train"),
            annotation_path=osp.join(
                dataset_dir, "train", "_annotations.createml.json"
            ),
        )
        added_val_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "valid"),
            annotation_path=osp.join(
                dataset_dir, "valid", "_annotations.createml.json"
            ),
        )
        added_test_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "test"),
            annotation_path=osp.join(dataset_dir, "test", "_annotations.createml.json"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    def _from_format(self, image_dir: str, annotation_path: str) -> ParserOutput:
        """Parses annotations from CreateML format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_path: str
        @param annotation_path: Path to annotation json file
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        with open(annotation_path) as f:
            annotations_data = json.load(f)

        class_names = set()
        images_annotations = []
        for annotations in annotations_data:
            path = osp.join(osp.abspath(image_dir), annotations["image"])
            if not osp.exists(path):
                continue
            else:
                img = cv2.imread(path)
                shape = img.shape
                height, width = shape[0], shape[1]

            curr_annotations = {"path": path, "classes": [], "bboxes": []}
            for curr_ann in annotations["annotations"]:
                class_name = curr_ann["label"]
                curr_annotations["classes"].append(class_name)
                class_names.add(class_name)

                bbox_ann = curr_ann["coordinates"]
                bbox_xywh = [
                    (bbox_ann["x"] - bbox_ann["width"] / 2) / width,
                    (bbox_ann["y"] - bbox_ann["height"] / 2) / height,
                    bbox_ann["width"] / width,
                    bbox_ann["height"] / height,
                ]
                curr_annotations["bboxes"].append((class_name, bbox_xywh))
            images_annotations.append(curr_annotations)

        def generator() -> DatasetGenerator:
            for curr_annotations in images_annotations:
                path = curr_annotations["path"]
                for class_name in curr_annotations["classes"]:
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }
                for bbox_class, bbox in curr_annotations["bboxes"]:
                    yield {
                        "file": path,
                        "class": bbox_class,
                        "type": "box",
                        "value": tuple(bbox),
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names), {}, added_images
