import csv
import os.path as osp
from pathlib import Path

import numpy as np

from luxonis_ml.data import DatasetGenerator

from .luxonis_parser import LuxonisParser, ParserOutput


class TensorflowCSVParser(LuxonisParser):
    def validate(self, dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if not split_path.exists():
                return False
            if not self._list_images(split_path):
                return False
            if not (split_path / "_annotations.csv").exists():
                return False
        return True

    def from_dir(self, dataset_dir: str) -> None:
        """Parses directory with TensorflowCSV annotations to LDF.

        Expected format::

            dataset_dir/
            ├── train/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   ├── ...
            │   └── _annotations.csv
            ├── valid/
            └── test/

        This is the default format returned when using U{Roboflow <https://roboflow.com/>}.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "train"),
            annotation_path=osp.join(dataset_dir, "train", "_annotations.csv"),
        )
        added_val_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "valid"),
            annotation_path=osp.join(dataset_dir, "valid", "_annotations.csv"),
        )
        added_test_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "test"),
            annotation_path=osp.join(dataset_dir, "test", "_annotations.csv"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    def _from_format(self, image_dir: str, annotation_path: str) -> ParserOutput:
        """Parses annotations from TensorflowCSV format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_path: str
        @param annotation_path: Path to annotation CSV file
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
        """
        with open(annotation_path) as f:
            reader = csv.reader(f, delimiter=",")

            class_names = set()
            images_annotations = {}
            for i, row in enumerate(reader):
                if i == 0:
                    idx_fname = row.index("filename")
                    idx_class = row.index("class")
                    idx_xmin = row.index("xmin")
                    idx_ymin = row.index("ymin")
                    idx_xmax = row.index("xmax")
                    idx_ymax = row.index("ymax")
                    idx_height = row.index("height")
                    idx_width = row.index("width")
                else:
                    path = osp.join(osp.abspath(image_dir), row[idx_fname])
                    if not osp.exists(path):
                        continue
                    if path not in images_annotations:
                        images_annotations[path] = {
                            "classes": [],
                            "bboxes": [],
                        }

                    class_name = row[idx_class]
                    images_annotations[path]["classes"].append(class_name)
                    class_names.add(class_name)

                    height = float(row[idx_height])
                    width = float(row[idx_width])
                    xmin = float(row[idx_xmin])
                    ymin = float(row[idx_ymin])
                    xmax = float(row[idx_xmax])
                    ymax = float(row[idx_ymax])
                    bbox_xywh = np.array([xmin, ymin, xmax - xmin, ymax - ymin])
                    bbox_xywh[::2] /= width
                    bbox_xywh[1::2] /= height
                    bbox_xywh = bbox_xywh.tolist()
                    images_annotations[path]["bboxes"].append((class_name, bbox_xywh))

        def generator() -> DatasetGenerator:
            for path in images_annotations:
                curr_annotations = images_annotations[path]
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
