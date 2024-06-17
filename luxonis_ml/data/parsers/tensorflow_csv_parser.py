from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class TensorflowCSVParser(BaseParser):
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

    This is one of the formats that can be generated by
    U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        if not split_path.exists():
            return None
        if not BaseParser._list_images(split_path):
            return None
        if not (split_path / "_annotations.csv").exists():
            return None
        return {
            "image_dir": split_path,
            "annotation_path": split_path / "_annotations.csv",
        }

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if TensorflowCSVParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(self, dataset_dir: Path) -> Tuple[List[str], List[str], List[str]]:
        added_train_imgs = self._parse_split(
            image_dir=dataset_dir / "train",
            annotation_path=dataset_dir / "train" / "_annotations.csv",
        )
        added_val_imgs = self._parse_split(
            image_dir=dataset_dir / "valid",
            annotation_path=dataset_dir / "valid" / "_annotations.csv",
        )
        added_test_imgs = self._parse_split(
            image_dir=dataset_dir / "test",
            annotation_path=dataset_dir / "test" / "_annotations.csv",
        )
        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(self, image_dir: Path, annotation_path: Path) -> ParserOutput:
        """Parses annotations from TensorflowCSV format to LDF. Annotations include
        classification and object detection.

        @type image_dir: Path
        @param image_dir: Path to directory with images
        @type annotation_path: Path
        @param annotation_path: Path to annotation CSV file
        @rtype: L{ParserOutput}
        @return: Annotation generator, list of classes names, skeleton dictionary for
        """
        df = pl.read_csv(annotation_path).filter(pl.col("filename").is_not_null())
        images_annotations = {}

        class_names = set(df["class"])
        for row in df.rows(named=True):
            path = str(image_dir / str(row["filename"]))
            if path not in images_annotations:
                images_annotations[path] = {
                    "classes": [],
                    "bboxes": [],
                }

            class_name = row["class"]
            images_annotations[path]["classes"].append(class_name)

            height = row["height"]
            width = row["width"]
            xmin = row["xmin"]
            ymin = row["ymin"]
            xmax = row["xmax"]
            ymax = row["ymax"]
            bbox_xywh = np.array([xmin, ymin, xmax - xmin, ymax - ymin], dtype=float)
            bbox_xywh[::2] /= width
            bbox_xywh[1::2] /= height
            bbox_xywh = bbox_xywh.tolist()
            images_annotations[path]["bboxes"].append((class_name, bbox_xywh))

        def generator() -> DatasetIterator:
            for path in images_annotations:
                curr_annotations = images_annotations[path]
                for class_name in curr_annotations["classes"]:
                    yield {
                        "file": path,
                        "annotation": {
                            "type": "classification",
                            "class": class_name,
                        },
                    }
                for bbox_class, (x, y, w, h) in curr_annotations["bboxes"]:
                    yield {
                        "file": path,
                        "annotation": {
                            "type": "boundingbox",
                            "class": bbox_class,
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                        },
                    }

        added_images = self._get_added_images(generator())

        return generator(), list(class_names), {}, added_images
