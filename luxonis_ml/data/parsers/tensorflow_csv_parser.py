from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from luxonis_ml.data import DatasetIterator
from luxonis_ml.utils.path import resolve_manifest_path

from .parser_plugin import ParsedDataset, SplitParserPlugin


class TensorflowCSVParser(SplitParserPlugin):
    """Parse a directory with TensorFlow CSV annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   ├── ...
        │   └── _annotations.csv
        ├── valid/
        └── test/

    This is one of the formats that Roboflow can generate.
    """

    dataset_types = ("tfcsv",)

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        if not TensorflowCSVParser._list_images(split_path):
            return None
        if not (split_path / "_annotations.csv").exists():
            return None
        return {
            "image_dir": split_path,
            "annotation_path": split_path / "_annotations.csv",
        }

    def _parse_split(
        self, image_dir: Path, annotation_path: Path
    ) -> ParsedDataset:
        """Parse TensorFlow CSV annotations into LDF records.

        Annotations include classification and object detection.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation CSV file.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        df = pl.read_csv(annotation_path).filter(
            pl.col("filename").is_not_null()
        )
        images_annotations = {}

        for row in df.rows(named=True):
            path = str(resolve_manifest_path(image_dir, str(row["filename"])))
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
            bbox_xywh = np.array(
                [xmin, ymin, xmax - xmin, ymax - ymin], dtype=float
            )
            bbox_xywh[::2] /= width
            bbox_xywh[1::2] /= height
            bbox_xywh = bbox_xywh.tolist()
            images_annotations[path]["bboxes"].append((class_name, bbox_xywh))

        def generator() -> DatasetIterator:
            for img_path in self._list_images(image_dir):
                path = str(img_path.resolve())
                curr_annotations = images_annotations.get(path, {"bboxes": []})
                if not curr_annotations["bboxes"]:
                    yield {"file": path, "annotation": None}
                    continue
                for bbox_class, (x, y, w, h) in curr_annotations["bboxes"]:
                    yield {
                        "file": path,
                        "annotation": {
                            "class": bbox_class,
                            "boundingbox": {
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                            },
                        },
                    }

        added_images = self._get_added_images(generator())

        return ParsedDataset(generator(), {}, added_images)
