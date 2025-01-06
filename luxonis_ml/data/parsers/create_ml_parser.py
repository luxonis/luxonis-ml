import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class CreateMLParser(BaseParser):
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

    This is one of the formats that can be generated by
    U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        if not split_path.exists():
            return None
        if not (split_path / "_annotations.createml.json").exists():
            return None
        if not BaseParser._list_images(split_path):
            return None
        return {
            "image_dir": split_path,
            "annotation_path": split_path / "_annotations.createml.json",
        }

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if CreateMLParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(
        self, dataset_dir: Path
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        added_train_imgs = self._parse_split(
            image_dir=dataset_dir / "train",
            annotation_path=dataset_dir
            / "train"
            / "_annotations.createml.json",
        )
        added_val_imgs = self._parse_split(
            image_dir=dataset_dir / "valid",
            annotation_path=dataset_dir
            / "valid"
            / "_annotations.createml.json",
        )
        added_test_imgs = self._parse_split(
            image_dir=dataset_dir / "test",
            annotation_path=dataset_dir
            / "test"
            / "_annotations.createml.json",
        )

        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(
        self, image_dir: Path, annotation_path: Path
    ) -> ParserOutput:
        """Parses annotations from CreateML format to LDF. Annotations
        include classification and object detection.

        @type image_dir: Path
        @param image_dir: Path to directory with images
        @type annotation_path: Path
        @param annotation_path: Path to annotation json file
        @rtype: L{ParserOutput}
        @return: Annotation generator, list of classes names, skeleton
            dictionary for keypoints and list of added images.
        """
        with open(annotation_path) as f:
            annotations_data = json.load(f)

        images_annotations = []
        for annotations in annotations_data:
            path = image_dir.absolute().resolve() / annotations["image"]
            if not path.exists():
                continue
            file = str(path)
            img = Image.open(file)
            width, height = img.size

            curr_annotations = {"path": str(path), "classes": [], "bboxes": []}
            for curr_ann in annotations["annotations"]:
                class_name = curr_ann["label"]
                curr_annotations["classes"].append(class_name)

                bbox_ann = curr_ann["coordinates"]
                bbox_xywh = [
                    (bbox_ann["x"] - bbox_ann["width"] / 2) / width,
                    (bbox_ann["y"] - bbox_ann["height"] / 2) / height,
                    bbox_ann["width"] / width,
                    bbox_ann["height"] / height,
                ]
                curr_annotations["bboxes"].append((class_name, bbox_xywh))
            images_annotations.append(curr_annotations)

        def generator() -> DatasetIterator:
            for curr_annotations in images_annotations:
                path = curr_annotations["path"]
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

        return generator(), {}, added_images
