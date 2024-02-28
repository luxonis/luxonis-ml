from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from luxonis_ml.data import DatasetGenerator, LuxonisDataset

from .base_parser import BaseParser, ParserOutput


class YoloV4Parser(BaseParser):
    """Parses directory with YoloV4 annotations to LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   ├── ...
        │   ├── _annotations.txt
        │   └── _classes.txt
        ├── valid/
        └── test/

    This is one of the formats that can be generated by
    U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        if not split_path.exists():
            return None
        annotations = split_path / "_annotations.txt"
        classes = split_path / "_classes.txt"
        if not annotations.exists() or not classes.exists():
            return None
        return {
            "image_dir": split_path,
            "annotation_path": annotations,
            "classes_path": classes,
        }

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if YoloV4Parser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(
        self, dataset: LuxonisDataset, dataset_dir: Path
    ) -> Tuple[List[str], List[str], List[str]]:
        added_train_imgs = self._parse_split(
            dataset,
            image_dir=dataset_dir / "train",
            annotation_path=dataset_dir / "train" / "_annotations.txt",
            classes_path=dataset_dir / "train" / "_classes.txt",
        )
        added_val_imgs = self._parse_split(
            dataset,
            image_dir=dataset_dir / "valid",
            annotation_path=dataset_dir / "valid" / "_annotations.txt",
            classes_path=dataset_dir / "valid" / "_classes.txt",
        )
        added_test_imgs = self._parse_split(
            dataset,
            image_dir=dataset_dir / "test",
            annotation_path=dataset_dir / "test" / "_annotations.txt",
            classes_path=dataset_dir / "test" / "_classes.txt",
        )
        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(
        self, image_dir: Path, annotation_path: Path, classes_path: Path
    ) -> ParserOutput:
        """Parses annotations from YoloV4 format to LDF. Annotations include
        classification and object detection.

        @type image_dir: Path
        @param image_dir: Path to directory with images
        @type annotation_path: Path
        @param annotation_path: Path to annotation file
        @type classes_path: Path
        @param classes_path: Path to file with class names
        @rtype: L{ParserOutput}
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

                path = image_dir.absolute() / img_path
                if not path.exists():
                    continue

                file = str(path)

                img = Image.open(file)
                width, height = img.size

                for ann_data in data[1:]:
                    curr_ann_data = ann_data.split(",")
                    class_name = class_names[int(curr_ann_data[4])]
                    yield {
                        "file": file,
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
                        "file": file,
                        "class": class_name,
                        "type": "box",
                        "value": tuple(bbox_xywh),
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names.values()), {}, added_images
