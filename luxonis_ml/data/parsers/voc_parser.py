import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from luxonis_ml.data import DatasetGenerator, LuxonisDataset

from .base_parser import BaseParser, ParserOutput


class VOCParser(BaseParser):
    """Parses directory with VOC annotations to LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img1.xml
        │   └── ...
        ├── valid/
        └── test/

    This is one of the formats that can be generated by
    U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        if not split_path.exists():
            return None

        images = BaseParser._list_images(split_path)
        labels = split_path.glob("*.xml")
        if not BaseParser._compare_stem_files(images, labels):
            return None
        return {"image_dir": split_path, "annotation_dir": split_path}

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if VOCParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(
        self, dataset: LuxonisDataset, dataset_dir: Path
    ) -> Tuple[List[str], List[str], List[str]]:
        added_train_imgs = self._parse_split(
            dataset,
            image_dir=dataset_dir / "train",
            annotation_dir=dataset_dir / "train",
        )
        added_val_imgs = self._parse_split(
            dataset,
            image_dir=dataset_dir / "valid",
            annotation_dir=dataset_dir / "valid",
        )
        added_test_imgs = self._parse_split(
            dataset,
            image_dir=dataset_dir / "test",
            annotation_dir=dataset_dir / "test",
        )
        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(
        self,
        image_dir: Path,
        annotation_dir: Path,
    ) -> ParserOutput:
        """Parses annotations from VOC format to LDF. Annotations include classification
        and object detection.

        @type image_dir: Path
        @param image_dir: Path to directory with images
        @type annotation_dir: Path
        @param annotation_dir: Path to directory with C{.xml} annotations
        @rtype: L{ParserOutput}
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """

        class_names = set()
        images_annotations = []
        for anno_xml in annotation_dir.glob("*.xml"):
            annotation_data = ET.parse(anno_xml)
            root = annotation_data.getroot()

            path = image_dir.absolute() / self._xml_find(root, "filename")
            if not path.exists():
                continue

            curr_annotations = {"path": path, "classes": [], "bboxes": []}
            size_item = root.find("size")
            assert size_item is not None
            height = float(self._xml_find(size_item, "height"))
            width = float(self._xml_find(size_item, "width"))

            for object_item in root.findall("object"):
                class_name = self._xml_find(object_item, "name")
                curr_annotations["classes"].append(class_name)
                class_names.add(class_name)

                bbox_info = object_item.find("bndbox")
                if bbox_info is not None:
                    bbox_xywh = np.array(
                        [
                            float(self._xml_find(bbox_info, "xmin")),
                            float(self._xml_find(bbox_info, "ymin")),
                            float(self._xml_find(bbox_info, "xmax"))
                            - float(self._xml_find(bbox_info, "xmin")),
                            float(self._xml_find(bbox_info, "ymax"))
                            - float(self._xml_find(bbox_info, "ymin")),
                        ]
                    )
                    bbox_xywh[::2] /= width
                    bbox_xywh[1::2] /= height
                    bbox_xywh = bbox_xywh.tolist()
                    curr_annotations["bboxes"].append((class_name, bbox_xywh))
            images_annotations.append(curr_annotations)

        def generator() -> DatasetGenerator:
            for curr_annotations in images_annotations:
                path = str(curr_annotations["path"])
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

    @staticmethod
    def _xml_find(root: ET.Element, tag: str) -> str:
        item = root.find(tag)
        if item is not None and item.text is not None:
            return item.text
        raise ValueError(f"Could not find {tag} in {root}")
