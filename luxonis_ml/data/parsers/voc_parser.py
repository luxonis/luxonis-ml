import os
import os.path as osp
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from luxonis_ml.data import DatasetGenerator

from .luxonis_parser import LuxonisParser, ParserOutput


class VOCParser(LuxonisParser):
    def validate(self, dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_content = os.listdir(dataset_dir / split)
            if "images" not in split_content or "annotations" not in split_content:
                return False

            images = self._list_images(dataset_dir / split / "images")
            labels = (dataset_dir / split / "annotations").glob("*.xml")
            if not self._compare_stem_files(images, labels):
                return False

        return True

    def from_dir(self, dataset_dir: str) -> None:
        """Parses directory with VOC annotations to LDF.

        Expected format::

            dataset_dir/
            ├── train/
            │   ├── images/
            │   │   ├── img1.jpg
            │   │   ├── img2.jpg
            │   │   └── ...
            │   └── annotations/
            │       ├── img1.xml
            │       ├── img2.xml
            │       └── ...
            ├── valid/
            └── test/


        This is the default format returned when using U{Roboflow <https://roboflow.com/>}.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "train"),
            annotation_dir=osp.join(dataset_dir, "train"),
        )
        added_val_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "valid"),
            annotation_dir=osp.join(dataset_dir, "valid"),
        )
        added_test_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "test"),
            annotation_dir=osp.join(dataset_dir, "test"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    def _from_format(
        self,
        image_dir: Path,
        annotation_dir: str,
    ) -> ParserOutput:
        """Parses annotations from VOC format to LDF. Annotations include classification
        and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_dir: str
        @param annotation_dir: Path to directory with .xml annotations
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        anno_files = [i for i in os.listdir(annotation_dir) if i.endswith(".xml")]

        class_names = set()
        images_annotations = []
        for anno_file in anno_files:
            anno_xml = osp.join(annotation_dir, anno_file)
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

    @staticmethod
    def _xml_find(root: ET.Element, tag: str) -> str:
        item = root.find(tag)
        if item is not None and item.text is not None:
            return item.text
        raise ValueError(f"Could not find {tag} in {root}")
