import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from defusedxml.ElementTree import parse

from luxonis_ml.data import DatasetIterator
from luxonis_ml.data.utils.enums import ParserIssue
from luxonis_ml.utils.path import resolve_manifest_path

from .parser_plugin import SplitParserPlugin


class VOCParser(SplitParserPlugin):
    """Parse a directory with VOC annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img1.xml
        │   └── ...
        ├── valid/
        └── test/

    This is one of the formats that Roboflow can generate.

    `_split_files` is deliberately left unimplemented: ``<filename>`` may
    name an image that is not there and the annotation is then skipped, so
    a directory listing would report images that never yield a record.
    """

    dataset_types = ("voc",)

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None

        image_stems = {
            image.stem for image in VOCParser._list_images(split_path)
        }
        label_stems = {label.stem for label in split_path.glob("*.xml")}
        if not image_stems or image_stems != label_stems:
            return None
        return {"image_dir": split_path, "annotation_dir": split_path}

    def _split_records(
        self, image_dir: Path, annotation_dir: Path
    ) -> DatasetIterator:
        """Parse VOC annotations into LDF records.

        Annotations include classification and object detection. Each
        ``.xml`` document is parsed once and its records are yielded
        before the next one is opened.

        Args:
            image_dir: Directory with images.
            annotation_dir: Directory with ``.xml`` annotations.

        Yields:
            One record per bounding box, and one per box-less image.

        Raises:
            ValueError: If an annotation XML file cannot be parsed or a
                required XML tag is missing.

        """
        # The same for every annotation, so resolved once.
        base_dir = image_dir.absolute().resolve()

        for anno_xml in annotation_dir.glob("*.xml"):
            annotation_data = parse(anno_xml)
            root = annotation_data.getroot()
            if root is None:
                raise ValueError(f"Could not parse {anno_xml}")

            path = resolve_manifest_path(
                base_dir, self._xml_find(root, "filename")
            )
            if not path.exists():
                self._warn_skipped_annotation(
                    ParserIssue.MISSING_IMAGE,
                    "referenced image file does not exist",
                    source=anno_xml,
                    image=path,
                )
                continue

            size_item = root.find("size")
            assert size_item is not None
            height = float(self._xml_find(size_item, "height"))
            width = float(self._xml_find(size_item, "width"))

            file = str(path)
            boxed = False
            for object_item in root.findall("object"):
                # Read before the box check, so that an object without a
                # `name` is an error whether or not it carries a box.
                class_name = self._xml_find(object_item, "name")

                bbox_info = object_item.find("bndbox")
                if bbox_info is None:
                    continue

                xmin = float(self._xml_find(bbox_info, "xmin"))
                ymin = float(self._xml_find(bbox_info, "ymin"))
                xmax = float(self._xml_find(bbox_info, "xmax"))
                ymax = float(self._xml_find(bbox_info, "ymax"))
                boxed = True
                yield {
                    "file": file,
                    "annotation": {
                        "class": class_name,
                        "boundingbox": {
                            "x": xmin / width,
                            "y": ymin / height,
                            "w": (xmax - xmin) / width,
                            "h": (ymax - ymin) / height,
                        },
                    },
                }

            if not boxed:
                yield {"file": file, "annotation": None}

    @staticmethod
    def _xml_find(root: ET.Element, tag: str) -> str:
        item = root.find(tag)
        if item is not None and item.text is not None:
            return item.text
        raise ValueError(f"Could not find {tag} in {root}")
