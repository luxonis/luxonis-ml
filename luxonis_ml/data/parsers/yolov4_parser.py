from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image
from typing_extensions import override

from luxonis_ml.data import DatasetIterator
from luxonis_ml.data.utils.enums import ParserIssue
from luxonis_ml.utils.path import resolve_manifest_path

from .base_parser import BaseParser, ParserOutput


class YoloV4Parser(BaseParser):
    """Parse a directory with YOLOv4 annotations into LDF.

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

    This is one of the formats that Roboflow can generate.
    """

    @staticmethod
    @override
    def validate_split(split_path: Path) -> dict[str, Any] | None:
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

    @override
    def from_dir(
        self, dataset_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
        added_train_imgs = self._parse_split(
            image_dir=dataset_dir / "train",
            annotation_path=dataset_dir / "train" / "_annotations.txt",
            classes_path=dataset_dir / "train" / "_classes.txt",
        )
        added_val_imgs = self._parse_split(
            image_dir=dataset_dir / "valid",
            annotation_path=dataset_dir / "valid" / "_annotations.txt",
            classes_path=dataset_dir / "valid" / "_classes.txt",
        )
        added_test_imgs = self._parse_split(
            image_dir=dataset_dir / "test",
            annotation_path=dataset_dir / "test" / "_annotations.txt",
            classes_path=dataset_dir / "test" / "_classes.txt",
        )
        return added_train_imgs, added_val_imgs, added_test_imgs

    @override
    def from_split(
        self, image_dir: Path, annotation_path: Path, classes_path: Path
    ) -> ParserOutput:
        """Parse YOLOv4 annotations into LDF records.

        Annotations include classification and object detection.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation file.
            classes_path: File with class names.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        with classes_path.open() as f:
            class_names = {
                i: line.rstrip() for i, line in enumerate(f.readlines())
            }

        records = partial(
            self._records, image_dir, annotation_path, class_names
        )
        added_images = self._get_added_images(records())

        return records(), {}, added_images

    def _records(
        self,
        image_dir: Path,
        annotation_path: Path,
        class_names: dict[int, str],
    ) -> DatasetIterator:
        """Yield the records of the annotation file, then the unlisted images."""
        with annotation_path.open() as f:
            annotation_data = [line.rstrip() for line in f]

        annotated_images: set[Path] = set()

        for ann_line in annotation_data:
            data = ann_line.split(" ")
            img_path = data[0]
            path = resolve_manifest_path(
                image_dir.absolute().resolve(), img_path
            )
            annotated_images.add(path.resolve())
            if not path.exists():
                self._warn_skipped_annotation(
                    ParserIssue.MISSING_IMAGE,
                    "referenced image file does not exist",
                    source=annotation_path,
                    image=path,
                )
                continue

            file = str(path)

            # Handle image names listed with no following annotation in the line
            if len(data) == 1:
                yield {"file": file, "annotation": None}
                continue

            with Image.open(file) as image:
                width, height = image.size
            for ann_data in data[1:]:
                yield {
                    "file": file,
                    "annotation": _box_annotation(
                        ann_data, class_names, width, height
                    ),
                }

        # Images in the directory not listed in annotations file
        for img_path in self._list_images(image_dir):
            if img_path.resolve() not in annotated_images:
                yield {"file": str(img_path), "annotation": None}


def _box_annotation(
    ann_data: str, class_names: dict[int, str], width: int, height: int
) -> dict[str, Any]:
    """Convert one ``x1,y1,x2,y2,class_id`` pixel box to a normalized LDF box."""
    curr_ann_data = ann_data.split(",")
    class_name = class_names[int(curr_ann_data[4])]
    x1, y1, x2, y2 = (float(i) for i in curr_ann_data[:4])
    return {
        "class": class_name,
        "boundingbox": {
            "x": x1 / width,
            "y": y1 / height,
            "w": (x2 - x1) / width,
            "h": (y2 - y1) / height,
        },
    }
