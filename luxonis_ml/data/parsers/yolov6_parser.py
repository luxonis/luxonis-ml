from pathlib import Path
from typing import Any, cast

import yaml
from typing_extensions import override

from luxonis_ml.data import DatasetIterator

from .parser_plugin import Layout, SplitParserPlugin


class YoloV6Parser(SplitParserPlugin):
    """Parse YOLOv6 annotations into LDF.

    Expected format::

        dataset_dir/
        ├── images/
        │   ├── train/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   ├── valid/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
        │   │   └── ...
        │   ├── valid/
        │   └── test/
        └── data.yaml


    ``data.yaml`` contains all class names.

    This is one of the formats that Roboflow can generate.
    """

    dataset_types = ("yolov6",)

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        label_split = split_path.parent.parent / "labels" / split_path.name
        if not split_path.exists():
            return None
        if not label_split.exists():
            return None

        images = YoloV6Parser._list_images(split_path)
        if not images:
            return None
        data_yaml = split_path.parent.parent / "data.yaml"
        if not data_yaml.exists():
            return None
        # The listing validation already paid for is handed to the parse as a
        # parse argument so that recognizing and parsing a split walk the image
        # directory once between them instead of once each.
        return {
            "image_dir": split_path,
            "annotation_dir": label_split,
            "classes_path": data_yaml,
            "images": images,
        }

    @classmethod
    @override
    def detect(cls, source: Path) -> Layout | None:
        if not source.is_dir():
            return None

        # Split roots live under images/<split> instead of <split>/.
        image_root = source / "images"
        discovered: dict[str | None, dict[str, Any]] = {}
        if image_root.is_dir():
            for split_name in cls.split_names:
                split_kwargs = cls.validate_split(image_root / split_name)
                if split_kwargs is None:
                    continue
                discovered[cls._canonicalize_split_name(split_name)] = (
                    split_kwargs
                )
        if discovered:
            return Layout(discovered)

        # One ``images/<split>`` directory parses on its own: its labels and
        # its ``data.yaml`` are resolved relative to its grandparent, which is
        # the dataset root either way.
        split_kwargs = cls.validate_split(source)
        if split_kwargs is None:
            return None
        return Layout({None: split_kwargs})

    @override
    def _split_files(
        self,
        image_dir: Path,
        annotation_dir: Path,
        classes_path: Path,
        images: list[Path] | None = None,
    ) -> list[Path]:
        """List the images of one split.

        Every listed image yields at least one record, so the listing is
        already the file list and no label file has to be read for it.

        Args:
            image_dir: Directory with images.
            annotation_dir: Directory with annotations.
            classes_path: YAML file with class names.
            images: Images of ``image_dir`` as already listed by
                `validate_split`. Listed here when not given.

        Returns:
            The images of the split.

        """
        del annotation_dir, classes_path
        # The copy keeps a caller trimming the enumerated files from reaching
        # into the listing the records are streamed from.
        return self._list_images(image_dir) if images is None else list(images)

    @override
    def _split_records(
        self,
        image_dir: Path,
        annotation_dir: Path,
        classes_path: Path,
        images: list[Path] | None = None,
    ) -> DatasetIterator:
        """Parse YOLOv6 annotations into LDF records.

        Annotations include classification and object detection.

        Args:
            image_dir: Directory with images.
            annotation_dir: Directory with annotations.
            classes_path: YAML file with class names.
            images: Images of ``image_dir`` as already listed by
                `validate_split`. Listed here when not given.

        Returns:
            One record per annotation, and one per unannotated image.

        """
        with open(classes_path, encoding="utf-8") as f:
            classes_data = cast(dict[str, Any], yaml.safe_load(f))
        # YOLO `data.yaml` files declare `names` either as a sequence or
        # as an index-to-name mapping; the YOLOv8 parser accepts both, so
        # the two must agree on the same file.
        names = classes_data["names"]
        class_names = (
            names if isinstance(names, dict) else dict(enumerate(names))
        )

        image_paths = (
            self._list_images(image_dir) if images is None else images
        )

        def generator() -> DatasetIterator:
            for img_path in image_paths:
                file = str(img_path)
                # A stem is the name minus its suffix, and `with_suffix`
                # appends when there is no suffix to replace, so appending to
                # the stem names the same label file in both cases.
                ann_path = annotation_dir / f"{img_path.stem}.txt"

                annotation_data = []
                if ann_path.exists():
                    with open(ann_path, encoding="utf-8") as f:
                        annotation_data = f.readlines()

                if not annotation_data:
                    yield {"file": file, "annotation": None}
                    continue

                for ann_line in annotation_data:
                    class_id, x_center, y_center, width, height = (
                        ann_line.split()
                    )
                    class_name = class_names[int(class_id)]
                    # Parsing a decimal literal is deterministic, so reusing
                    # the size for the origin gives the same floats as
                    # converting the same text twice.
                    w = float(width)
                    h = float(height)

                    yield {
                        "file": file,
                        "annotation": {
                            "class": class_name,
                            "boundingbox": {
                                "x": float(x_center) - w / 2,
                                "y": float(y_center) - h / 2,
                                "w": w,
                                "h": h,
                            },
                        },
                    }

        return generator()
