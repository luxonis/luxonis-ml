"""Parser for classification datasets with one directory per class."""

import filecmp
from collections import defaultdict
from pathlib import Path
from typing import Any

from typing_extensions import override

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class ClassificationDirectoryParser(BaseParser):
    """Parse a directory with classification annotations into LDF.

    Supports two directory structures:

    Split structure with train/valid/test subdirectories::

        dataset_dir/
        ├── train/
        │   ├── class1/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   ├── class2/
        │   └── ...
        ├── valid/
        └── test/

    Flat structure (class subdirectories directly in root,
    random splits applied at parse time)::

        dataset_dir/
        ├── class1/
        │   ├── img1.jpg
        │   └── ...
        ├── class2/
        │   └── ...
        └── info.json  (optional metadata file)

    The split structure is one of the formats that Roboflow can generate.

    An image with more than one class has a copy in each of its class
    directories. The copies must have the same file name and the same
    content, as `ClassificationDirectoryExporter` writes them. The parser
    adds such copies as one image with all of their classes. Files with the
    same name and different content stay separate images.
    """

    @staticmethod
    @override
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        classes = [
            d
            for d in split_path.iterdir()
            if d.is_dir()
            and d.name
            not in {
                "train",
                "valid",
                "test",
                "val",
                "validation",
                "images",
                "labels",
            }
        ]
        if not classes:
            return None
        # For now allow info.json, can be extended to other metadata files
        fnames = [
            f
            for f in split_path.iterdir()
            if f.is_file() and f.name != "info.json"
        ]
        if fnames:
            return None
        return {"class_dir": split_path}

    @override
    def from_dir(
        self, dataset_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
        added_train_imgs = self._parse_split(class_dir=dataset_dir / "train")
        added_val_imgs = self._parse_split(class_dir=dataset_dir / "valid")
        added_test_imgs = self._parse_split(class_dir=dataset_dir / "test")
        return added_train_imgs, added_val_imgs, added_test_imgs

    @override
    def from_split(self, class_dir: Path) -> ParserOutput:
        """Parse classification-directory annotations into LDF records.

        Annotations include classification labels.

        Args:
            class_dir: Top-level class directory.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        image_classes = self._image_classes(class_dir)

        def generator() -> DatasetIterator:
            for img_path, class_names in image_classes.items():
                for class_name in class_names:
                    yield {
                        "file": str(img_path),
                        "annotation": {"class": class_name},
                    }

        added_images = self._get_added_images(generator())

        return generator(), {}, added_images

    def _image_classes(self, class_dir: Path) -> dict[Path, list[str]]:
        """Map each image of a split to its class directory names."""
        image_classes: dict[Path, list[str]] = {}
        images_by_name: defaultdict[str, list[Path]] = defaultdict(list)
        for class_path in class_dir.iterdir():
            if not class_path.is_dir():
                continue
            for img_path in self._list_images(class_path):
                path = img_path.absolute().resolve()
                same_name = images_by_name[img_path.name]
                image = next(
                    (
                        p
                        for p in same_name
                        if filecmp.cmp(p, path, shallow=False)
                    ),
                    None,
                )
                if image is None:
                    same_name.append(path)
                    image = path
                image_classes.setdefault(image, []).append(class_path.name)
        return image_classes
