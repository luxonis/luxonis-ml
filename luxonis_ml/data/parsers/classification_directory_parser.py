from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator

from .parser_plugin import ParsedDataset, SplitParserPlugin


class ClassificationDirectoryParser(SplitParserPlugin):
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
    """

    dataset_types = ("clsdir",)

    #: Directory names that belong to other layouts and are never classes.
    _RESERVED_DIR_NAMES = frozenset(
        {
            "train",
            "valid",
            "test",
            "val",
            "validation",
            "images",
            "labels",
            "data",
            "raw",
            "masks",
        }
    )

    @classmethod
    def _list_class_dirs(cls, split_path: Path) -> list[Path]:
        return [
            path
            for path in split_path.iterdir()
            if path.is_dir() and path.name not in cls._RESERVED_DIR_NAMES
        ]

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        classes = ClassificationDirectoryParser._list_class_dirs(split_path)
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

    def _parse_split(self, class_dir: Path) -> ParsedDataset:
        """Parse classification-directory annotations into LDF records.

        Annotations include classification labels.

        Args:
            class_dir: Top-level class directory.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        class_dirs = self._list_class_dirs(class_dir)

        def generator() -> DatasetIterator:
            for class_path in class_dirs:
                class_name = class_path.name
                for img_path in self._list_images(class_path):
                    yield {
                        "file": str(img_path.absolute().resolve()),
                        "annotation": {"class": class_name},
                    }

        added_images = self._get_added_images(generator())

        return ParsedDataset(generator(), {}, added_images)
