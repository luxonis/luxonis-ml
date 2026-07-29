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

    @staticmethod
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
                "data",
                "raw",
                "masks",
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

    def _parse_split(self, class_dir: Path) -> ParsedDataset:
        """Parse classification-directory annotations into LDF records.

        Annotations include classification labels.

        Args:
            class_dir: Top-level class directory.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        class_names = [d.name for d in class_dir.iterdir() if d.is_dir()]

        def generator() -> DatasetIterator:
            for class_name in class_names:
                for img_path in self._list_images(class_dir / class_name):
                    yield {
                        "file": str(img_path.absolute().resolve()),
                        "annotation": {"class": class_name},
                    }

        added_images = self._get_added_images(generator())

        return ParsedDataset(generator(), {}, added_images)
