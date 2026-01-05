import json
from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class FiftyOneClassificationParser(BaseParser):
    """Parses directory with FiftyOne Classification annotations to LDF.

    This parser supports only flat directory structures (no train/valid/test
    splits). The FiftyOne Classification format is inherently flat, and splits
    are handled internally by LDF via random splitting.

    Expected format::

        dataset_dir/
        ├── data/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── labels.json
        └── info.json (optional)

    The C{labels.json} file should have the following structure::

        {
            "classes": ["class1", "class2", ...],
            "labels": {
                "img1": 0,
                "img2": 1,
                ...
            }
        }

    Where each key in C{labels} is the image filename (without extension)
    and the value is the index into the C{classes} list.

    @note: This format does not support pre-defined train/valid/test splits.
        When parsing, all data is treated as a single split and LDF's
        random splitting is used to create train/val/test splits.
    """

    @classmethod
    def validate(cls, dataset_dir: Path) -> bool:
        """FiftyOne Classification format is flat-only, so this always
        returns False to ensure from_dir is never called via auto-
        detection.

        @type dataset_dir: Path
        @param dataset_dir: Path to source dataset directory.
        @rtype: bool
        @return: Always False for this flat-only format.
        """
        return False

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        """Validates if the directory is a valid FiftyOne Classification
        dataset (flat structure with data/ folder and labels.json).

        @type split_path: Path
        @param split_path: Path to dataset directory.
        @rtype: Optional[Dict[str, Any]]
        @return: Dictionary with kwargs for from_split if valid, None
            otherwise.
        """
        if not split_path.exists():
            return None

        labels_file = split_path / "labels.json"
        data_dir = split_path / "data"

        if not labels_file.exists() or not data_dir.exists():
            return None

        if not data_dir.is_dir():
            return None

        try:
            with open(labels_file) as f:
                labels_data = json.load(f)
            if "classes" not in labels_data or "labels" not in labels_data:
                return None
        except (json.JSONDecodeError, OSError):
            return None

        return {"data_dir": data_dir, "labels_file": labels_file}

    def from_dir(
        self, dataset_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Not supported for FiftyOne Classification format.

        FiftyOne Classification is a flat format without
        train/valid/test splits. Use parse_split() instead, which will
        apply random splitting.

        @raises NotImplementedError: Always, as this format is flat-
            only.
        """
        raise NotImplementedError(
            "FiftyOne Classification format does not support split directories. "
            "Use parse_split() with random_split=True instead."
        )

    def from_split(self, data_dir: Path, labels_file: Path) -> ParserOutput:
        """Parses annotations from FiftyOne classification format to
        LDF. Annotations include classification.

        @type data_dir: Path
        @param data_dir: Path to directory containing images
        @type labels_file: Path
        @param labels_file: Path to labels.json file
        @rtype: L{ParserOutput}
        @return: Annotation generator, skeleton dictionary for keypoints
            and list of added images.
        """
        with open(labels_file) as f:
            labels_data = json.load(f)

        classes = labels_data["classes"]
        labels = labels_data["labels"]

        def generator() -> DatasetIterator:
            for img_path in self._list_images(data_dir):
                img_stem = img_path.stem
                if img_stem in labels:
                    class_idx = labels[img_stem]
                    class_name = classes[class_idx]
                    yield {
                        "file": str(img_path.absolute().resolve()),
                        "annotation": {"class": class_name},
                    }

        added_images = self._get_added_images(generator())

        return generator(), {}, added_images
