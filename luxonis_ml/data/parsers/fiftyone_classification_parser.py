import json
from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class FiftyOneClassificationParser(BaseParser):
    """Parses FiftyOneImageClassificationDataset format to LDF.

    Supports two directory structures:

    Split structure with train/test/validation subdirectories::

        dataset_dir/
        ├── train/
        │   ├── data/
        │   │   ├── img1.jpg
        │   │   └── ...
        │   └── labels.json
        ├── validation/
        │   ├── data/
        │   └── labels.json
        └── test/
            ├── data/
            └── labels.json

    Flat structure (single directory, random splits applied at parse time)::

        dataset_dir/
        ├── data/
        │   ├── img1.jpg
        │   └── ...
        └── labels.json

    The labels.json format is::

        {
            "classes": ["class1", "class2", ...],
            "labels": {
                "image_stem": class_index,
                ...
            }
        }

    U{FiftyOneImageClassificationDataset <https://docs.voxel51.com/user_guide/export_datasets.html#fiftyone-image-classification-dataset>}.
    """

    SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None

        labels_path = split_path / "labels.json"
        data_path = split_path / "data"

        if not labels_path.exists() or not data_path.exists():
            return None

        if not data_path.is_dir():
            return None

        try:
            with open(labels_path) as f:
                labels_data = json.load(f)
            if "classes" not in labels_data or "labels" not in labels_data:
                return None
        except (json.JSONDecodeError, OSError):
            return None

        return {"split_path": split_path}

    def from_dir(
        self, dataset_dir: Path, **kwargs
    ) -> tuple[list[Path], list[Path], list[Path]]:
        added_train_imgs: list[Path] = []
        added_val_imgs: list[Path] = []
        added_test_imgs: list[Path] = []

        if (dataset_dir / "train").exists():
            added_train_imgs = self._parse_split(
                split_path=dataset_dir / "train"
            )

        if (dataset_dir / "validation").exists():
            added_val_imgs = self._parse_split(
                split_path=dataset_dir / "validation"
            )

        if (dataset_dir / "test").exists():
            added_test_imgs = self._parse_split(
                split_path=dataset_dir / "test"
            )

        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(self, split_path: Path) -> ParserOutput:
        labels_path = split_path / "labels.json"
        data_path = split_path / "data"

        with open(labels_path) as f:
            labels_data = json.load(f)

        classes = labels_data["classes"]
        labels = labels_data["labels"]

        images = self._list_images(data_path)
        stem_to_path = {img.stem: img for img in images}

        def generator() -> DatasetIterator:
            for image_stem, class_idx in labels.items():
                if image_stem not in stem_to_path:
                    continue

                img_path = stem_to_path[image_stem]
                class_name = classes[class_idx]

                yield {
                    "file": img_path,
                    "annotation": {"class": class_name},
                }

        added_images = self._get_added_images(generator())

        return generator(), {}, added_images
