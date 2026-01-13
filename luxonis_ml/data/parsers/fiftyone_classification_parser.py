import json
from pathlib import Path
from typing import Any

from loguru import logger

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

        train_labels_path = dataset_dir / "train" / "labels.json"
        if train_labels_path.exists():
            native_classes = self._extract_native_classes(train_labels_path)
            if native_classes:
                self.dataset.set_native_classes(native_classes, "imagenet")

        return added_train_imgs, added_val_imgs, added_test_imgs

    @staticmethod
    def _extract_native_classes(labels_path: Path) -> dict[int, str]:
        with open(labels_path) as f:
            labels_data = json.load(f)

        classes = labels_data.get("classes", [])
        return dict(enumerate(classes))

    def from_split(
        self, split_path: Path, skip_clean: bool = False
    ) -> ParserOutput:
        labels_path = split_path / "labels.json"
        data_path = split_path / "data"

        # For flat structure (not a standard split directory), clean
        # ImageNet annotations to fix known issues with class names
        # and label indices, and set native classes
        is_flat_structure = split_path.name not in self.SPLIT_NAMES
        if is_flat_structure:
            if not skip_clean:
                labels_path = clean_imagenet_annotations(labels_path)
            native_classes = self._extract_native_classes(labels_path)
            if native_classes:
                self.dataset.set_native_classes(native_classes, "imagenet")

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


def clean_imagenet_annotations(labels_path: Path) -> Path:
    """Cleans ImageNet annotations by fixing known issues with class
    names and label indices.

    This function handles two known issues in ImageNet FiftyOne exports:

        1. Duplicate class names: First instance of "crane" is renamed
           to "crane_bird", second instance of "maillot" is renamed to
           "maillot_swim_suit".

        2. Misindexed labels: "006742" label 517 is corrected to 134,
           "031933" label 639 is corrected to 638.

    @type labels_path: Path
    @param labels_path: Path to the labels.json file.
    @rtype: Path
    @return: Path to the cleaned labels file.
    """
    with open(labels_path) as f:
        labels_data = json.load(f)

    classes = labels_data["classes"]
    labels = labels_data["labels"]

    modified = False

    # Fix duplicate class names
    # First "crane" (bird) should be renamed to "crane_bird"
    crane_indices = [i for i, c in enumerate(classes) if c == "crane"]
    if len(crane_indices) >= 1:
        first_crane_idx = crane_indices[0]
        classes[first_crane_idx] = "crane_bird"
        logger.info(
            f"Renamed class 'crane' at index {first_crane_idx} to 'crane_bird'"
        )
        modified = True

    # Second "maillot" should be renamed to "maillot_swim_suit"
    maillot_indices = [i for i, c in enumerate(classes) if c == "maillot"]
    if len(maillot_indices) >= 2:
        second_maillot_idx = maillot_indices[1]
        classes[second_maillot_idx] = "maillot_swim_suit"
        logger.info(
            f"Renamed class 'maillot' at index {second_maillot_idx} "
            "to 'maillot_swim_suit'"
        )
        modified = True

    # Fix misindexed labels
    # Image 006742 should map to index 134, not 517
    if labels.get("006742") == 517:
        labels["006742"] = 134
        logger.info("Fixed label index for image '006742': 517 -> 134")
        modified = True

    # Image 031933 should map to index 638, not 639
    if labels.get("031933") == 639:
        labels["031933"] = 638
        logger.info("Fixed label index for image '031933': 639 -> 638")
        modified = True

    if not modified:
        return labels_path

    labels_data["classes"] = classes
    labels_data["labels"] = labels

    cleaned_labels_path = labels_path.with_name("labels_fixed.json")
    with open(cleaned_labels_path, "w") as f:
        json.dump(labels_data, f)

    logger.info(f"Cleaned annotations saved to {cleaned_labels_path}")
    return cleaned_labels_path
