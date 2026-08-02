import json
from pathlib import Path
from typing import Any, cast

from loguru import logger
from typing_extensions import override

from luxonis_ml.data import DatasetIterator
from luxonis_ml.data.utils.enums import ParserIssue

from .parser_plugin import Layout, SplitParserPlugin


class FiftyOneClassificationParser(SplitParserPlugin):
    """Parse FiftyOne image classification data into LDF.

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

    This parser supports the FiftyOne image classification export layout.

    """

    dataset_types = ("fiftyone-classification",)
    split_names = ("train", "validation", "test")

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
            with open(labels_path, encoding="utf-8") as f:
                labels_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        # Checked by type rather than by key: `"classes" in labels_data`
        # raises on a `labels.json` holding a number, which would abort
        # the detection of every other format too.
        if not isinstance(labels_data, dict):
            return None
        classes = labels_data.get("classes")
        labels = labels_data.get("labels")
        if (
            not isinstance(classes, list)
            or not all(isinstance(name, str) for name in classes)
            or not isinstance(labels, dict)
        ):
            return None

        # `labels.json` is handed on rather than read a second time.
        return {"split_path": split_path, "labels_data": labels_data}

    @staticmethod
    def _names_a_class(class_idx: Any, classes: list[str]) -> bool:
        """Report whether a label selects a class that exists.

        Indexing alone does not answer this. A negative index picks a
        class from the end of the list, and ``True`` picks the second
        one, so both would quietly label an image with the wrong class
        where an index past the end at least raises. `bool` is rejected
        on purpose: it is an `int`, but a label of ``true`` is malformed
        rather than a request for class 1.
        """
        return type(class_idx) is int and 0 <= class_idx < len(classes)

    @classmethod
    @override
    def detect(cls, source: Path) -> Layout | None:
        layout = super().detect(source)
        if layout is None:
            return None
        # Which layout a directory belongs to is positional, not something
        # the directory itself shows: a `train` split and a flat source
        # sitting in a directory called `train` look alike, so only
        # detection can tell them apart.
        return Layout(
            {
                split_name: {**split_kwargs, "is_flat": split_name is None}
                for split_name, split_kwargs in layout.splits.items()
            }
        )

    @staticmethod
    def _read_labels(labels_path: Path) -> dict[str, Any]:
        with open(labels_path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))

    def _stem_to_path(self, split_path: Path) -> dict[str, Path]:
        """Map each image stem in a split's ``data`` directory to its path.

        Labels name an image by its stem, so the extension a split happens
        to use is resolved through a single listing of the directory.
        """
        return {
            image.stem: image
            for image in self._list_images(split_path / "data")
        }

    def _split_records(
        self,
        split_path: Path,
        labels_data: dict[str, Any] | None = None,
        is_flat: bool = True,
    ) -> DatasetIterator:
        """Stream the records of one FiftyOne split directory.

        Args:
            split_path: Directory holding ``data`` and ``labels.json``.
            labels_data: Content of ``labels.json`` as parsed by
                `validate_split`. Read from disk when not given.
            is_flat: Whether ``split_path`` is a whole flat source rather
                than one split of a split-based one, as `detect`
                determined. Only a flat source runs the ImageNet cleanup.

        Yields:
            One classification record per label naming an image that the
            split actually contains, in label order.

        """
        labels_path = split_path / "labels.json"

        if is_flat:
            cleaned_path = clean_imagenet_annotations(labels_path)
            if cleaned_path != labels_path:
                # The cleanup rewrote the labels, so what validation read
                # is no longer what this split is parsed from.
                labels_data = None
            labels_path = cleaned_path

        if labels_data is None:
            labels_data = self._read_labels(labels_path)

        classes = labels_data["classes"]
        labels = labels_data["labels"]
        stem_to_path = self._stem_to_path(split_path)

        for image_stem, class_idx in labels.items():
            image_path = stem_to_path.get(image_stem)
            if image_path is None:
                self._warn_skipped_annotation(
                    ParserIssue.MISSING_IMAGE_STEM,
                    "label references an image stem that is not present in the split",
                    source=labels_path,
                    image=image_stem,
                )
                continue

            # A label naming no class is an error the parser raises, not a
            # record it quietly drops - checked here, as the label is
            # reached, so the walk stays single-pass.
            if not self._names_a_class(class_idx, classes):
                raise IndexError(
                    f"Label for image '{image_stem}' in '{labels_path}' "
                    f"names class index {class_idx!r}, which the "
                    f"{len(classes)}-class list does not hold."
                )

            yield {
                "file": image_path,
                "annotation": {"class": classes[class_idx]},
            }

    def _split_files(
        self,
        split_path: Path,
        labels_data: dict[str, Any] | None = None,
        is_flat: bool = True,
    ) -> list[Path]:
        """List the images one FiftyOne split parses into records.

        Args:
            split_path: Directory holding ``data`` and ``labels.json``.
            labels_data: Content of ``labels.json`` as parsed by
                `validate_split`. Read from disk when not given.
            is_flat: Unused. Listing the files is the same either way.

        Returns:
            The split's images in label order, leaving out both the images
            no label names and the labels naming a missing image.

        """
        del is_flat
        if labels_data is None:
            labels_data = self._read_labels(split_path / "labels.json")

        # The ImageNet cleanup a flat layout runs before parsing only
        # renames classes and re-points two label indices, so the images
        # are the same either way and this need not run it - or rewrite
        # anything on disk.
        stem_to_path = self._stem_to_path(split_path)
        return [
            stem_to_path[image_stem]
            for image_stem in labels_data["labels"]
            if image_stem in stem_to_path
        ]


def clean_imagenet_annotations(labels_path: Path) -> Path:
    """Clean known ImageNet issues in FiftyOne labels.

    The cleanup fixes duplicate class names and known label-index errors in
    ImageNet FiftyOne exports.

    Args:
        labels_path: Path to ``labels.json``.

    Returns:
        Path to the cleaned labels file.

    """
    with open(labels_path, encoding="utf-8") as f:
        labels_data = json.load(f)

    classes = labels_data["classes"]
    labels = labels_data["labels"]

    modified = False

    # Fix duplicate class names
    # First "crane" (bird) should be renamed to "crane bird"
    crane_indices = [i for i, c in enumerate(classes) if c == "crane"]
    if len(crane_indices) >= 1:
        first_crane_idx = crane_indices[0]
        classes[first_crane_idx] = "crane bird"
        logger.info(
            f"Renamed class 'crane' at index {first_crane_idx} to 'crane bird'"
        )
        modified = True

    # Second "maillot" should be renamed to "maillot swim suit"
    maillot_indices = [i for i, c in enumerate(classes) if c == "maillot"]
    if len(maillot_indices) >= 2:
        second_maillot_idx = maillot_indices[1]
        classes[second_maillot_idx] = "maillot swim suit"
        logger.info(
            f"Renamed class 'maillot' at index {second_maillot_idx} "
            "to 'maillot swim suit'"
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
    with open(cleaned_labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_data, f)

    logger.info(f"Cleaned annotations saved to {cleaned_labels_path}")
    return cleaned_labels_path
