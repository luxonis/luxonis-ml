import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from luxonis_ml.data import DatasetIterator

from .parser_plugin import SplitParserPlugin

# A stem holding one of these is a pattern rather than a literal name
# prefix, so it cannot be looked up in a directory listing.
_GLOB_MAGIC = re.compile(r"[*?\[]")


class SegmentationMaskDirectoryParser(SplitParserPlugin):
    """Parse a directory with segmentation mask annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img1_mask.png
        │   ├── ...
        │   └── _classes.csv
        ├── valid/
        └── test/

    ``_classes.csv`` maps pixel values to class names.

    This is one of the formats that Roboflow can generate.
    """

    dataset_types = ("segmask",)

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        if not (split_path / "_classes.csv").exists():
            return None
        # One listing answers "is the image there?" for every mask, through
        # the same prefix rule the pairing uses - the records accept any
        # image suffix, so validation must not demand `.jpg`. A listed name
        # exists unless it is a dangling symlink.
        entries, by_prefix = SegmentationMaskDirectoryParser._index_directory(
            split_path
        )
        for mask_path in split_path.glob("*_mask.*"):
            stem = mask_path.stem[:-5]
            if _GLOB_MAGIC.search(stem):
                # A stem that is a pattern must keep matching as one,
                # exactly as the pairing resolves it.
                if next(split_path.glob(f"{stem}.*"), None) is None:
                    return None
                continue
            image = entries.get(by_prefix.get(stem, ""))
            if image is None or (
                image.is_symlink() and not Path(image.path).exists()
            ):
                return None
        return {
            "image_dir": split_path,
            "seg_dir": split_path,
            "classes_path": split_path / "_classes.csv",
        }

    @staticmethod
    def _index_directory(
        directory: Path,
    ) -> tuple[dict[str, os.DirEntry[str]], dict[str, str]]:
        """List ``directory`` once, indexing names by their dot prefixes.

        A name matches the ``{stem}.*`` glob exactly when it starts with
        ``{stem}.``, so keeping the first listed name - the one ``next``
        would have taken - under each of its dot-delimited prefixes
        answers every such lookup from a single listing.
        """
        entries: dict[str, os.DirEntry[str]] = {}
        by_prefix: dict[str, str] = {}
        try:
            with os.scandir(directory) as listing:
                for entry in listing:
                    name = entry.name
                    entries[name] = entry
                    dot = name.find(".")
                    while dot >= 0:
                        by_prefix.setdefault(name[:dot], name)
                        dot = name.find(".", dot + 1)
        except OSError:
            # An unlistable directory leaves every lookup to `glob`.
            pass
        return entries, by_prefix

    def _split_files(
        self, image_dir: Path, seg_dir: Path, classes_path: Path
    ) -> list[Path]:
        """List the images of one split without decoding any mask.

        Args:
            image_dir: Directory with images.
            seg_dir: Directory with segmentation masks.
            classes_path: CSV file with class names.

        Returns:
            The images the split's masks annotate, in the order the masks
            are listed in and without duplicates.

        Raises:
            RuntimeError: If a mask has no matching image.

        """
        del classes_path
        # The files of a split are the images its masks name, which the
        # pairing already answers.
        return list(
            dict.fromkeys(
                image_path
                for _, image_path in self._pair_masks_with_images(
                    image_dir, seg_dir
                )
            )
        )

    def _split_records(
        self, image_dir: Path, seg_dir: Path, classes_path: Path
    ) -> DatasetIterator:
        """Stream segmentation mask annotations as LDF records.

        Annotations include classification and segmentation.

        Args:
            image_dir: Directory with images.
            seg_dir: Directory with segmentation masks.
            classes_path: CSV file with class names.

        Returns:
            Iterator over the records of the split.

        Raises:
            RuntimeError: If a mask has no matching image.
            ValueError: If a mask cannot be read.

        """
        # NOTE: space prefix included
        idx_class = " Class"

        df = pl.read_csv(classes_path).filter(pl.col(idx_class).is_not_null())
        class_names = df[idx_class].to_list()

        # Paired up front, so that a mask naming no image fails the parse
        # rather than a half-finished import. Decoding the masks - the
        # expensive part - stays in the generator.
        pairs = list(self._pair_masks_with_images(image_dir, seg_dir))
        for mask_path, _ in pairs:
            # An undecodable mask has to fail up front too, and checking
            # the format signature costs no decode.
            if not cv2.haveImageReader(mask_path):
                raise ValueError(f"Failed to read mask image: {mask_path}")

        def generator() -> DatasetIterator:
            for mask_path, image_path in pairs:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to read mask image: {mask_path}")

                file = str(image_path)
                # `IMREAD_GRAYSCALE` always decodes to `uint8`, so the
                # pixel values present are the non-empty bins of a 256-bin
                # count, in ascending order.
                class_ids = np.flatnonzero(np.bincount(mask.ravel())).tolist()
                for class_id in class_ids:
                    class_name = class_names[class_id]

                    curr_seg_mask = (mask == class_id).astype(np.uint8)
                    yield {
                        "file": file,
                        "annotation": {
                            "class": class_name,
                            "segmentation": {"mask": curr_seg_mask},
                        },
                    }

        return generator()

    @staticmethod
    def _pair_masks_with_images(
        image_dir: Path, seg_dir: Path
    ) -> Iterator[tuple[str, Path]]:
        """Pair every mask with the image it annotates.

        Args:
            image_dir: Directory with images.
            seg_dir: Directory with segmentation masks.

        Yields:
            The mask to decode and the resolved path of its image, in
            the order the masks are listed in.

        Raises:
            RuntimeError: If a mask has no matching image. The exhausted
                `glob` below raises `StopIteration`, which a generator
                turns into this.

        """
        # Globbing `{stem}.*` would walk `image_dir` once per mask, so the
        # lookups go through the prefix index instead - the same one
        # `validate_split` checks, which keeps the two in agreement about
        # which masks have an image.
        entries, by_prefix = SegmentationMaskDirectoryParser._index_directory(
            image_dir
        )
        links = {name for name, entry in entries.items() if entry.is_symlink()}

        # For a name that is not itself a symlink, the resolved directory
        # joined with it is what resolving the entry would give.
        resolved_dir = image_dir.absolute().resolve()

        for mask_path in seg_dir.glob("*_mask.*"):
            stem = mask_path.stem[:-5]
            name = None if _GLOB_MAGIC.search(stem) else by_prefix.get(stem)
            if name is None:
                # A stem that is a pattern must keep matching as one,
                # and a stem that matched nothing must still raise.
                image_path = next(image_dir.glob(f"{stem}.*"))
                yield str(mask_path), image_path.absolute().resolve()
            elif name in links:
                yield str(mask_path), (image_dir / name).absolute().resolve()
            else:
                yield str(mask_path), resolved_dir / name
