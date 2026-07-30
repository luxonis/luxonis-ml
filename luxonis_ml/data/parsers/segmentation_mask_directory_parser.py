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
        # One listing answers "is the image there?" for every mask at
        # once. A listed name always exists unless it is a dangling
        # symlink, the only case `Path.exists` still disagrees about.
        entries: dict[str, os.DirEntry[str]] = {}
        try:
            with os.scandir(split_path) as listing:
                entries = {entry.name: entry for entry in listing}
        except OSError:
            # A directory that cannot be listed has no masks to check
            # either: the glob below is driven by the same listing.
            pass
        for mask_path in split_path.glob("*_mask.*"):
            image = entries.get(f"{mask_path.stem[:-5]}.jpg")
            if image is None or (
                image.is_symlink() and not Path(image.path).exists()
            ):
                return None
        return {
            "image_dir": split_path,
            "seg_dir": split_path,
            "classes_path": split_path / "_classes.csv",
        }

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
        # pairing already answers - the class table and the masks
        # themselves say nothing about which files are there.
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

        # Every mask is paired with its image before a single record is
        # emitted, so a mask that names no image fails the parse itself
        # rather than half-way through an import. Decoding a mask - the
        # expensive part - stays in the generator, so a mask is read
        # once, when its records are pulled.
        pairs = list(self._pair_masks_with_images(image_dir, seg_dir))
        for mask_path, _ in pairs:
            # An undecodable mask has to fail up front for the same
            # reason. Checking the format signature keeps that failure
            # eager without paying for a decode the generator repeats.
            if not cv2.haveImageReader(mask_path):
                raise ValueError(f"Failed to read mask image: {mask_path}")

        def generator() -> DatasetIterator:
            for mask_path, image_path in pairs:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to read mask image: {mask_path}")

                file = str(image_path)
                # `IMREAD_GRAYSCALE` always decodes to `uint8`, so the
                # pixel values present are the non-empty bins of a
                # 256-bin count: the same ascending values sorting the
                # whole mask would report.
                ids = np.flatnonzero(np.bincount(mask.ravel())).tolist()
                for id in ids:
                    class_name = class_names[id]

                    # A cast comparison is the same 0/1 `uint8` array as
                    # a zero-filled one indexed by that comparison, with
                    # one full-size buffer written instead of two.
                    curr_seg_mask = (mask == id).astype(np.uint8)
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
            RuntimeError: If a mask has no matching image. A generator
                is what turns the exhausted lookup below into one, so
                this stays a generator function even though its result
                is consumed at once.

        """
        # Globbing `{stem}.*` per mask walks `image_dir` once per mask.
        # A name matches that pattern exactly when it starts with
        # `{stem}.`, so indexing every name under each of its
        # dot-delimited prefixes - keeping the first name listed, the
        # one `next` would have taken - answers every lookup from a
        # single listing.
        by_prefix: dict[str, str] = {}
        links: set[str] = set()
        try:
            with os.scandir(image_dir) as listing:
                for entry in listing:
                    name = entry.name
                    if entry.is_symlink():
                        links.add(name)
                    dot = name.find(".")
                    while dot >= 0:
                        by_prefix.setdefault(name[:dot], name)
                        dot = name.find(".", dot + 1)
        except OSError:
            # An index that could not be built leaves every lookup to
            # `glob`, which then fails - or stays silent about an
            # unreadable directory - exactly as it did before.
            pass

        # `realpath` resolves a path prefix first, so for a name that is
        # not itself a symlink the result is the resolved directory
        # joined with the name - no need to walk the shared prefix again
        # for every image.
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
