import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from luxonis_ml.data import DatasetIterator

from .parser_plugin import SplitParserPlugin


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

    @staticmethod
    def _resolve_listed(
        directory: Path, entries: list[Path]
    ) -> tuple[Path, list[str]]:
        """Resolve entries listed in one directory against its real path.

        Args:
            directory: Directory ``entries`` were listed from.
            entries: Paths of the form ``directory / name``.

        Returns:
            The resolved directory and, for every entry in order, the tail
            to join onto it: the entry name, or the entry's own resolved
            path when the entry is a symlink. Joining stays correct for
            both, because `Path.__truediv__` and `os.path.join` alike drop
            the directory when the tail is absolute.

        """
        # A resolved directory has no symlinked component left, so joining
        # it with the name of an entry that is not itself a symlink gives
        # what resolving that entry would - without walking the shared
        # parent components once per image.
        resolved_dir = directory.absolute().resolve()
        with os.scandir(directory) as scan:
            symlinks = {entry.name for entry in scan if entry.is_symlink()}
        return resolved_dir, [
            str(entry.absolute().resolve())
            if entry.name in symlinks
            else entry.name
            for entry in entries
        ]

    def _walk_classes(
        self, class_dir: Path
    ) -> Iterator[tuple[str, Path, list[str]]]:
        """Walk the class directories of one split, listing each once.

        Args:
            class_dir: Top-level class directory.

        Yields:
            The class name, the resolved directory holding its images, and
            the tail to join onto that directory for every image in it.

        """
        for class_path in class_dir.iterdir():
            if not class_path.is_dir():
                continue
            # An empty class directory contributes neither a record nor a
            # file.
            images = self._list_images(class_path)
            if not images:
                continue
            if class_path.name in self._RESERVED_DIR_NAMES:
                # The directory holds images but its name belongs to
                # another layout. It may still be a class legitimately
                # called `train`, so say so rather than dropping it
                # silently.
                logger.warning(
                    f"Not importing '{class_path}' as a class: its name "
                    "belongs to another dataset layout. Rename the "
                    "directory if it really is a class."
                )
                continue
            resolved_dir, tails = self._resolve_listed(class_path, images)
            yield class_path.name, resolved_dir, tails

    def _split_records(self, class_dir: Path) -> DatasetIterator:
        """Parse classification-directory annotations into LDF records.

        Annotations include classification labels.

        Args:
            class_dir: Top-level class directory.

        Yields:
            One classification record per image, class directory by class
            directory. Nothing is listed before the first record is asked
            for, so a source too large to hold in memory streams.

        """
        for class_name, resolved_dir, tails in self._walk_classes(class_dir):
            dir_path = str(resolved_dir)
            for tail in tails:
                # Joined as strings: a record only needs the string.
                yield {
                    "file": os.path.join(dir_path, tail),  # noqa: PTH118
                    "annotation": {"class": class_name},
                }

    def _split_files(self, class_dir: Path) -> list[Path]:
        """List the images of one split without building its records.

        The class directories are the annotation, so listing them is all a
        parse does anyway.

        Args:
            class_dir: Top-level class directory.

        Returns:
            The images of the split, deduplicated, in listing order. Two
            entries resolving to the same image - a file and a symlink to
            it in one class directory - name a single file.

        """
        return list(
            dict.fromkeys(
                resolved_dir / tail
                for _, resolved_dir, tails in self._walk_classes(class_dir)
                for tail in tails
            )
        )
