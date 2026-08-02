from collections import defaultdict
from os.path import islink, join
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from luxonis_ml.data import DatasetIterator
from luxonis_ml.utils.path import parse_manifest_path, resolve_manifest_path

from .parser_plugin import SplitParserPlugin


class TensorflowCSVParser(SplitParserPlugin):
    """Parse a directory with TensorFlow CSV annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   ├── ...
        │   └── _annotations.csv
        ├── valid/
        └── test/

    This is one of the formats that Roboflow can generate.
    """

    dataset_types = ("tfcsv",)

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        images = TensorflowCSVParser._list_images(split_path)
        if not images:
            return None
        annotation_path = split_path / "_annotations.csv"
        if not annotation_path.exists():
            return None
        # The listing is handed on, so the directory is walked once
        # between recognizing and parsing the split.
        return {
            "image_dir": split_path,
            "annotation_path": annotation_path,
            "images": images,
        }

    def _split_files(
        self,
        image_dir: Path,
        annotation_path: Path,
        images: list[Path],
    ) -> list[Path]:
        """List the images of one split.

        Every listed image yields at least one record naming it, so the
        listing already is the file list and the CSV is not read for it.

        Args:
            image_dir: Directory the images were listed from.
            annotation_path: Annotation CSV file.
            images: Images of the split, as listed by `validate_split`.

        Returns:
            The images of the split, in record order.

        """
        del annotation_path
        paths = self._resolve_images(image_dir, images)
        return [Path(path) for path in dict.fromkeys(paths)]

    def _split_records(
        self,
        image_dir: Path,
        annotation_path: Path,
        images: list[Path],
    ) -> DatasetIterator:
        """Parse TensorFlow CSV annotations into LDF records.

        Annotations include classification and object detection.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation CSV file.
            images: Images of the split, as listed by `validate_split`.

        Returns:
            The records of the split.

        """
        df = pl.read_csv(annotation_path).filter(
            pl.col("filename").is_not_null()
        )
        paths = self._resolve_images(image_dir, images)
        spellings = self._known_spellings(images, paths)
        # A CSV holds one row per bounding box, so each filename is
        # spelled once per box of its image. How a spelling resolves
        # depends only on the spelling and the split directory, so every
        # distinct spelling is resolved at most once.
        resolved: dict[str, str] = {}

        annotations: dict[str, list[tuple[Any, list[float]]]] = defaultdict(
            list
        )
        column = {name: index for index, name in enumerate(df.columns)}
        for row in df.iter_rows():
            filename = str(row[column["filename"]])
            path = spellings.get(filename) or resolved.get(filename)
            if path is None:
                # A spelling is read as a POSIX path first, so one that
                # reads as the name of a listed image - Windows-style or
                # not - resolves to that image.
                path = spellings.get(str(parse_manifest_path(filename)))
                if path is None:
                    path = str(resolve_manifest_path(image_dir, filename))
                resolved[filename] = path

            class_name = row[column["class"]]
            height = row[column["height"]]
            width = row[column["width"]]
            xmin = row[column["xmin"]]
            ymin = row[column["ymin"]]
            xmax = row[column["xmax"]]
            ymax = row[column["ymax"]]
            bbox_xywh = np.array(
                [xmin, ymin, xmax - xmin, ymax - ymin], dtype=float
            )
            bbox_xywh[::2] /= width
            bbox_xywh[1::2] /= height
            bbox_xywh = bbox_xywh.tolist()
            annotations[path].append((class_name, bbox_xywh))

        def generator() -> DatasetIterator:
            # Deduplicated exactly like `_split_files`: a symlink next to
            # its target must not repeat that image's boxes while the
            # enumeration reports the file once.
            for path in dict.fromkeys(paths):
                image_annotations = annotations.get(path)
                if not image_annotations:
                    yield {"file": path, "annotation": None}
                    continue
                for bbox_class, (x, y, w, h) in image_annotations:
                    yield {
                        "file": path,
                        "annotation": {
                            "class": bbox_class,
                            "boundingbox": {
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                            },
                        },
                    }

        return generator()

    @staticmethod
    def _resolve_images(image_dir: Path, images: list[Path]) -> list[str]:
        """Return the resolved path of every listed image.

        Args:
            image_dir: Directory the images were listed from.
            images: Listed images.

        Returns:
            One resolved path per image, in listing order.

        """
        resolved_dir = str(image_dir.resolve())
        paths = []
        for image in images:
            # The images all live in `image_dir`, whose resolved form has
            # no symlink left to follow, so only the file name itself can
            # still redirect. `os.path.islink` is what resolving uses: a
            # failed `lstat` counts as "not a link" instead of raising.
            if islink(image):  # noqa: PTH114
                paths.append(str(image.resolve()))
            else:
                paths.append(join(resolved_dir, image.name))  # noqa: PTH118
        return paths

    @staticmethod
    def _known_spellings(
        images: list[Path], paths: list[str]
    ) -> dict[str, str]:
        """Return the resolved path of each way of naming a listed image.

        Args:
            images: Listed images.
            paths: Their resolved paths.

        Returns:
            Mapping from a manifest spelling to the path it resolves to.

        """
        spellings: dict[str, str] = {}
        for image, path in zip(images, paths, strict=True):
            # A resolved path resolves to itself, and a bare file name of
            # the annotated directory resolves to the image of that name.
            spellings[path] = path
            # A backslash is a directory separator to a manifest, so a name
            # holding one is not spelled that way in the CSV.
            if "\\" not in image.name:
                spellings[image.name] = path
        return spellings
