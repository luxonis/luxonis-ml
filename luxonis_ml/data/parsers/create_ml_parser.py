import json
from pathlib import Path
from typing import Any

from PIL import Image

from luxonis_ml.data import DatasetIterator
from luxonis_ml.data.utils.enums import ParserIssue
from luxonis_ml.utils.path import resolve_manifest_path

from .parser_plugin import SplitParserPlugin


class CreateMLParser(SplitParserPlugin):
    """Parse a directory with CreateML annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        │   └── _annotations.createml.json
        ├── valid/
        └── test/

    This is one of the formats that Roboflow can generate.
    """

    dataset_types = ("createml",)

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        if not (split_path / "_annotations.createml.json").exists():
            return None
        if not CreateMLParser._list_images(split_path):
            return None
        return {
            "image_dir": split_path,
            "annotation_path": split_path / "_annotations.createml.json",
        }

    @staticmethod
    def _resolve_image(base_dir: Path, reference: Any) -> Path:
        """Resolve a manifest image reference against its split
        directory.

        Args:
            base_dir: Already absolute and resolved directory the
                manifest lives in.
            reference: The ``image`` value of a manifest entry.

        Returns:
            Absolute, symlink-resolved path to the referenced image.

        """
        # A reference holding neither separator is a bare file name on any
        # platform, which `resolve_manifest_path` would only join onto
        # `base_dir` anyway.
        if (
            isinstance(reference, str)
            and "/" not in reference
            and "\\" not in reference
        ):
            return (base_dir / reference).resolve()
        return resolve_manifest_path(base_dir, reference)

    def _split_records(
        self, image_dir: Path, annotation_path: Path
    ) -> DatasetIterator:
        """Stream CreateML annotations of one split as LDF records.

        Annotations include classification and object detection.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation JSON file.

        Yields:
            One record per box, in manifest order. The manifest is walked
            a single time and each record is emitted as it is read.

        """
        with open(annotation_path, encoding="utf-8") as f:
            annotations_data = json.load(f)

        # The same for every entry, so resolved once.
        base_dir = image_dir.absolute().resolve()

        for annotations in annotations_data:
            path = self._resolve_image(base_dir, annotations["image"])
            if not path.exists():
                self._warn_skipped_annotation(
                    ParserIssue.MISSING_IMAGE,
                    "referenced image file does not exist",
                    source=annotation_path,
                    image=path,
                )
                continue
            file = str(path)
            # Ahead of the boxes, so that an unreadable image fails the
            # parse even for a frame that carries none. Only the header is
            # needed, so the file is closed right away.
            with Image.open(file) as img:
                width, height = img.size

            # A frame without boxes yields nothing at all.
            for curr_ann in annotations["annotations"]:
                bbox_ann = curr_ann["coordinates"]
                yield {
                    "file": file,
                    "annotation": {
                        "class": curr_ann["label"],
                        "boundingbox": {
                            "x": (bbox_ann["x"] - bbox_ann["width"] / 2)
                            / width,
                            "y": (bbox_ann["y"] - bbox_ann["height"] / 2)
                            / height,
                            "w": bbox_ann["width"] / width,
                            "h": bbox_ann["height"] / height,
                        },
                    },
                }

    def _split_files(
        self, image_dir: Path, annotation_path: Path
    ) -> list[Path]:
        """List the images of one split straight from its manifest.

        The manifest names every image that can produce a record, so a
        count-based import picks its subset without decoding one.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation JSON file.

        Returns:
            The images yielding at least one record, in manifest order and
            deduplicated, mirroring a manifest that names one image twice.

        """
        with open(annotation_path, encoding="utf-8") as f:
            annotations_data = json.load(f)

        base_dir = image_dir.absolute().resolve()

        files: dict[Path, None] = {}
        for annotations in annotations_data:
            # A frame without boxes produces no record, so it is not a
            # file an import can choose.
            if not annotations["annotations"]:
                continue
            path = self._resolve_image(base_dir, annotations["image"])
            # The parse reports a missing image; this only decides what
            # can be selected.
            if path.exists():
                files[path] = None
        return list(files)
