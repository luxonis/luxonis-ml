from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from luxonis_ml.data import DatasetIterator
from luxonis_ml.data.utils.enums import ParserIssue
from luxonis_ml.utils.path import resolve_manifest_path

from .parser_plugin import SplitParserPlugin


class YoloV4Parser(SplitParserPlugin):
    """Parse a directory with YOLOv4 annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   ├── ...
        │   ├── _annotations.txt
        │   └── _classes.txt
        ├── valid/
        └── test/

    This is one of the formats that Roboflow can generate.
    """

    dataset_types = ("yolov4",)

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        annotations = split_path / "_annotations.txt"
        classes = split_path / "_classes.txt"
        if not annotations.exists() or not classes.exists():
            return None
        return {
            "image_dir": split_path,
            "annotation_path": annotations,
            "classes_path": classes,
        }

    def _annotation_lines(
        self,
        base_dir: Path,
        annotation_path: Path,
        annotated: set[Path],
    ) -> Iterator[tuple[Path, str]]:
        """Walk the annotation file, resolving the image of every line.

        The annotations themselves are never held in memory: each line is
        handed to the caller as it is read.

        Args:
            base_dir: Resolved directory relative image paths are read
                against.
            annotation_path: Annotation file.
            annotated: Filled with the resolved image of every line, the
                lines naming a missing image included, so that a caller
                can tell which images of the directory are unlisted.

        Yields:
            The resolved image of each line whose image exists, together
            with the rest of the line. A line naming a missing image is
            reported as a skipped annotation instead.

        """
        with open(annotation_path, encoding="utf-8") as f:
            for ann_line in f:
                img_path, _, boxes = ann_line.rstrip().partition(" ")
                path = resolve_manifest_path(base_dir, img_path)
                # Every path `resolve_manifest_path` returns as absolute has
                # already been resolved, and resolving is idempotent, so only
                # its relative fallback for Windows paths needs it again.
                annotated.add(path if path.is_absolute() else path.resolve())
                if not path.exists():
                    self._warn_skipped_annotation(
                        ParserIssue.MISSING_IMAGE,
                        "referenced image file does not exist",
                        source=annotation_path,
                        image=path,
                    )
                    continue

                yield path, boxes

    def _unlisted_images(
        self, image_dir: Path, base_dir: Path, annotated: set[Path]
    ) -> Iterator[Path]:
        """Yield the images of the directory no annotation line names.

        Args:
            image_dir: Directory with images.
            base_dir: Resolved ``image_dir``.
            annotated: Resolved image of every annotation line, as filled
                by `_annotation_lines`.

        Yields:
            Images present in the directory but absent from the
            annotation file.

        """
        # A resolved path never ends in a symlink, so an image of the
        # directory whose name is claimed by a resolved direct child of that
        # same directory is that child, and resolving it again cannot tell
        # the two apart. Names of annotations pointing elsewhere - a nested
        # directory, another disk - are deliberately not collected, so those
        # images still take the resolving path below.
        annotated_names = {
            path.name for path in annotated if path.parent == base_dir
        }

        for img_path in self._list_images(image_dir):
            if (
                img_path.name not in annotated_names
                and img_path.resolve() not in annotated
            ):
                yield img_path

    def _split_records(
        self,
        image_dir: Path,
        annotation_path: Path,
        classes_path: Path,
    ) -> DatasetIterator:
        """Parse YOLOv4 annotations into LDF records.

        Annotations include classification and object detection.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation file.
            classes_path: File with class names.

        Yields:
            One record per annotation, one for every listed image carrying
            none, and one for every image the annotations do not list.

        """
        with open(classes_path, encoding="utf-8") as f:
            class_names = {
                i: line.rstrip() for i, line in enumerate(f.readlines())
            }

        base_dir = image_dir.absolute().resolve()
        annotated: set[Path] = set()

        for path, boxes in self._annotation_lines(
            base_dir, annotation_path, annotated
        ):
            file = str(path)

            # Handle image names listed with no annotation following them
            if not boxes:
                yield {"file": file, "annotation": None}
                continue

            # Hoisted out of the box loop: the size is the only thing the
            # records need the image for, so an image carrying several
            # boxes is still read once. Closed right away - only the
            # header is needed, and this streams one open image per
            # annotated file otherwise.
            with Image.open(file) as img:
                width, height = img.size

            for ann_data in boxes.split(" "):
                curr_ann_data = ann_data.split(",")
                class_name = class_names[int(curr_ann_data[4])]

                bbox_xyxy = [float(i) for i in curr_ann_data[:4]]
                yield {
                    "file": file,
                    "annotation": {
                        "class": class_name,
                        "boundingbox": {
                            "x": bbox_xyxy[0] / width,
                            "y": bbox_xyxy[1] / height,
                            "w": (bbox_xyxy[2] - bbox_xyxy[0]) / width,
                            "h": (bbox_xyxy[3] - bbox_xyxy[1]) / height,
                        },
                    },
                }

        # Which images of the directory are unlisted is only known once
        # every annotation line has been read, so they are emitted last.
        for img_path in self._unlisted_images(image_dir, base_dir, annotated):
            yield {"file": str(img_path), "annotation": None}

    def _split_files(
        self,
        image_dir: Path,
        annotation_path: Path,
        classes_path: Path,
    ) -> list[Path]:
        """List the images of one split without reading any of them.

        The annotation file names every image the split holds a record
        for, so the same walk the records make answers this too - without
        the class names, the boxes, or a single image decode.

        Missing images are reported here as well; the issue collector
        keeps one message per distinct issue, so a parse that follows does
        not report them a second time.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation file.
            classes_path: File with class names.

        Returns:
            The images the records name, in the order they name them.

        """
        del classes_path

        base_dir = image_dir.absolute().resolve()
        annotated: set[Path] = set()
        # A `dict` rather than a `set`, so that an image named by several
        # annotation lines is reported once, in the order the records name
        # it.
        files: dict[Path, None] = {}

        for path, _ in self._annotation_lines(
            base_dir, annotation_path, annotated
        ):
            files[path] = None

        for img_path in self._unlisted_images(image_dir, base_dir, annotated):
            files[img_path] = None

        return list(files)
