from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator

from .parser_plugin import SplitParserPlugin


class DarknetParser(SplitParserPlugin):
    """Parse a directory with Darknet annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img1.txt
        │   ├── ...
        │   └── _darknet.labels
        ├── valid/
        └── test/

    This is one of the formats that Roboflow can generate.
    """

    dataset_types = ("darknet",)

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        if not (split_path / "_darknet.labels").exists():
            return None
        # Recognizing a split already requires listing its images, so the
        # listing is handed on as a parse argument instead of being dropped
        # and rebuilt by `_split_records`.
        image_paths = DarknetParser._list_images(split_path)
        if not image_paths:
            return None
        return {
            "image_dir": split_path,
            "classes_path": split_path / "_darknet.labels",
            "image_paths": image_paths,
        }

    def _split_files(
        self,
        image_dir: Path,
        classes_path: Path,
        image_paths: list[Path] | None = None,
    ) -> list[Path]:
        """List the images of one split without reading its labels.

        Every listed image produces at least one record - one without
        annotations still yields a record with `annotation` set to None -
        so the listing already is the file list, in the very order the
        records refer to it.

        Args:
            image_dir: Directory with images.
            classes_path: File with class names, which the file list does
                not depend on.
            image_paths: Images of the split, as already listed by
                `validate_split`. Listed from `image_dir` when omitted.

        Returns:
            The images of the split.

        """
        del classes_path
        if image_paths is None:
            return self._list_images(image_dir)
        return image_paths

    def _split_records(
        self,
        image_dir: Path,
        classes_path: Path,
        image_paths: list[Path] | None = None,
    ) -> DatasetIterator:
        """Parse Darknet annotations into LDF records.

        Annotations include classification and object detection.

        Args:
            image_dir: Directory with images.
            classes_path: File with class names.
            image_paths: Images of the split, as already listed by
                `validate_split`. Listed from `image_dir` when omitted.

        Returns:
            One record per annotation, plus one with no annotation for
            every image that carries none.

        """
        # Read before the generator is entered so that an unreadable class
        # file fails the parse itself rather than half-way through an
        # import, and so that the label files stay unread until the records
        # are actually pulled.
        with open(classes_path) as f:
            class_names = {
                i: line.rstrip() for i, line in enumerate(f.readlines())
            }

        # `validate_split` hands its listing on, so this only runs for a
        # caller that assembled the arguments itself.
        if image_paths is None:
            image_paths = self._list_images(image_dir)

        def generator() -> DatasetIterator:
            for img_path in image_paths:
                ann_path = img_path.with_suffix(".txt")
                file = str(img_path)
                if ann_path.exists():
                    with open(ann_path) as f:
                        annotation_data = f.readlines()
                else:
                    annotation_data = []

                if not annotation_data:
                    yield {"file": file, "annotation": None}
                    continue

                for ann_line in annotation_data:
                    class_id, x_center, y_center, width, height = (
                        ann_line.split(" ")
                    )
                    class_name = class_names[int(class_id)]
                    # Both extents are needed twice by the centered box;
                    # converting once is bit for bit the same float.
                    box_width = float(width)
                    box_height = float(height)

                    yield {
                        "file": file,
                        "annotation": {
                            "class": class_name,
                            "boundingbox": {
                                "x": float(x_center) - box_width / 2,
                                "y": float(y_center) - box_height / 2,
                                "w": box_width,
                                "h": box_height,
                            },
                        },
                    }

        return generator()
