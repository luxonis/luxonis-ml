from enum import Enum
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import yaml
from typing_extensions import override

from luxonis_ml.data import DatasetIterator
from luxonis_ml.data.utils.enums import ParserIssue

from .parser_plugin import Layout, SplitParserPlugin, centered_box


class Format(str, Enum):
    """YOLOv8 directory layout variant.

    Attributes:
        ROBOFLOW: Roboflow split layout.
        ULTRALYTICS: Ultralytics ``images``/``labels`` layout.

    """

    ROBOFLOW = "roboflow"
    ULTRALYTICS = "ultralytics"


class YOLOv8Parser(SplitParserPlugin):
    """Parse YOLOv8 and Ultralytics annotations into LDF.

    Expected format::

        dataset_dir/
        ├── images/
        │   ├── train/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
        │   │   └── ...
        │   ├── val/
        │   └── test/
        └── *.yaml

        OR::

        dataset_dir/
        ├── train/
        │   ├── images/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   ├── labels/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
        │   │   └── ...
        ├── valid/
        │   ├── images/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
        │   │   └── ...
        │   ├── labels/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
        │   │   └── ...
        ├── test/
        │   ├── images/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
        │   │   └── ...
        │   ├── labels/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
        │   │   └── ...
        └── *.yaml

    ``*.yaml`` contains all class names.

    This is one of the formats that Roboflow can generate.
    """

    dataset_types = (
        "yolov8",
        "yolov8instancesegmentation",
        "yolov8keypoints",
    )

    def fit_boundingbox(self, points: np.ndarray) -> dict[str, float]:
        """Fit a bounding box around a polygon mask."""
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])
        return {
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min,
            "h": y_max - y_min,
        }

    @staticmethod
    def _reshape_keypoints(
        coordinates: list[float], n_kpts: Any, kpt_dim: Any
    ) -> list[tuple[float, float, int]]:
        """Group flat keypoint values into ``(x, y, visibility)``.

        Args:
            coordinates: Flat keypoint values of one annotation line.
            n_kpts: Number of keypoints declared by ``kpt_shape``.
            kpt_dim: Values per keypoint declared by ``kpt_shape``.

        Returns:
            One triplet per keypoint, with full visibility added when the
            annotations carry none.

        """
        keypoints = np.array(coordinates).reshape(n_kpts, kpt_dim)
        if kpt_dim == 2:
            # add full visibility as last dimension
            keypoints = np.concatenate(
                [keypoints, np.ones((n_kpts, 1)) * 2], axis=1
            )
        return [(p[0], p[1], int(p[2])) for p in keypoints.tolist()]

    @staticmethod
    def _has_polygon(annotation_path: Path) -> bool:
        """Report whether a label file holds a polygon line.

        Only meaningful where the split declares no ``kpt_shape``, in
        which case a line of more than five values is a polygon - the one
        annotation whose image the parse has to read.

        Args:
            annotation_path: Label file belonging to one image. It does
                not have to exist.

        Returns:
            Whether any line of the file is a polygon.

        """
        if not annotation_path.exists():
            return False

        with open(annotation_path, encoding="utf-8") as f:
            for line in f:
                # Six fields already answer the question.
                if len(line.split(None, 5)) > 5:
                    return True
        return False

    @staticmethod
    def _detect_dataset_dir_format(
        dataset_dir: Path,
    ) -> tuple[Format | None, list[str]]:
        """Detect whether a dataset uses Ultralytics or Roboflow
        layout.
        """
        if not dataset_dir.is_dir():
            return None, []

        roboflow_splits = ("train", "valid", "test")
        ultralytics_folders = ["images", "labels"]

        existing = [d.name for d in dataset_dir.iterdir() if d.is_dir()]

        present_roboflow_splits = [
            split for split in roboflow_splits if split in existing
        ]
        if present_roboflow_splits:
            return Format.ROBOFLOW, present_roboflow_splits
        if all(folder in existing for folder in ultralytics_folders):
            return Format.ULTRALYTICS, existing
        return None, []

    @staticmethod
    def _find_classes_yaml(yaml_roots: tuple[Path, ...]) -> Path | None:
        """Return the YAML holding the class names, if there is one.

        Args:
            yaml_roots: Directories to search, most specific first.

        Returns:
            ``data.yaml`` where the export named it that, as both Roboflow
            and Ultralytics do, so an unrelated YAML beside it cannot win.
            Otherwise the first YAML of the first directory that has one,
            sorted so a directory holding several resolves consistently.

        """
        for yaml_root in yaml_roots:
            candidates = [
                file
                for ext in ("*.yaml", "*.yml")
                for file in sorted(yaml_root.glob(ext))
            ]
            if not candidates:
                continue
            return next(
                (file for file in candidates if file.stem == "data"),
                candidates[0],
            )
        return None

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None

        candidates = [
            (
                split_path / "images",
                split_path / "labels",
                # A source without splits is validated as one, and then
                # `split_path` is the dataset root holding the class file.
                # Searching there first also keeps a genuine split from
                # reaching for an unrelated YAML beside the dataset.
                (split_path, split_path.parent),
            ),  # ROBOFLOW
            (
                split_path.parent.parent / "images" / split_path.name,
                split_path.parent.parent / "labels" / split_path.name,
                (split_path.parent.parent,),
            ),  # ULTRALYTICS
        ]

        images_path = labels_path = None
        yaml_roots: tuple[Path, ...] = ()
        for img_dir, lbl_dir, yroots in candidates:
            if img_dir.exists() and lbl_dir.exists():
                images_path, labels_path, yaml_roots = img_dir, lbl_dir, yroots
                break

        if images_path is None or labels_path is None or not yaml_roots:
            return None

        images = YOLOv8Parser._list_images(images_path)
        if not images:
            return None

        yaml_file = YOLOv8Parser._find_classes_yaml(yaml_roots)
        if yaml_file is None:
            return None

        return {
            "image_dir": images_path,
            "annotation_dir": labels_path,
            "classes_path": yaml_file,
        }

    @classmethod
    @override
    def detect(cls, source: Path) -> Layout | None:
        if not source.is_dir():
            return None

        # Split roots may live under images/<split> instead of <split>/,
        # which is the only reason this is not the inherited detection.
        dir_format, _splits = cls._detect_dataset_dir_format(source)
        if dir_format is Format.ROBOFLOW:
            split_paths = [
                source / split_name
                for split_name in ("train", "valid", "test")
            ]
        elif dir_format is Format.ULTRALYTICS:
            split_paths = [
                source / "images" / split_name
                for split_name in ("train", "val", "test")
            ]
        else:
            split_paths = []

        discovered: dict[str | None, dict[str, Any]] = {}
        for split_path in split_paths:
            split_kwargs = cls.validate_split(split_path)
            if split_kwargs is None:
                continue
            discovered[cls._canonicalize_split_name(split_path.name)] = (
                split_kwargs
            )
        if discovered:
            return Layout(discovered)

        split_kwargs = cls.validate_split(source)
        if split_kwargs is None:
            return None
        return Layout({None: split_kwargs})

    def _split_files(
        self,
        image_dir: Path,
        annotation_dir: Path,
        classes_path: Path,
    ) -> list[Path]:
        """List the images of one split.

        Almost the plain listing: every image yields at least one record,
        background images included. The exception is an image whose every
        annotation line is too short to parse, which yields nothing - so
        its label file is looked at, or a count-based `split_ratios` would
        sample it and leave the split a record short.

        Args:
            image_dir: Directory with images.
            annotation_dir: Directory with annotations.
            classes_path: YAML file with class names.

        Returns:
            The images of the split that yield at least one record.

        """
        del classes_path
        return [
            image
            for image in self._list_images(image_dir)
            if self._yields_records(annotation_dir / f"{image.stem}.txt")
        ]

    @staticmethod
    def _yields_records(annotation_path: Path) -> bool:
        """Return whether the image owning this label file is recorded.

        Mirrors the generator's decisions without parsing any value: a
        missing or empty label file yields one background record, a line
        of at least 5 fields yields an annotation record, shorter lines
        are skipped. Only an image whose every non-empty line is too short
        produces nothing.

        Args:
            annotation_path: Label file of one image. It need not exist.

        Returns:
            Whether parsing the image yields at least one record.

        """
        if not annotation_path.exists():
            return True

        skipped_any = False
        with open(annotation_path, encoding="utf-8") as f:
            for line in f:
                elements = line.split()
                if not elements:
                    continue
                # The first usable line settles it.
                if len(elements) >= 5:
                    return True
                skipped_any = True
        return not skipped_any

    def _split_records(
        self, image_dir: Path, annotation_dir: Path, classes_path: Path
    ) -> DatasetIterator:
        """Parse YOLOv8 or Ultralytics annotations into LDF records.

        Annotations include object detection, instance segmentation and
        keypoints.

        Args:
            image_dir: Directory with images.
            annotation_dir: Directory with annotations.
            classes_path: YAML file with class names.

        Returns:
            One record per annotation, and one per unannotated image.

        """
        with open(classes_path, encoding="utf-8") as f:
            classes_data = cast(dict[str, Any], yaml.safe_load(f))

        if isinstance(classes_data["names"], list):
            """
            names: ["class1", "class2", "class3"]
            """
            class_names = dict(enumerate(classes_data["names"]))
        else:
            """
            names:
                0: class1
                1: class2
                2: class3
            """
            class_names = classes_data["names"]

        kpt_shape = classes_data.get("kpt_shape", None)
        n_kpts = kpt_dim = 0
        if kpt_shape is not None:
            # Unpacked before any record streams: a malformed shape
            # failing inside the generator would leave a registered,
            # half-populated dataset behind. Same for the polygon
            # pre-check below.
            try:
                n_kpts, kpt_dim = kpt_shape
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid `kpt_shape` in '{classes_path}': expected "
                    f"two values, got {kpt_shape!r}."
                ) from error
        images = self._list_images(image_dir)

        if kpt_shape is None:
            # A polygon is the only annotation whose image is read, so an
            # unreadable image is only fatal here. Checking the format
            # signature costs no decode.
            for img_path in images:
                if self._has_polygon(
                    annotation_dir / f"{img_path.stem}.txt"
                ) and not cv2.haveImageReader(str(img_path)):
                    raise ValueError(f"Failed to read image: {img_path}")

        def generator() -> DatasetIterator:
            for img_path in images:
                ann_path = annotation_dir / f"{img_path.stem}.txt"
                file = str(img_path)

                annotations: list[list[str]] = []
                # First check if file exists (would crash otherwise)
                if ann_path.exists():
                    with open(ann_path, encoding="utf-8") as f:
                        # A line is empty exactly when it has no fields,
                        # so this both drops the blank lines and produces
                        # the fields every kept line is parsed into.
                        annotations = [
                            elements
                            for line in f
                            if (elements := line.split())
                        ]

                # Handle missing annotations
                if not annotations:
                    yield {"file": file, "annotation": None}
                    continue

                image_size: tuple[int, int] | None = None
                for instance_id, annotation_elements in enumerate(annotations):
                    # object detection format: class_id x_center y_center width height
                    # segmentation format: class_id x1 y1 x2 y2 x3 y3 ... xn yn (min 3 points)
                    # keypoints format: class_id x_center y_center width height kp1_x kp1_y kp2_x kp2_y ... kpn_x kpn_y (it can also have 3rd dimension for visibility)
                    n_elements = len(annotation_elements)

                    if n_elements == 5:
                        class_id, x_center, y_center, width, height = (
                            annotation_elements
                        )
                        yield {
                            "file": file,
                            "annotation": {
                                "class": class_names[int(class_id)],
                                "instance_id": instance_id,
                                "boundingbox": centered_box(
                                    x_center, y_center, width, height
                                ),
                            },
                        }

                    elif n_elements < 5:
                        self._warn_skipped_annotation(
                            ParserIssue.MALFORMED_ANNOTATION,
                            "annotation line has fewer than 5 values",
                            source=ann_path,
                            image=img_path,
                            annotation_id=instance_id,
                        )
                        continue

                    elif kpt_shape is not None:
                        (
                            class_id,
                            x_center,
                            y_center,
                            width,
                            height,
                            *keypoint_values,
                        ) = annotation_elements
                        coordinates = list(map(float, keypoint_values))

                        # Striding the flat values is the same as reading
                        # the rows of an `(n_kpts, kpt_dim)` reshape, but
                        # only when the count fits exactly. Any other shape
                        # keeps the numpy path, for its `-1` inference and
                        # its error messages.
                        if kpt_dim == 3 and len(coordinates) == n_kpts * 3:
                            keypoints = list(
                                zip(
                                    coordinates[0::3],
                                    coordinates[1::3],
                                    map(int, coordinates[2::3]),
                                    strict=True,
                                )
                            )
                        elif kpt_dim == 2 and len(coordinates) == n_kpts * 2:
                            # add full visibility as last dimension
                            keypoints = [
                                (x, y, 2)
                                for x, y in zip(
                                    coordinates[0::2],
                                    coordinates[1::2],
                                    strict=True,
                                )
                            ]
                        else:
                            keypoints = self._reshape_keypoints(
                                coordinates, n_kpts, kpt_dim
                            )

                        yield {
                            "file": file,
                            "annotation": {
                                "class": class_names[int(class_id)],
                                "instance_id": instance_id,
                                "boundingbox": centered_box(
                                    x_center, y_center, width, height
                                ),
                                "keypoints": {
                                    "keypoints": keypoints,
                                },
                            },
                        }

                    else:
                        if image_size is None:
                            # Polygons are normalized, so every polygon of
                            # one image shares this decode.
                            img = cv2.imread(file)
                            if img is None:
                                raise ValueError(
                                    f"Failed to read image: {img_path}"
                                )
                            decoded_height, decoded_width = img.shape[:2]
                            image_size = (
                                int(decoded_height),
                                int(decoded_width),
                            )
                        img_height, img_width = image_size

                        class_id, *point_values = annotation_elements
                        coordinates = list(map(float, point_values))
                        if len(coordinates) % 2:
                            # Let numpy raise on an odd count rather than
                            # silently dropping the trailing value.
                            np.array(coordinates).reshape(-1, 2)
                        xs = coordinates[0::2]
                        ys = coordinates[1::2]
                        x_min = min(xs)
                        y_min = min(ys)
                        class_name = class_names[int(class_id)]

                        yield {
                            "file": file,
                            "annotation": {
                                "class": class_name,
                                "instance_id": instance_id,
                                "boundingbox": {
                                    "x": x_min,
                                    "y": y_min,
                                    "w": max(xs) - x_min,
                                    "h": max(ys) - y_min,
                                },
                                "instance_segmentation": {
                                    "height": img_height,
                                    "width": img_width,
                                    "points": list(zip(xs, ys, strict=True)),
                                },
                            },
                        }

        return generator()
