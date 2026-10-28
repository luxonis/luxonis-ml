from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from typing_extensions import override

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput, yolo_box_to_ldf


class Format(str, Enum):
    """YOLOv8 directory layout variant.

    Attributes:
        ROBOFLOW: Roboflow split layout.
        ULTRALYTICS: Ultralytics ``images``/``labels`` layout.

    """

    ROBOFLOW = "roboflow"
    ULTRALYTICS = "ultralytics"


class YOLOv8Parser(BaseParser):
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

    def _parse_annotation(
        self,
        elements: list[str],
        img_path: Path,
        class_names: dict[int, str],
        kpt_shape: tuple[int, int] | None,
    ) -> dict[str, Any]:
        """Parse the elements of one label line.

        Five elements are a box: ``class_id x_center y_center width
        height``. More elements are the box and keypoints when the dataset
        YAML has a ``kpt_shape``, else a polygon: ``class_id x1 y1 ...``.

        Returns:
            The LDF annotation with ``class`` and ``boundingbox``. A
            keypoint line adds ``keypoints``, and a polygon line adds
            ``instance_segmentation``.

        Raises:
            ValueError: If ``elements`` matches none of these layouts,
                or if the image of a polygon line cannot be read.

        """
        if len(elements) < 5:
            raise ValueError(
                f"Malformed YOLO annotation line: {' '.join(elements)!r}"
            )
        if len(elements) == 5:
            return {
                "class": class_names[int(elements[0])],
                "boundingbox": yolo_box_to_ldf(*elements[1:]),
            }
        if kpt_shape is not None:
            keypoints = _keypoints_array(elements[5:], kpt_shape)
            class_name = class_names[int(elements[0])]
            triples = [(p[0], p[1], int(p[2])) for p in keypoints.tolist()]
            return {
                "class": class_name,
                "boundingbox": yolo_box_to_ldf(*elements[1:5]),
                "keypoints": {"keypoints": triples},
            }

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        height, width = img.shape[:2]
        points = np.array([float(p) for p in elements[1:]]).reshape(-1, 2)
        boundingbox = self.fit_boundingbox(points)
        return {
            "class": class_names[int(elements[0])],
            "boundingbox": boundingbox,
            "instance_segmentation": {
                "height": height,
                "width": width,
                "points": [(p[0], p[1]) for p in points.tolist()],
            },
        }

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
    def _detect_dataset_dir_format(
        dataset_dir: Path,
    ) -> tuple[Format | None, list[str]]:
        """Detect whether a dataset uses the Ultralytics or Roboflow layout.

        Returns:
            The detected format and a list of folder names. For
            Roboflow, the list holds the present ``train``, ``valid``
            and ``test`` folders. For Ultralytics, it holds all folders
            in ``dataset_dir``. ``None`` and an empty list if neither
            layout matches.

        """
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
    @override
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None

        candidates = [
            (
                split_path / "images",
                split_path / "labels",
                split_path.parent,
            ),  # ROBOFLOW
            (
                split_path.parent.parent / "images" / split_path.name,
                split_path.parent.parent / "labels" / split_path.name,
                split_path.parent.parent,
            ),  # ULTRALYTICS
        ]

        images_path = labels_path = yaml_root = None
        for img_dir, lbl_dir, yroot in candidates:
            if img_dir.exists() and lbl_dir.exists():
                images_path, labels_path, yaml_root = img_dir, lbl_dir, yroot
                break

        if images_path is None or labels_path is None or yaml_root is None:
            return None

        images = BaseParser._list_images(images_path)
        if not images:
            return None

        yaml_file = next(
            (f for ext in ("*.yaml", "*.yml") for f in yaml_root.glob(ext)),
            None,
        )
        if yaml_file is None:
            return None

        return {
            "image_dir": images_path,
            "annotation_dir": labels_path,
            "classes_path": yaml_file,
        }

    @classmethod
    @override
    def validate(cls, dataset_dir: Path) -> bool:
        dir_format, splits = cls._detect_dataset_dir_format(dataset_dir)
        if dir_format is None:
            return False

        yaml_file = next(
            (f for ext in ("*.yaml", "*.yml") for f in dataset_dir.glob(ext)),
            None,
        )
        if not yaml_file:
            return False

        root = dataset_dir
        if dir_format is Format.ULTRALYTICS:
            root = dataset_dir / "images"
            splits = [d.name for d in root.iterdir() if d.is_dir()]
        if "train" not in splits or len(splits) < 2:
            return False
        return all(cls.validate_split(root / split) for split in splits)

    @classmethod
    @override
    def discover_dir_splits(
        cls, dataset_dir: Path
    ) -> dict[str, dict[str, Any]]:
        # Split roots may live under images/<split> instead of <split>/.
        dir_format, _splits = cls._detect_dataset_dir_format(dataset_dir)
        if dir_format is None:
            return {}

        discovered: dict[str, dict[str, Any]] = {}
        if dir_format is Format.ROBOFLOW:
            split_paths = [
                dataset_dir / split_name
                for split_name in ("train", "valid", "test")
            ]
        else:
            split_paths = [
                dataset_dir / "images" / split_name
                for split_name in ("train", "val", "test")
            ]

        for split_path in split_paths:
            split_kwargs = cls.validate_split(split_path)
            if split_kwargs is None:
                continue
            discovered[cls._canonicalize_split_name(split_path.name)] = (
                split_kwargs
            )
        return discovered

    @override
    def from_dir(
        self, dataset_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Parse all YOLOv8 splits in a source dataset directory.

        Args:
            dataset_dir: Source dataset directory containing split
                folders and one class YAML file.

        Returns:
            Added images for the train, validation, and test splits.

        Raises:
            ValueError: If the dataset directory does not contain a class
                YAML file.

        """
        yaml_file = next(
            (f for ext in ("*.yaml", "*.yml") for f in dataset_dir.glob(ext)),
            None,
        )
        if not yaml_file:
            raise ValueError("Exactly one yaml file is expected")
        classes_path = dataset_dir / yaml_file.name
        dir_format, _splits = self._detect_dataset_dir_format(dataset_dir)
        added_train_imgs = self._parse_split(
            image_dir=(
                dataset_dir / "images" / "train"
                if dir_format is Format.ULTRALYTICS
                else dataset_dir / "train" / "images"
            ),
            annotation_dir=(
                dataset_dir / "labels" / "train"
                if dir_format is Format.ULTRALYTICS
                else dataset_dir / "train" / "labels"
            ),
            classes_path=classes_path,
        )
        added_val_imgs = self._parse_split(
            image_dir=(
                dataset_dir / "images" / "val"
                if dir_format is Format.ULTRALYTICS
                else dataset_dir / "valid" / "images"
            ),
            annotation_dir=(
                dataset_dir / "labels" / "val"
                if dir_format is Format.ULTRALYTICS
                else dataset_dir / "valid" / "labels"
            ),
            classes_path=classes_path,
        )
        added_test_imgs = self._parse_split(
            image_dir=(
                dataset_dir / "images" / "test"
                if dir_format is Format.ULTRALYTICS
                else dataset_dir / "test" / "images"
            ),
            annotation_dir=(
                dataset_dir / "labels" / "test"
                if dir_format is Format.ULTRALYTICS
                else dataset_dir / "test" / "labels"
            ),
            classes_path=classes_path,
        )

        return added_train_imgs, added_val_imgs, added_test_imgs

    @override
    def from_split(
        self, image_dir: Path, annotation_dir: Path, classes_path: Path
    ) -> ParserOutput:
        """Parse YOLOv8 or Ultralytics annotations into LDF records.

        Annotations include object detection, instance segmentation and
        keypoints.

        Args:
            image_dir: Directory with images.
            annotation_dir: Directory with annotations.
            classes_path: YAML file with class names.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        with classes_path.open() as f:
            classes_data = yaml.safe_load(f)

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

        kpt_shape = classes_data.get("kpt_shape")

        def generator() -> DatasetIterator:
            for img_path in self._list_images(image_dir):
                ann_path = annotation_dir / img_path.with_suffix(".txt").name
                lines = _annotation_lines(ann_path)
                if not lines:
                    yield {"file": str(img_path), "annotation": None}
                    continue

                for instance_id, line in enumerate(lines):
                    annotation = self._parse_annotation(
                        line.split(), img_path, class_names, kpt_shape
                    )
                    annotation["instance_id"] = instance_id
                    yield {"file": str(img_path), "annotation": annotation}

        added_images = self._get_added_images(generator())

        return generator(), {}, added_images


def _annotation_lines(ann_path: Path) -> list[str]:
    """Return the non-blank lines of a label file; none for a missing file.

    Returns:
        The non-blank lines, each with its line ending kept.

    Examples:
        >>> _annotation_lines(Path("no_such_labels.txt"))
        []

    """
    if not ann_path.exists():
        return []
    with ann_path.open() as f:
        return [line for line in f if line.strip()]


def _keypoints_array(
    elements: list[str], kpt_shape: tuple[int, int]
) -> np.ndarray:
    """Reshape flat keypoint elements into ``(x, y, visibility)`` rows.

    Two-dimensional keypoints get full visibility.

    Returns:
        A float array with one ``(x, y, visibility)`` row per keypoint.

    Examples:
        >>> _keypoints_array(["0.1", "0.2", "0.3", "0.4"], (2, 2)).tolist()
        [[0.1, 0.2, 2.0], [0.3, 0.4, 2.0]]

    """
    n_kpts, kpt_dim = kpt_shape
    keypoints = np.array([float(p) for p in elements]).reshape(n_kpts, kpt_dim)
    if kpt_dim == 2:
        # add full visibility as last dimension
        keypoints = np.concatenate(
            [keypoints, np.ones((n_kpts, 1)) * 2], axis=1
        )
    return keypoints
