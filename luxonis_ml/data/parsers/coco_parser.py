"""Parser for datasets in the COCO format."""

import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_util
from loguru import logger
from typing_extensions import override

from luxonis_ml.data import DatasetIterator
from luxonis_ml.data.utils import COCOFormat
from luxonis_ml.data.utils.enums import ParserIssue
from luxonis_ml.utils.path import resolve_manifest_path

from .base_parser import BaseParser, ParserOutput


class COCOParser(BaseParser):
    """Parse a directory with COCO annotations into LDF.

    Expected formats::

        dataset_dir/
        ├── train/
        │   ├── data/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   └── labels.json
        ├── validation/
        │   ├── data/
        │   └── labels.json
        └── test/
            ├── data/
            └── labels.json

        This is default format returned when using FiftyOne package.

    or::

        dataset_dir/
            ├── train/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            │   └── _annotations.coco.json
            ├── valid/
            └── test/

        This is one of the formats that Roboflow can generate.
    """

    @staticmethod
    def _detect_dataset_dir_format(
        dataset_dir: Path,
    ) -> tuple[COCOFormat | None, list[str]]:
        """Detect whether a dataset uses FiftyOne or Roboflow layout."""
        existing = [d.name for d in dataset_dir.iterdir() if d.is_dir()]

        # Clash with NATIVE format
        if "val" in existing:
            return None, []

        splits = {
            COCOFormat.FIFTYONE: [
                s for s in ("train", "validation", "test") if s in existing
            ],
            COCOFormat.ROBOFLOW: [
                s for s in ("train", "valid", "test") if s in existing
            ],
        }
        fiftyone_splits = splits[COCOFormat.FIFTYONE]
        roboflow_splits = splits[COCOFormat.ROBOFLOW]

        if fiftyone_splits and len(fiftyone_splits) == len(roboflow_splits):
            dir_format = COCOParser._disambiguate_dir_format(
                dataset_dir, existing, fiftyone_splits
            )
            if dir_format is not None:
                return dir_format, splits[dir_format]

        if fiftyone_splits and len(fiftyone_splits) >= len(roboflow_splits):
            return COCOFormat.FIFTYONE, fiftyone_splits
        if roboflow_splits:
            return COCOFormat.ROBOFLOW, roboflow_splits
        return None, []

    @staticmethod
    def _disambiguate_dir_format(
        dataset_dir: Path, existing: list[str], splits: list[str]
    ) -> COCOFormat | None:
        """Tell FiftyOne from Roboflow when both layouts match equally.

        The validation split name decides when it is present. Partial
        layouts like train-only are ambiguous by directory names alone,
        so the annotation filename inside any present split decides:
        Roboflow has ``_annotations.coco.json``, FiftyOne has
        ``labels.json``.
        """
        if "validation" in existing:
            return COCOFormat.FIFTYONE
        if "valid" in existing:
            return COCOFormat.ROBOFLOW

        for split_name in splits:
            split_info = COCOParser.validate_split(dataset_dir / split_name)
            if split_info is None:
                continue
            if split_info["annotation_path"].name == "_annotations.coco.json":
                return COCOFormat.ROBOFLOW
            return COCOFormat.FIFTYONE
        return None

    @staticmethod
    def _is_coco_json(json_path: Path) -> bool:
        """Check if JSON file has required COCO format fields."""
        try:
            with open(json_path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return False
            # images is required, annotations is optional (test sets don't have them)
            if "images" not in data:
                return False
            # Categories can be at top level or nested inside info
            return "categories" in data or (
                "info" in data
                and isinstance(data["info"], dict)
                and "categories" in data["info"]
            )
        except (json.JSONDecodeError, OSError):
            return False

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        if not split_path.exists():
            return None
        json_path = next(split_path.glob("*.json"), None)
        if not json_path:
            return None
        if json_path.name == "_annotations.coco.json":
            if not COCOParser._is_coco_json(json_path):
                return None
            logger.info("Identified Roboflow format")
            image_dir = split_path
            return {"image_dir": image_dir, "annotation_path": json_path}
        if not COCOParser._is_coco_json(json_path):
            return None
        logger.info("Identified FiftyOne format")
        dirs = [d for d in split_path.iterdir() if d.is_dir()]
        if len(dirs) != 1:
            return None
        image_dir = dirs[0]
        return {"image_dir": image_dir, "annotation_path": json_path}

    @classmethod
    def validate(cls, dataset_dir: Path) -> bool:
        dir_format, splits = cls._detect_dataset_dir_format(dataset_dir)
        if dir_format is None:
            return False

        return all(cls.validate_split(dataset_dir / split) for split in splits)

    @classmethod
    @override
    def discover_dir_splits(
        cls, dataset_dir: Path
    ) -> dict[str, dict[str, Any]]:
        dir_format, splits = cls._detect_dataset_dir_format(dataset_dir)
        if dir_format is None:
            return {}

        discovered: dict[str, dict[str, Any]] = {}
        for split_name in splits:
            split_kwargs = cls.validate_split(dataset_dir / split_name)
            if split_kwargs is None:
                continue
            discovered[cls._canonicalize_split_name(split_name)] = split_kwargs
        return discovered

    def from_dir(
        self,
        dataset_dir: Path,
        use_keypoint_ann: bool = False,
        keypoint_ann_paths: dict[str, str] | None = None,
        split_val_to_test: bool = True,
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Parse all COCO splits in a source dataset directory.

        Args:
            dataset_dir: Source dataset directory.
            use_keypoint_ann: Whether to use separate COCO keypoint
                annotation files for FiftyOne-style datasets.
            keypoint_ann_paths: Optional paths to keypoint annotation files,
                relative to ``dataset_dir`` and keyed by ``"train"``,
                ``"val"``, and ``"test"``.
            split_val_to_test: Whether to split validation images
                :math:`50/50` into validation and test when no test
                annotations are available.

        Returns:
            Added images for the train, validation, and test splits.

        Raises:
            ValueError: If the dataset directory, train split, validation
                split, or present test split is not in a recognized COCO
                layout.

        """
        split_definitions = self._parse_available_splits(
            dataset_dir,
            use_keypoint_ann=use_keypoint_ann,
            keypoint_ann_paths=keypoint_ann_paths,
            split_val_to_test=split_val_to_test,
        )
        return (
            split_definitions.get("train", []),
            split_definitions.get("val", []),
            split_definitions.get("test", []),
        )

    def _resolve_dir_format_and_keypoint_paths(
        self,
        dataset_dir: Path,
        use_keypoint_ann: bool,
        keypoint_ann_paths: dict[str, str] | None,
    ) -> tuple[COCOFormat, list[str], dict[str, str] | None]:
        dir_format, splits = COCOParser._detect_dataset_dir_format(dataset_dir)
        if dir_format is None:
            raise ValueError("Dataset is not in any expected format.")

        if dir_format is COCOFormat.ROBOFLOW:
            logger.warning(
                "Roboflow dataset format detected, following arguments won't be taken "
                "into account: ['use_keypoint_ann', 'keypoint_ann_paths', 'split_val_to_test']."
            )
        elif (
            dir_format is COCOFormat.FIFTYONE
            and use_keypoint_ann
            and not keypoint_ann_paths
        ):
            keypoint_ann_paths = {
                "train": "raw/person_keypoints_train2017.json",
                "val": "raw/person_keypoints_val2017.json",
                # NOTE: this file is not present by default
                "test": "raw/person_keypoints_test2017.json",
            }
        return dir_format, splits, keypoint_ann_paths

    def _finalize_split_definitions(
        self,
        split_definitions: dict[str, list[Path]],
        split_val_to_test: bool,
    ) -> dict[str, list[Path]]:
        added_test_imgs = split_definitions.get("test", [])
        added_val_imgs = split_definitions.get("val", [])

        if len(added_test_imgs) == 0 and split_val_to_test and added_val_imgs:
            split_point = round(len(added_val_imgs) * 0.5)
            split_definitions["val"] = added_val_imgs[:split_point]
            split_definitions["test"] = added_val_imgs[split_point:]
        elif (
            len(added_test_imgs) == 0
            and not split_val_to_test
            and "test" in split_definitions
        ):
            logger.warning(
                "Sampling from the test set cannot be done since the "
                "labels are missing. This is expected for COCO datasets "
                "where the test set annotations are not publicly available."
            )
        return split_definitions

    def from_split(
        self, image_dir: Path, annotation_path: Path
    ) -> ParserOutput:
        """Parse COCO annotations into LDF records.

        Annotations include classification, segmentation, object detection,
        and keypoints when present.

        The method also builds the skeleton metadata. It reads every
        category that holds both a ``keypoints`` field and a ``skeleton``
        field. It uses ``keypoints`` as the keypoint labels, and it
        converts the ``skeleton`` pairs to edges. COCO numbers the skeleton
        joints from 1, so the method subtracts 1 from each index.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation JSON file.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        with open(annotation_path) as f:
            annotation_data = json.load(f)

        coco_categories = annotation_data.get("categories", [])
        categories = {cat["id"]: cat["name"] for cat in coco_categories}
        skeletons = {
            categories[cat["id"]]: {
                "labels": cat["keypoints"],
                "edges": [
                    tuple(edge)
                    for edge in (np.array(cat["skeleton"]) - 1).tolist()
                ],
            }
            for cat in coco_categories
            if "keypoints" in cat and "skeleton" in cat
        }

        records = partial(
            self._records,
            image_dir,
            annotation_path,
            annotation_data,
            categories,
            skip_unlabeled=bool(skeletons),
        )
        added_images = self._get_added_images(records())

        return records(), skeletons, added_images

    def _records(
        self,
        image_dir: Path,
        annotation_path: Path,
        annotation_data: dict[str, Any],
        categories: dict[int, str],
        *,
        skip_unlabeled: bool,
    ) -> DatasetIterator:
        """Yield one record per COCO annotation of an existing image.

        An image without annotations yields one record with no
        annotation, unless ``skip_unlabeled`` is set.
        """
        coco_annotations = annotation_data.get("annotations", [])
        img_dict = {img["id"]: img for img in annotation_data["images"]}
        ann_dict: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco_annotations:
            ann_dict[ann["image_id"]].append(ann)
        instance_ids = _InstanceIds(coco_annotations)
        root = image_dir.absolute().resolve()

        for img_id, img in img_dict.items():
            file = resolve_manifest_path(root, img["file_name"])
            if not file.exists():
                self._warn_skipped_annotation(
                    ParserIssue.MISSING_IMAGE,
                    "referenced image file does not exist",
                    source=annotation_path,
                    image=file,
                )
                continue

            img_anns = ann_dict.get(img_id, [])
            img_size = (img["height"], img["width"])
            if not img_anns and not skip_unlabeled:
                yield {"file": file, "annotation": None}

            for ann in img_anns:
                annotation = self._annotation(
                    ann,
                    img_size,
                    categories,
                    instance_ids,
                    source=annotation_path,
                    image=file,
                )
                if annotation is not None:
                    yield {"file": file, "annotation": annotation}

    def _annotation(
        self,
        ann: dict[str, Any],
        img_size: tuple[int, int],
        categories: dict[int, str],
        instance_ids: "_InstanceIds",
        *,
        source: Path,
        image: Path,
    ) -> dict[str, Any] | None:
        """Convert one COCO annotation, or warn and return ``None``."""
        if ann.get("iscrowd"):
            self._warn_skipped_annotation(
                ParserIssue.COCO_ISCROWD,
                "COCO annotation has iscrowd=1",
                source=source,
                image=image,
                annotation_id=ann.get("id"),
            )
            return None

        class_name = categories[ann["category_id"]]
        img_h, img_w = img_size
        segmentation = _coco_segmentation(
            ann.get("segmentation", []), img_h, img_w
        )
        bbox = _coco_bbox(ann["bbox"], img_w, img_h)
        if bbox is None:
            self._warn_skipped_annotation(
                ParserIssue.NON_NUMERIC_ANNOTATION,
                "Annotation contains non-numeric bbox values",
                source=source,
                image=image,
                annotation_id=ann.get("id"),
            )
            return None

        annotation: dict[str, Any] = {
            "class": class_name,
            "instance_id": instance_ids.of(ann),
            "boundingbox": bbox,
        }
        if segmentation is not None:
            annotation["segmentation"] = segmentation
            annotation["instance_segmentation"] = segmentation
        if "keypoints" in ann:
            annotation["keypoints"] = {
                "keypoints": _coco_keypoints(ann["keypoints"], img_w, img_h)
            }
        return annotation

    @override
    def _parse_available_splits(
        self,
        dataset_dir: Path,
        use_keypoint_ann: bool = False,
        keypoint_ann_paths: dict[str, str] | None = None,
        split_val_to_test: bool = True,
    ) -> dict[str, list[Path]]:
        dir_format, splits, keypoint_ann_paths = (
            self._resolve_dir_format_and_keypoint_paths(
                dataset_dir,
                use_keypoint_ann=use_keypoint_ann,
                keypoint_ann_paths=keypoint_ann_paths,
            )
        )

        split_definitions: dict[str, list[Path]] = {}

        for split_name in splits:
            split_kwargs = COCOParser.validate_split(dataset_dir / split_name)
            if split_kwargs is None:
                raise ValueError(
                    f"{split_name.title()} split not in expected format"
                )

            canonical_name = self._canonicalize_split_name(split_name)
            annotation_path = split_kwargs["annotation_path"]

            if (
                keypoint_ann_paths
                and use_keypoint_ann
                and dir_format is COCOFormat.FIFTYONE
            ):
                annotation_path = (
                    dataset_dir / keypoint_ann_paths[canonical_name]
                )
                if canonical_name == "test" and not annotation_path.exists():
                    logger.warning(
                        f"Keypoint annotation file not found: {annotation_path}. "
                        "Skipping test split."
                    )
                    split_definitions["test"] = []
                    continue

            if canonical_name == "train":
                annotation_path = clean_annotations(annotation_path)

            split_definitions[canonical_name] = self._parse_split(
                image_dir=split_kwargs["image_dir"],
                annotation_path=annotation_path,
            )

        return self._finalize_split_definitions(
            split_definitions, split_val_to_test
        )


class _InstanceIds:
    """Hand out the instance ID of each COCO annotation.

    An annotation without an ``id`` gets the lowest non-negative integer
    that no annotation of the split uses.
    """

    def __init__(self, annotations: list[dict[str, Any]]):
        self._taken = {ann["id"] for ann in annotations if "id" in ann}
        self._next = 0

    def of(self, ann: dict[str, Any]) -> int:
        """Return the instance ID of one annotation."""
        if "id" in ann:
            return ann["id"]
        while self._next in self._taken:
            self._next += 1
        instance_id = self._next
        self._next += 1
        return instance_id


def _coco_bbox(
    bbox: list[Any], img_w: int, img_h: int
) -> dict[str, float] | None:
    """Normalize a COCO ``[x, y, w, h]`` box, or ``None`` if not numeric."""
    try:
        x, y, w, h = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return None
    return {"x": x / img_w, "y": y / img_h, "w": w / img_w, "h": h / img_h}


def _coco_segmentation(
    coco_seg: Any, img_h: int, img_w: int
) -> dict[str, Any] | None:
    """Convert a COCO polygon list or RLE dict to an LDF RLE dict."""
    if isinstance(coco_seg, list):
        if not coco_seg:
            return None
        coco_seg = mask_util.merge(
            mask_util.frPyObjects(coco_seg, img_h, img_w)
        )
    elif not isinstance(coco_seg, dict):
        return None
    return {
        "height": coco_seg["size"][0],
        "width": coco_seg["size"][1],
        "counts": coco_seg["counts"],
    }


def _coco_keypoints(
    flat: list[float], img_w: int, img_h: int
) -> list[tuple[float, float, int]]:
    """Normalize a flat COCO keypoint list, clipped to the image."""
    kpts = np.array(flat).reshape(-1, 3)
    np.clip(kpts[:, 0], 0, img_w, out=kpts[:, 0])
    np.clip(kpts[:, 1], 0, img_h, out=kpts[:, 1])
    return [(kp[0] / img_w, kp[1] / img_h, int(kp[2])) for kp in kpts]


def clean_annotations(annotation_path: Path) -> Path:
    """Remove COCO images that are known to cause parsing issues.

    Args:
        annotation_path: Annotation JSON file.

    Returns:
        Path to the cleaned annotation JSON file.

    """
    files_to_avoid = [
        "000000341448.jpg",
        "000000279522.jpg",
        "000000090169.jpg",
        "000000321238.jpg",
        "000000242807.jpg",
        "000000297126.jpg",
        "000000411274.jpg",
        "000000407259.jpg",
        "000000446141.jpg",
        "000000373199.jpg",
        "000000410810.jpg",
        "000000397819.jpg",
        "000000578492.jpg",
        "000000531721.jpg",
    ]
    with open(annotation_path) as f:
        annotation_data = json.load(f)

    filtered_images = [
        img
        for img in annotation_data["images"]
        if img["file_name"] not in files_to_avoid
    ]

    if len(filtered_images) == len(annotation_data["images"]):
        return annotation_path

    filtered_image_ids = {img["id"] for img in filtered_images}
    filtered_annotations = [
        ann
        for ann in annotation_data["annotations"]
        if ann["image_id"] in filtered_image_ids
    ]

    annotation_data["images"] = filtered_images
    annotation_data["annotations"] = filtered_annotations

    cleaned_annotation_path = annotation_path.with_name("labels_fixed.json")
    with open(cleaned_annotation_path, "w") as f:
        json.dump(annotation_data, f)

    return cleaned_annotation_path
