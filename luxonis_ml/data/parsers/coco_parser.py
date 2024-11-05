import json
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput

logger = getLogger(__name__)


class Format(str, Enum):
    FIFTYONE = "fiftyone"
    ROBOFLOW = "roboflow"


class COCOParser(BaseParser):
    """Parses directory with COCO annotations to LDF.

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

        This is one of the formats that can be generated by
        U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def _detect_dataset_dir_format(
        dataset_dir: Path,
    ) -> Tuple[Optional[Format], List[str]]:
        """Checks if dataset directory structure is in FiftyOne or
        Roboflow format."""
        fiftyone_splits = ["train", "validation", "test"]
        roboflow_splits = ["train", "valid", "test"]
        if all((dataset_dir / split).exists() for split in fiftyone_splits):
            return Format.FIFTYONE, fiftyone_splits
        elif all((dataset_dir / split).exists() for split in roboflow_splits):
            return Format.ROBOFLOW, roboflow_splits
        else:
            return None, []

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        if not split_path.exists():
            return None
        json_path = next(split_path.glob("*.json"), None)
        if not json_path:
            return None
        if json_path.name == "_annotations.coco.json":
            # Roboflow format
            image_dir = split_path
            return {"image_dir": image_dir, "annotation_path": json_path}
        else:
            # FiftyOne format
            dirs = [d for d in split_path.iterdir() if d.is_dir()]
            if len(dirs) != 1:
                return None
            image_dir = dirs[0]
            return {"image_dir": image_dir, "annotation_path": json_path}

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        dir_format, splits = COCOParser._detect_dataset_dir_format(dataset_dir)
        if dir_format is None:
            return False

        for split in splits:
            split_path = dataset_dir / split
            if COCOParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(
        self,
        dataset_dir: Path,
        use_keypoint_ann: bool = False,
        keypoint_ann_paths: Optional[Dict[str, str]] = None,
        split_val_to_test: bool = True,
    ) -> Tuple[List[str], List[str], List[str]]:
        dir_format, splits = COCOParser._detect_dataset_dir_format(dataset_dir)
        if dir_format is None:
            raise ValueError("Dataset is not in any expected format.")

        elif dir_format == Format.ROBOFLOW:
            logger.warning(
                "Roboflow dataset format detected, following arguments won't be taken "
                "into account: ['use_keypoint_ann', 'keypoint_ann_paths', 'split_val_to_test']."
            )
        elif dir_format == Format.FIFTYONE:
            if use_keypoint_ann and not keypoint_ann_paths:
                keypoint_ann_paths = {
                    "train": "raw/person_keypoints_train2017.json",
                    "val": "raw/person_keypoints_val2017.json",
                    # NOTE: this file is not present by default
                    "test": "raw/person_keypoints_test2017.json",
                }

        train_paths = COCOParser.validate_split(dataset_dir / splits[0])
        if train_paths is None:
            raise ValueError("Train split not in expected format")

        train_ann_path = (
            dataset_dir / keypoint_ann_paths["train"]
            if keypoint_ann_paths
            and use_keypoint_ann
            and dir_format == Format.FIFTYONE
            else train_paths["annotation_path"]
        )
        cleaned_annotation_path = clean_annotations(train_ann_path)
        added_train_imgs = self._parse_split(
            image_dir=train_paths["image_dir"],
            annotation_path=cleaned_annotation_path,
        )

        val_paths = COCOParser.validate_split(dataset_dir / splits[1])
        if val_paths is None:
            raise ValueError("Val split not in expected format")

        val_ann_path = (
            dataset_dir / keypoint_ann_paths["val"]
            if keypoint_ann_paths
            and use_keypoint_ann
            and dir_format == Format.FIFTYONE
            else val_paths["annotation_path"]
        )
        _added_val_imgs = self._parse_split(
            image_dir=val_paths["image_dir"],
            annotation_path=val_ann_path,
        )

        if split_val_to_test and dir_format == Format.FIFTYONE:
            split_point = round(len(_added_val_imgs) * 0.5)
            added_val_imgs = _added_val_imgs[:split_point]
            added_test_imgs = _added_val_imgs[split_point:]
        else:
            added_val_imgs = _added_val_imgs
            added_test_imgs = []
            # NOTE: test split annotations are not included by default for FiftyOne format
            test_paths = COCOParser.validate_split(dataset_dir / splits[2])
            if test_paths is None:
                raise ValueError("Test split not in expected format")

            test_ann_path = (
                dataset_dir / keypoint_ann_paths["test"]
                if keypoint_ann_paths
                and use_keypoint_ann
                and dir_format == Format.FIFTYONE
                else test_paths["annotation_path"]
            )

            added_test_imgs = self._parse_split(
                image_dir=test_paths["image_dir"],
                annotation_path=test_ann_path,
            )

        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(
        self, image_dir: Path, annotation_path: Path
    ) -> ParserOutput:
        """Parses annotations from COCO format to LDF. Annotations
        include classification, segmentation, object detection and
        keypoints if present.

        @type image_dir: Path
        @param image_dir: Path to directory with images
        @type annotation_path: Path
        @param annotation_path: Path to annotation json file
        @rtype: L{ParserOutput}
        @return: Annotation generator, list of classes names, skeleton
            dictionary for keypoints and list of added images.
        """

        with open(annotation_path) as f:
            annotation_data = json.load(f)

        coco_images = annotation_data["images"]
        coco_annotations = annotation_data["annotations"]
        coco_categories = annotation_data["categories"]
        categories = {cat["id"]: cat["name"] for cat in coco_categories}

        class_names = list(categories.values())
        skeletons = {}
        for cat in coco_categories:
            if "keypoints" in cat.keys() and "skeleton" in cat.keys():
                skeletons[categories[cat["id"]]] = {
                    "labels": cat["keypoints"],
                    "edges": (np.array(cat["skeleton"]) - 1).tolist(),
                }

        def generator() -> DatasetIterator:
            img_dict = {img["id"]: img for img in coco_images}
            ann_dict = {}
            for ann in coco_annotations:
                img_id = ann["image_id"]
                if img_id not in ann_dict:
                    ann_dict[img_id] = []
                ann_dict[img_id].append(ann)

            for img_id, img in img_dict.items():
                path = image_dir.absolute().resolve() / img["file_name"]
                if not path.exists():
                    continue
                path = str(path)

                img_anns = ann_dict.get(img_id, [])
                img_h = img["height"]
                img_w = img["width"]

                for i, ann in enumerate(img_anns):
                    if ann.get("iscrowd", True):
                        continue
                    class_name = categories[ann["category_id"]]
                    yield {
                        "file": path,
                        "annotation": {
                            "type": "classification",
                            "class": class_name,
                        },
                    }

                    seg = ann["segmentation"]
                    if isinstance(seg, list):
                        if len(seg) > 0:
                            poly = []
                            for s in seg:
                                poly_arr = np.array(s).reshape(-1, 2)
                                poly += [
                                    (
                                        poly_arr[i, 0] / img_w,
                                        poly_arr[i, 1] / img_h,
                                    )
                                    for i in range(len(poly_arr))
                                ]
                            yield {
                                "file": path,
                                "annotation": {
                                    "type": "polyline",
                                    "class": class_name,
                                    "points": poly,
                                },
                            }
                    else:
                        yield {
                            "file": path,
                            "annotation": {
                                "type": "rle",
                                "class": "person",
                                "height": seg["size"][0],
                                "width": seg["size"][1],
                                "counts": seg["counts"],
                            },
                        }

                    x, y, w, h = ann["bbox"]
                    yield {
                        "file": path,
                        "annotation": {
                            "type": "boundingbox",
                            "class": class_name,
                            "instance_id": i,
                            "x": x / img_w,
                            "y": y / img_h,
                            "w": w / img_w,
                            "h": h / img_h,
                        },
                    }

                    if "keypoints" in ann.keys():
                        kpts = np.array(ann["keypoints"]).reshape(-1, 3)
                        # clip values inplace
                        np.clip(kpts[:, 0], 0, img_w, out=kpts[:, 0])
                        np.clip(kpts[:, 1], 0, img_h, out=kpts[:, 1])

                        keypoints = []
                        for kp in kpts:
                            keypoints.append(
                                (kp[0] / img_w, kp[1] / img_h, int(kp[2]))
                            )
                        yield {
                            "file": path,
                            "annotation": {
                                "type": "keypoints",
                                "class": class_name,
                                "instance_id": i,
                                "keypoints": keypoints,
                            },
                        }

        added_images = self._get_added_images(generator())

        return generator(), class_names, skeletons, added_images


def clean_annotations(annotation_path: Path) -> Path:
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
    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    filtered_images = [
        img
        for img in annotation_data["images"]
        if img["file_name"] not in files_to_avoid
    ]
    filtered_image_ids = {img["id"] for img in filtered_images}
    filtered_annotations = [
        ann
        for ann in annotation_data["annotations"]
        if ann["image_id"] in filtered_image_ids
    ]

    annotation_data["images"] = filtered_images
    annotation_data["annotations"] = filtered_annotations

    # Save the cleaned annotation file
    cleaned_annotation_path = annotation_path.with_name("labels_fixed.json")
    with open(cleaned_annotation_path, "w") as f:
        json.dump(annotation_data, f)

    return cleaned_annotation_path
