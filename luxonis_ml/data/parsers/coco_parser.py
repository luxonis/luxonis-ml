import json
import os.path as osp
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from luxonis_ml.data import DatasetGenerator

from .luxonis_parser import LuxonisParser, ParserOutput


class COCOParser(LuxonisParser):
    def validate(self, dataset_dir: Path) -> bool:
        for split in ["train", "validation", "test"]:
            split_path = dataset_dir / split
            if not split_path.exists():
                return False
            if not (split_path / "data").exists():
                return False
            if not (split_path / "labels.json").exists():
                return False
        return True

    def from_dir(
        self,
        dataset_dir: str,
        use_keypoint_ann: bool = False,
        keypoint_ann_paths: Optional[Dict[str, str]] = None,
        split_val_to_test: bool = True,
    ) -> None:
        """Parses directory with COCO annotations to LDF.

        Expected format::

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

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory

        @type use_keypoint_ann: bool
        @param use_keypoint_ann: If keypoint annotations should be used.
            Defaults to C{False}.

        @type keypoint_ann_paths: Optional[Dict[str, str]]
        @param keypoint_ann_paths: Path to keypoint annotations for each split.
            Defaults to C{None}.

        @type split_val_to_test: bool
        @param split_val_to_test: If part of validation data should be used
            as test data. Defaults to C{True}.
        """
        if use_keypoint_ann and not keypoint_ann_paths:
            keypoint_ann_paths = {
                "train": "raw/person_keypoints_train2017.json",
                "val": "raw/person_keypoints_val2017.json",
                # NOTE: this file is not present by default
                "test": "raw/person_keypoints_test2017.json",
            }

        train_ann_path = (
            osp.join(dataset_dir, keypoint_ann_paths["train"])
            if keypoint_ann_paths and use_keypoint_ann
            else osp.join(dataset_dir, "train", "labels.json")
        )
        added_train_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "train", "data"),
            annotation_path=train_ann_path,
        )

        val_ann_path = (
            osp.join(dataset_dir, keypoint_ann_paths["val"])
            if keypoint_ann_paths and use_keypoint_ann
            else osp.join(dataset_dir, "validation", "labels.json")
        )
        _added_val_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "validation", "data"),
            annotation_path=val_ann_path,
        )

        if not split_val_to_test:
            # NOTE: test split annotations are not included by default
            test_ann_path = (
                osp.join(dataset_dir, keypoint_ann_paths["test"])
                if keypoint_ann_paths and use_keypoint_ann
                else osp.join(dataset_dir, "test", "labels.json")
            )
            added_test_imgs = self.from_format(
                image_dir=osp.join(dataset_dir, "test", "data"),
                annotation_path=test_ann_path,
            )

        if split_val_to_test:
            split_point = round(len(_added_val_imgs) * 0.5)
            added_val_imgs = _added_val_imgs[:split_point]
            added_test_imgs = _added_val_imgs[split_point:]
        else:
            added_val_imgs = _added_val_imgs

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    def _from_format(self, image_dir: str, annotation_path: str) -> ParserOutput:
        """Parses annotations from COCO format to LDF. Annotations include
        classification, segmentation, object detection and keypoints if present.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_path: str
        @param annotation_path: Path to annotation json file
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

        def generator() -> DatasetGenerator:
            for img in coco_images:
                img_id = img["id"]

                path = osp.join(osp.abspath(image_dir), img["file_name"])
                if not osp.exists(path):
                    continue

                img_anns = [
                    ann for ann in coco_annotations if ann["image_id"] == img_id
                ]

                img_h = img["height"]
                img_w = img["width"]

                for ann in img_anns:
                    class_name = categories[ann["category_id"]]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

                    seg = ann["segmentation"]
                    if isinstance(seg, list):
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
                            "class": class_name,
                            "type": "polyline",
                            "value": poly,
                        }

                    x, y, w, h = ann["bbox"]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "box",
                        "value": (x / img_w, y / img_h, w / img_w, h / img_h),
                    }

                    if "keypoints" in ann.keys():
                        kpts = np.array(ann["keypoints"]).reshape(-1, 3)
                        keypoints = []
                        for kp in kpts:
                            keypoints.append(
                                ([kp[0] / img_w, kp[1] / img_h, int(kp[2])])
                            )
                        yield {
                            "file": path,
                            "class": class_name,
                            "type": "keypoints",
                            "value": keypoints,
                        }

        added_images = self._get_added_images(generator)

        return generator, class_names, skeletons, added_images
