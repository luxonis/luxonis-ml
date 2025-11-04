from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import polars as pl
from PIL import Image

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.export_utils import PreparedLDF
from luxonis_ml.data.utils import COCOFormat


class CocoExporter(BaseExporter):
    """Exporter for COCO dataset format (Roboflow/FiftyOne variants)."""

    def __init__(
        self, dataset_identifier: str, format: COCOFormat = COCOFormat.ROBOFLOW
    ):
        super().__init__(dataset_identifier)
        self.format = format
        self.class_name_to_category_id = {
            s: {} for s in self.get_split_names()
        }
        self.last_category_id = {s: 1 for s in self.get_split_names()}
        self.class_to_keypoints: dict[str, list[str]] = {}
        self.image_registry = {s: {} for s in self.get_split_names()}

    @staticmethod
    def dataset_type() -> str:
        return "COCO"

    @staticmethod
    def supported_annotation_types() -> list[str]:
        return ["boundingbox", "segmentation", "keypoints"]

    @staticmethod
    def construct_empty_entry() -> dict[str, Any]:
        return {
            "id": 0,
            "image_id": 0,
            "category_id": None,
            "bbox": [],
            "area": 0,
            "iscrowd": 0,
        }

    def get_split_names(self) -> dict[str, str]:
        if self.format == COCOFormat.ROBOFLOW:
            return {"train": "train", "val": "valid", "test": "test"}
        return {"train": "train", "val": "validation", "test": "test"}

    def annotation_filename(self, split: str | None = None) -> str:
        return (
            "_annotations.coco.json"
            if self.format == COCOFormat.ROBOFLOW
            else "labels.json"
        )

    def transform(
        self, prepared_ldf: PreparedLDF
    ) -> dict[str, dict[str, Any]]:
        splits = self.get_split_names()
        annotation_splits = {
            s: {"images": [], "categories": [], "annotations": []}
            for s in splits
        }
        seen_classes = {s: [] for s in splits}
        ann_counter = {s: 0 for s in splits}
        grouped = prepared_ldf.processed_df.groupby(
            ["file", "instance_id", "group_id"], maintain_order=True
        )
        self.check_group_file_correspondence(prepared_ldf)

        for key, entry in grouped:
            key = cast(tuple[str, str, str], key)
            file_name, instance_id, group_id = key
            split = next(
                (
                    s
                    for s, gids in prepared_ldf.splits.items()
                    if group_id in gids
                ),
                None,
            )
            if not split:
                continue

            path = Path(str(file_name))
            img_id, w, h = self._get_or_register_image(
                path, split, annotation_splits
            )
            ann = self._make_base_annotation(ann_counter, split, img_id)
            if instance_id == -1:
                continue

            for row in entry.iter_rows(named=True):
                self._process_row(
                    row, split, seen_classes, annotation_splits, ann, w, h
                )
            annotation_splits[split]["annotations"].append(ann)

        self._attach_keypoints_to_categories(annotation_splits)
        return annotation_splits

    def _get_or_register_image(
        self, path: Path, split: str, annotation_splits: dict
    ) -> tuple[int, int, int]:
        idx = self.image_indices.setdefault(path, len(self.image_indices))
        new_name = f"{idx}{path.suffix}"
        return self.register_image(
            str(path), split, annotation_splits, new_name
        )

    def _make_base_annotation(
        self, counters: dict, split: str, img_id: int
    ) -> dict:
        ann = self.construct_empty_entry()
        ann["id"] = counters[split]
        ann["image_id"] = img_id
        counters[split] += 1
        return ann

    def _process_row(
        self,
        row: dict,
        split: str,
        seen: dict,
        ann_splits: dict,
        ann: dict,
        w: int,
        h: int,
    ) -> None:
        ttype, ann_str, cname = (
            row["task_type"],
            row["annotation"],
            row["class_name"],
        )

        if cname and cname not in seen[split]:
            cat_id = self.last_category_id[split]
            ann_splits[split]["categories"].append(
                {"id": cat_id, "name": cname}
            )
            seen[split].append(cname)
            self.class_name_to_category_id[split][cname] = cat_id
            self.last_category_id[split] += 1

        if ttype == "classification":
            ann["category_id"] = self.class_name_to_category_id[split][cname]
        elif ttype == "boundingbox":
            self._fill_bbox(ann, json.loads(ann_str), w, h, split, cname)
        elif ttype == "keypoints":
            self._fill_keypoints(
                ann, json.loads(ann_str)["keypoints"], split, cname, row, w, h
            )

    def _fill_bbox(
        self, ann: dict, data: dict, w: int, h: int, split: str, cname: str
    ) -> None:
        ann["bbox"] = [
            data["x"] * w,
            data["y"] * h,
            data["w"] * w,
            data["h"] * h,
        ]
        ann["area"] = (data["w"] * w) * (data["h"] * h)
        ann["category_id"] = self.class_name_to_category_id[split][cname]

    def _fill_keypoints(
        self,
        ann: dict,
        keypoints: list,
        split: str,
        cname: str,
        row: dict,
        w: int,
        h: int,
    ) -> None:
        ann["num_keypoints"] = len(keypoints)
        cat_id = ann.get("category_id") or self.class_name_to_category_id[
            split
        ].get(cname)
        if cat_id is None:
            return
        if cat_id not in self.class_to_keypoints:
            self.class_to_keypoints[cat_id] = [
                f"{row['task_name']}_{i}" for i in range(len(keypoints))
            ]
        ann["keypoints"] = [
            round(c, 2) if isinstance(c, float) else c
            for x, y, v in keypoints
            for c in (x * w, y * h, v)
        ]

    def _attach_keypoints_to_categories(self, annotation_splits: dict) -> None:
        for split_data in annotation_splits.values():
            for cat in split_data["categories"]:
                cid = cat["id"]
                if cid in self.class_to_keypoints:
                    kps = self.class_to_keypoints[cid]
                    cat["keypoints"], cat["skeleton"] = (
                        kps,
                        [[i, i + 1] for i in range(1, len(kps))],
                    )

    def register_image(
        self,
        file_name: str,
        split: str,
        annotation_splits: dict[str, dict[str, Any]],
        new_file_name: str,
    ) -> tuple[int, int, int]:
        images_list = annotation_splits[split]["images"]

        if file_name in self.image_registry[split]:
            img_data = self.image_registry[split][file_name]
            return img_data["id"], img_data["width"], img_data["height"]

        pil_image = Image.open(file_name)
        width, height = pil_image.size
        image_id = len(images_list) + 1

        img_entry = {
            "id": image_id,
            "file_name": new_file_name,
            "height": height,
            "width": width,
        }

        images_list.append(img_entry)
        self.image_registry[split][file_name] = img_entry

        return image_id, width, height

    @staticmethod
    def check_group_file_correspondence(prepared_ldf: PreparedLDF) -> None:
        df = prepared_ldf.processed_df

        group_to_files = df.groupby("group_id").agg(
            pl.col("file").n_unique().alias("file_count")
        )

        invalid_groups = group_to_files.filter(pl.col("file_count") > 1)

        assert invalid_groups.is_empty(), (
            f"Each group_id must correspond to exactly one file. "
            f"Found groups with multiple files: {invalid_groups['group_id'].to_list()}"
        )

    def _compute_annotations_size(
        self, transformed_data: dict, split: str
    ) -> int:
        return sys.getsizeof(transformed_data[split])

    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        split_name = self.get_split_names().get(split, split)
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        data_path = base / split_name
        if self.format == COCOFormat.FIFTYONE:
            data_path = data_path / "data"
        return data_path
