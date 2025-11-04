from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
            split: {} for split in self.get_split_names()
        }
        self.last_category_id = {split: 1 for split in self.get_split_names()}

        self.class_to_keypoints = {}
        self.image_registry = {split: {} for split in self.get_split_names()}

    @staticmethod
    def dataset_type() -> str:
        return "COCO"

    @staticmethod
    def supported_annotation_types() -> list[str]:
        return ["boundingbox", "segmentation", "keypoints"]

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
        annotation_splits = {
            split: {"images": [], "categories": [], "annotations": []}
            for split in self.get_split_names()
        }
        seen_class_names = {split: [] for split in self.get_split_names()}
        annotation_counter = {split: 0 for split in self.get_split_names()}
        grouped = prepared_ldf.grouped_df.groupby(
            ["file", "instance_id", "group_id"], maintain_order=True
        )
        self.check_group_file_correspondence(prepared_ldf)
        for (
            (file_name, instance_id, group_id),
            entry,
        ) in grouped:
            path = Path(file_name)
            index = self.image_indices.setdefault(
                path, len(self.image_indices)
            )
            new_filename = f"{index}{path.suffix}"
            split_name = None
            for split, group_ids in prepared_ldf.splits.items():
                if group_id in group_ids:
                    split_name = split
                    break
            final_entry = self.construct_empty_entry()
            im_id, im_width, im_height = self.register_image(
                file_name, split_name, annotation_splits, new_filename
            )
            final_entry["id"], final_entry["image_id"] = (
                annotation_counter[split_name],
                im_id,
            )
            annotation_counter[split_name] += 1
            if instance_id == -1:
                continue  # skip semantic segmentation
            for row in entry.iter_rows(named=True):
                task_type = row["task_type"]
                annotation_str = row["annotation"]
                class_name = row["class_name"]
                if (
                    class_name is not None
                    and class_name not in seen_class_names[split_name]
                ):
                    cat_id = self.last_category_id[split_name]
                    annotation_splits[split_name]["categories"].append(
                        {"id": cat_id, "name": class_name}
                    )
                    seen_class_names[split_name].append(class_name)
                    self.class_name_to_category_id[split_name][class_name] = (
                        cat_id
                    )
                    self.last_category_id[split_name] += 1
                if task_type == "classification":
                    final_entry["category_id"] = (
                        self.class_name_to_category_id[split_name][class_name]
                    )
                if task_type == "boundingbox":
                    ann_data = json.loads(annotation_str)
                    final_entry["bbox"] = [
                        ann_data["x"] * im_width,
                        ann_data["y"] * im_height,
                        ann_data["w"] * im_width,
                        ann_data["h"] * im_height,
                    ]
                    final_entry["area"] = (ann_data["w"] * im_width) * (
                        ann_data["h"] * im_height
                    )
                    final_entry["category_id"] = (
                        self.class_name_to_category_id[split_name][class_name]
                    )
                elif task_type == "keypoints":
                    ann_data = json.loads(annotation_str)["keypoints"]
                    final_entry["num_keypoints"] = len(ann_data)
                    if (
                        class_name is None
                        or final_entry["category_id"]
                        not in self.class_to_keypoints
                    ):
                        class_name = final_entry["category_id"]
                        self.class_to_keypoints[class_name] = [
                            row["task_name"] + "_" + str(i)
                            for i in range(len(ann_data))
                        ]
                        final_entry["keypoints"] = [
                            coord
                            for x, y, v in ann_data
                            for coord in (x * im_width, y * im_height, v)
                        ]
            annotation_splits[split_name]["annotations"].append(final_entry)

        for split_name, split_data in annotation_splits.items():
            for category in split_data["categories"]:
                cat_id = category["id"]
                if cat_id in self.class_to_keypoints:
                    category["keypoints"] = self.class_to_keypoints[cat_id]
                    category["skeleton"] = [
                        [i, i + 1]
                        for i in range(1, len(category["keypoints"]))
                    ]

        return annotation_splits

    def register_image(
        self, file_name, split, annotation_splits, new_file_name
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
    def construct_empty_entry():
        return {
            "id": 0,
            "image_id": 0,
            "category_id": None,
            "bbox": [],
            "area": 0,
            "iscrowd": 0,
        }

    @staticmethod
    def check_group_file_correspondence(prepared_ldf: PreparedLDF) -> None:
        df = prepared_ldf.grouped_df

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
