from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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

        self.class_name_to_category_id = {}
        self.last_category_id = 0

        self.image_name_to_id = {}
        self.last_image_id = 0

        self.class_to_keypoints = {}

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
        return "_annotations.coco.json"

    def transform(
        self, prepared_ldf: PreparedLDF
    ) -> dict[str, dict[str, Any]]:
        annotation_splits = {
            split: {"images": [], "annotations": [], "categories": []}
            for split in self.get_split_names()
        }

        # Here assert that each group is one image since we cannot have depth and rgb yet in this export format
        grouped = prepared_ldf.grouped_df.groupby(
            ["file", "instance_id", "group_id"], maintain_order=True
        )

        annotation_counter = 0  # we need a separate counter per split
        for (
            (file_name, instance_id, group_id),
            entry,
        ) in (
            grouped
        ):  # this is fine because we asserted that each file has 1 group_id
            # here we register the image if the relative file_name is not in the annotations
            split_name = None
            for split, group_ids in prepared_ldf.splits.items():
                if group_id in group_ids:
                    split_name = split
                    break
            final_entry = self.construct_empty_entry()
            if (
                file_name not in self.image_name_to_id
            ):  # find a way to register the height and the width too
                self.image_name_to_id[file_name] = self.last_image_id
                self.last_image_id += 1
            final_entry["id"] = annotation_counter
            annotation_counter += 1
            if instance_id == -1:
                continue  # skip semantic segmentation for now (will see how to integrate it)
            for row in entry.iter_rows(named=True):
                annotation_str = row["annotation"]
                class_name = row["class_name"]
                if class_name not in self.class_name_to_category_id:
                    self.class_name_to_category_id[class_name] = (
                        self.last_category_id
                    )
                    self.last_category_id += 1
                task_type = row["task_type"]
                if (
                    task_type == "classification"
                    and final_entry["category_id"] is None
                ):
                    final_entry["category_id"] = (
                        self.class_name_to_category_id[class_name]
                    )
                elif task_type == "boundingbox":
                    ann_data = json.loads(annotation_str)
                    final_entry["bbox"] = [
                        ann_data["x"],
                        ann_data["y"],
                        ann_data["w"],
                        ann_data["h"],
                    ]
                    final_entry["area"] = ann_data["w"] * ann_data["h"]
                elif task_type == "keypoints":
                    ann_data = json.loads(annotation_str)["keypoints"]
                    if (
                        class_name is None
                        or final_entry["category_id"]
                        not in self.class_to_keypoints.keys()
                    ):
                        class_name = final_entry["category_id"]
                        self.class_to_keypoints[class_name] = [
                            row["task_name"] + "_" + str(i)
                            for i in range(len(ann_data))
                        ]
                        final_entry["keypoints"] = ann_data
                    print()
            annotation_splits[split_name]["annotations"].append(final_entry)

        return annotation_splits

    @staticmethod
    def construct_empty_entry():
        return {
            "id": 0,
            "image_id": 0,
            "category_id": None,
            "bbox": [],
            "area": 0,
            "segmentation": [],
            "iscrowd": 0,
        }

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
