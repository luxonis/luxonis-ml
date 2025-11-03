from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import polars as pl

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.utils import COCOFormat


class CocoExporter(BaseExporter):
    """Exporter for COCO Roboflow dataset format"""

    def __init__(self, dataset_identifier: str, format: COCOFormat = COCOFormat.ROBOFLOW):
        super().__init__(dataset_identifier)
        self.format = format

    @staticmethod
    def dataset_type() -> str:
        return "COCO"

    def get_split_names(self) -> Dict[str, str]:
        if self.format == COCOFormat.ROBOFLOW:
            return {"train": "train", "val": "valid", "test": "test"}
        return {"train": "train", "val": "validation", "test": "test"}  # (FiftyOne format)

    @staticmethod
    def supported_annotation_types() -> List[str]:
        return ["boundingbox", "segmentation", "keypoints"]

    def annotation_filename(self, split: str | None = None) -> str:
        return "_annotations_coco.json"

    def transform(self, prepared_ldf) -> Dict[str, Dict[str, Any]]:
        """Convert native LDF annotations to COCO format per split."""
        annotation_splits = {
            split: {"images": [], "annotations": [], "categories": []}
            for split in self.get_split_names().keys()
        }

        grouped_image_sources = prepared_ldf.grouped_image_sources

        # Track IDs
        class_to_id: Dict[str, int] = {}
        image_id_map: Dict[str, int] = {}
        ann_id = 1

        for group_id, group_df in prepared_ldf.grouped_df:
            matched_df = grouped_image_sources.filter(pl.col("group_id") == group_id)
            group_files = matched_df.get_column("file").to_list()
            group_source_names = matched_df.get_column("source_name").to_list()

            split = next(
                (s for s, group_ids in prepared_ldf.splits.items() if group_id in group_ids),
                None,
            )
            assert split is not None
            coco_split = annotation_splits[split]

            for row in group_df.iter_rows(named=True):
                task_type = row["task_type"]
                class_name = row["class_name"]
                ann_str = row["annotation"]
                instance_id = row["instance_id"]

                # Assign category IDs
                if class_name not in class_to_id:
                    class_to_id[class_name] = len(class_to_id) + 1

                # For each file in this group
                for name, f in zip(group_source_names, group_files, strict=True):
                    file_name = f"{self.image_indices.setdefault(Path(f), len(self.image_indices))}{Path(f).suffix}"
                    file_path = Path("images") / file_name

                    # Register image once
                    if str(file_path) not in image_id_map:
                        image_id = len(image_id_map) + 1
                        image_id_map[str(file_path)] = image_id
                        coco_split["images"].append({
                            "id": image_id,
                            "file_name": file_name,
                            "width": None,
                            "height": None
                        })
                    else:
                        image_id = image_id_map[str(file_path)]

                    if not ann_str:
                        continue

                    data = json.loads(ann_str)

                    bbox = []
                    segmentation = []
                    keypoints = []
                    num_keypoints = 0

                    if task_type == "boundingbox" and data:
                        # Assume normalized [x, y, w, h]
                        bbox = [data["x"], data["y"], data["w"], data["h"]]

                    elif task_type in {"segmentation", "instance_segmentation"}:
                        segmentation = data if isinstance(data, list) else [data]

                    elif task_type == "keypoints":
                        # Flatten list of (x,y,visible)
                        keypoints = [v for kp in data for v in kp]
                        num_keypoints = len(data)

                    coco_split["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_to_id[class_name],
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "area": bbox[2] * bbox[3] if bbox else 0,
                        "iscrowd": 0,
                        "keypoints": keypoints,
                        "num_keypoints": num_keypoints,
                        "instance_id": instance_id
                    })
                    ann_id += 1

        # Add category definitions to each split
        for split, coco_split in annotation_splits.items():
            coco_split["categories"] = [
                {"id": cid, "name": cname}
                for cname, cid in class_to_id.items()
            ]

        return annotation_splits

    def _compute_annotations_size(self, transformed_data, split):
        return sys.getsizeof(transformed_data[split])

    def _get_data_path(self, output_path, split, part):
        save_name = self.get_split_names().get(split, split)
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        data_path = base / save_name
        if self.format == COCOFormat.FIFTYONE:
            data_path = data_path / "data"
        return data_path
