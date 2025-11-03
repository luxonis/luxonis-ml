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

        # Internal tracking
        self.class_to_id: dict[str, int] = {}
        self.image_id_map: dict[str, int] = {}
        self.next_ann_id: int = 1

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
        return "_annotations_coco.json"

    def transform(
        self, prepared_ldf: PreparedLDF
    ) -> dict[str, dict[str, Any]]:
        annotation_splits = {
            split: {"images": [], "annotations": [], "categories": []}
            for split in self.get_split_names()
        }

        grouped_image_sources = prepared_ldf.grouped_image_sources

        for group_id, group_df in prepared_ldf.grouped_df:
            matched_df = grouped_image_sources.filter(
                pl.col("group_id") == group_id
            )
            group_files = matched_df.get_column("file").to_list()
            group_sources = matched_df.get_column("source_name").to_list()

            split = next(
                (
                    s
                    for s, gids in prepared_ldf.splits.items()
                    if group_id in gids
                ),
                None,
            )
            assert split is not None, (
                f"Group {group_id} missing split assignment"
            )

            for row in group_df.iter_rows(named=True):
                image_entries, ann_entries = self._process_row(
                    row, group_sources, group_files
                )
                annotation_splits[split]["images"].extend(image_entries)
                annotation_splits[split]["annotations"].extend(ann_entries)

        categories = [
            {"id": cid, "name": cname}
            for cname, cid in self.class_to_id.items()
        ]
        for split_data in annotation_splits.values():
            split_data["categories"] = categories

        return annotation_splits

    def _process_row(
        self,
        row: dict[str, Any],
        group_source_names: list[str],
        group_files: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert one annotation row into COCO-compatible image +
        annotation entries.

        Returns a tuple: (new_images, new_annotations)
        """
        task_type = row["task_type"]
        class_name = row["class_name"]
        instance_id = row["instance_id"]
        ann_str = row["annotation"]

        new_images: list[dict[str, Any]] = []
        new_annotations: list[dict[str, Any]] = []

        if not ann_str:
            return new_images, new_annotations

        ann_data = json.loads(ann_str)

        if class_name not in self.class_to_id:
            self.class_to_id[class_name] = len(self.class_to_id) + 1
        category_id = self.class_to_id[class_name]

        for _name, file_path in zip(
            group_source_names, group_files, strict=True
        ):
            img_entry, is_new = self._get_or_register_image(file_path)
            if is_new:
                new_images.append(img_entry)

            ann_entry = self._annotation_to_coco(
                ann_data, task_type, img_entry["id"], category_id, instance_id
            )
            new_annotations.append(ann_entry)

        return new_images, new_annotations

    def _get_or_register_image(
        self, file_path: str
    ) -> tuple[dict[str, Any], bool]:
        path = Path(file_path)
        image_name = f"{self.image_indices.setdefault(path, len(self.image_indices))}{path.suffix}"
        rel_path = Path("images") / image_name
        key = str(rel_path)

        # Only register once
        if key not in self.image_id_map:
            image_id = len(self.image_id_map) + 1
            self.image_id_map[key] = image_id

            try:
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:
                width, height = None, None

            image_entry = {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
            return image_entry, True

        # Already registered
        image_id = self.image_id_map[key]
        return {
            "id": image_id,
            "file_name": image_name,
            "width": None,
            "height": None,
        }, False

    def _annotation_to_coco(
        self,
        data: dict[str, Any],
        task_type: str,
        image_id: int,
        category_id: int,
        instance_id: Any,
    ) -> dict[str, Any]:
        """Convert annotation dict to COCO schema."""
        bbox, segmentation, keypoints, num_keypoints = [], [], [], 0

        if task_type == "boundingbox" and data:
            bbox = [
                data.get("x", 0),
                data.get("y", 0),
                data.get("w", 0),
                data.get("h", 0),
            ]
        elif task_type in {"segmentation", "instance_segmentation"}:
            segmentation = data if isinstance(data, list) else [data]
        elif task_type == "keypoints":
            keypoints = [v for kp in data for v in kp]
            num_keypoints = len(data)

        ann = {
            "id": self.next_ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "segmentation": segmentation,
            "area": bbox[2] * bbox[3] if bbox else 0,
            "iscrowd": 0,
            "keypoints": keypoints,
            "num_keypoints": num_keypoints,
            "instance_id": instance_id,
        }

        self.next_ann_id += 1
        return ann

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
