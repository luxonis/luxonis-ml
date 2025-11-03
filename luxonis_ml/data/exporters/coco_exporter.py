from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import polars as pl
from PIL import Image

from ..utils import COCOFormat
from .base_exporter import BaseExporter
from .export_utils import PreparedLDF, create_zip_output, dump_annotations


class CocoExporter(BaseExporter):
    """Exporter for COCO Roboflow dataset format."""

    def __init__(
        self, dataset_identifier: str, format: COCOFormat = COCOFormat.ROBOFLOW
    ):
        super().__init__(dataset_identifier)
        self.format = format
        self.class_to_id: dict[str, int] = {}
        self.image_id_map: dict[str, int] = {}
        self.ann_id = 1

    @staticmethod
    def dataset_type() -> str:
        return "COCO"

    @staticmethod
    def supported_annotation_types() -> list[str]:
        return ["boundingbox", "segmentation", "keypoints"]

    def get_split_names(self) -> dict[str, str]:
        if self.format == COCOFormat.ROBOFLOW:
            return {"train": "train", "val": "valid", "test": "test"}
        return {
            "train": "train",
            "val": "validation",
            "test": "test",
        }  # (FiftyOne format)

    def transform(
        self, prepared_ldf: PreparedLDF
    ) -> dict[str, dict[str, Any]]:
        """Convert native LDF annotations to COCO format per split."""
        annotation_splits = {
            split: {"images": [], "annotations": [], "categories": []}
            for split in self.get_split_names()
        }

        for group_id, group_df in prepared_ldf.grouped_df:
            split = next(
                (
                    s
                    for s, group_ids in prepared_ldf.splits.items()
                    if group_id in group_ids
                ),
                None,
            )
            assert split is not None
            coco_split = annotation_splits[split]

            matched_df = prepared_ldf.grouped_image_sources.filter(
                pl.col("group_id") == group_id
            )
            group_files = matched_df.get_column("file").to_list()

            for row in group_df.iter_rows(named=True):
                task_type = row["task_type"]
                class_name = row["class_name"]
                instance_id = row["instance_id"]
                ann_str = row["annotation"]

                # Register class ID
                cat_id = self._get_class_id(class_name)

                for f in group_files:
                    file_path = Path(f)
                    image_id = self._register_image(file_path, coco_split)

                    if not ann_str:
                        continue

                    data = json.loads(ann_str)
                    coco_ann = self._convert_annotation(
                        task_type, data, cat_id, image_id, instance_id
                    )
                    coco_split["annotations"].append(coco_ann)

        # Finalize category list
        for split, coco_split in annotation_splits.items():
            coco_split["categories"] = [
                {"id": cid, "name": cname}
                for cname, cid in self.class_to_id.items()
            ]

        return annotation_splits

    def _get_class_id(self, class_name: str) -> int:
        if class_name not in self.class_to_id:
            self.class_to_id[class_name] = len(self.class_to_id) + 1
        return self.class_to_id[class_name]

    def _register_image(
        self, file_path: Path, coco_split: dict[str, Any]
    ) -> int:
        """Ensure each image is registered with width/height."""
        str_path = str(file_path)
        if str_path in self.image_id_map:
            return self.image_id_map[str_path]

        image_id = len(self.image_id_map) + 1
        width, height = self._get_image_size(file_path)
        coco_split["images"].append(
            {
                "id": image_id,
                "file_name": str(file_path),
                "width": width,
                "height": height,
            }
        )
        self.image_id_map[str_path] = image_id
        return image_id

    def _get_image_size(self, file_path: Path) -> tuple[int, int]:
        """Try reading image size from file; fallback to None."""
        try:
            with Image.open(file_path) as img:
                return img.width, img.height
        except Exception:
            return None, None

    def _convert_annotation(
        self,
        task_type: str,
        ann_str: str,
        category_id: int,
        image_id: int,
        instance_id: int,
    ) -> dict:
        bbox, segmentation, keypoints, num_keypoints = [], [], [], 0

        if task_type == "boundingbox":
            bbox = [data["x"], data["y"], data["w"], data["h"]]

        elif task_type in {"segmentation", "instance_segmentation"}:
            segmentation = (
                [data["segmentation"]] if "segmentation" in data else []
            )

        elif task_type == "keypoints":
            keypoints = [v for kp in data for v in kp]
            num_keypoints = len(data)

        ann = {
            "id": self.ann_id,
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
        self.ann_id += 1
        return ann

    def save(
        self,
        transformed_data: dict[str, dict[str, Any]],
        prepared_ldf: PreparedLDF,
        output_path: Path,
        max_partition_size_gb: float | None,
        zip_output: bool,
    ) -> Path | list[Path]:
        output_path = Path(output_path)
        if output_path.exists():
            raise ValueError(f"Export path '{output_path}' already exists.")
        output_path.mkdir(parents=True)

        current_size = 0
        copied_files = set()
        part = 0 if max_partition_size_gb else None
        max_partition_size = (
            max_partition_size_gb * 1024**3 if max_partition_size_gb else None
        )

        for group_id, group_df in prepared_ldf.grouped_df:
            matched_df = prepared_ldf.grouped_image_sources.filter(
                pl.col("group_id") == group_id
            )
            group_files = matched_df.get_column("file").to_list()
            split = next(
                (
                    s
                    for s, group_ids in prepared_ldf.splits.items()
                    if group_id in group_ids
                ),
                None,
            )
            assert split is not None

            group_total_size = sum(Path(f).stat().st_size for f in group_files)
            annotations_size = sys.getsizeof(transformed_data[split])

            if (
                max_partition_size
                and part is not None
                and current_size + group_total_size + annotations_size
                > max_partition_size
            ):
                dump_annotations(
                    transformed_data,
                    output_path,
                    self.dataset_identifier,
                    part,
                )
                current_size = 0
                part += 1

            if max_partition_size:
                data_path = (
                    output_path
                    / f"{self.dataset_identifier}_part{part}"
                    / self.get_split_names().get(split, split)
                )
            else:
                data_path = (
                    output_path
                    / self.dataset_identifier
                    / self.get_split_names().get(split, split)
                )
            if self.format == COCOFormat.FIFTYONE:
                data_path = data_path / "data"
            data_path.mkdir(parents=True, exist_ok=True)

            for file in group_files:
                file_path = Path(file)
                if file_path not in copied_files:
                    copied_files.add(file_path)
                    image_index = prepared_ldf.image_indices[file_path]
                    dest_file = data_path / f"{image_index}{file_path.suffix}"
                    shutil.copy(file_path, dest_file)
                    current_size += file_path.stat().st_size

        dump_annotations(
            transformed_data, output_path, self.dataset_identifier, part
        )

        if zip_output:
            create_zip_output(
                self.dataset_identifier, max_partition_size, part, output_path
            )

        return output_path
