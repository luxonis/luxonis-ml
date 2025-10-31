from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
import polars as pl
from .base_exporter import BaseExporter
from ..parsers.coco_parser import Format


class CocoExporter(BaseExporter):
    """Exporter for COCO Roboflow dataset format"""

    def __init__(self, dataset_identifier: str, format=Format.ROBOFLOW):
        super().__init__(dataset_identifier)
        self.format = format

    @staticmethod
    def dataset_type() -> str:
        return "COCO"

    @staticmethod
    def supported_annotation_types() -> List[str]:
        return ["boundingbox", "segmentation", "keypoints"]

    def get_split_names(self) -> Dict[str, str]:
        if self.format == Format.ROBOFLOW:
            return {"train": "train", "val": "valid", "test": "test"}
        return {"train": "train", "val": "validation", "test": "test"} # (FiftyOne format)

    def transform(self, prepared_ldf) -> Dict[str, Dict[str, Any]]:
        """Convert native LDF annotations to COCO format per split."""
        annotation_splits = {
            split: {"images": [], "annotations": [], "categories": []}
            for split in self.get_split_names().keys()
        }

        grouped_image_sources = prepared_ldf.grouped_image_sources
        image_indices = prepared_ldf.image_indices

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
                    file_name = f"{image_indices.setdefault(Path(f), len(image_indices))}{Path(f).suffix}"
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

    def native_keypoints_to_coco(self, ann_json):
        return

    def native_boundingbox_to_coco(self, ann_json):
        return

    def native_segmentation_to_coco(self, ann_json):
        return

    def save(
            self,
            transformed_data: Dict[str, Dict[str, Any]],
            prepared_ldf,
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
            max_partition_size_gb * 1024 ** 3 if max_partition_size_gb else None
        )

        split_name_map = self.get_split_names()

        def _dump_annotations(annotations, output_path, identifier, part=None):
            for split_name, annotation_data in annotations.items():
                save_name = split_name_map.get(split_name, split_name)
                if part is not None:
                    split_path = output_path / f"{identifier}_part{part}" / save_name
                else:
                    split_path = output_path / identifier / save_name
                split_path.mkdir(parents=True, exist_ok=True)
                annotation_file_name = "_annotations.coco.json" if self.format == Format.ROBOFLOW else "labels.json"
                with open(split_path / annotation_file_name, "w") as f:
                    json.dump(annotation_data, f, indent=4)

        # Copy images, handle partitions
        for group_id, group_df in prepared_ldf.grouped_df:
            matched_df = prepared_ldf.grouped_image_sources.filter(pl.col("group_id") == group_id)
            group_files = matched_df.get_column("file").to_list()
            split = next(
                (s for s, group_ids in prepared_ldf.splits.items() if group_id in group_ids),
                None,
            )
            assert split is not None

            group_total_size = sum(Path(f).stat().st_size for f in group_files)
            annotations_size = sys.getsizeof(transformed_data[split])

            if (
                    max_partition_size
                    and part is not None
                    and current_size + group_total_size + annotations_size > max_partition_size
            ):
                _dump_annotations(transformed_data, output_path, self.dataset_identifier, part)
                current_size = 0
                part += 1

            save_name = split_name_map.get(split, split)
            if max_partition_size:
                data_path = output_path / f"{self.dataset_identifier}_part{part}" / save_name
            else:
                data_path = output_path / self.dataset_identifier / save_name
            if self.format == Format.FIFTYONE:
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

        _dump_annotations(transformed_data, output_path, self.dataset_identifier, part)

        # Optional ZIP export
        if zip_output:
            archives = []
            if max_partition_size:
                for i in range(part + 1):
                    folder = output_path / f"{self.dataset_identifier}_part{i}"
                    if folder.exists():
                        archive_file = shutil.make_archive(str(folder), "zip", root_dir=folder)
                        archives.append(Path(archive_file))
            else:
                folder = output_path / self.dataset_identifier
                if folder.exists():
                    archive_file = shutil.make_archive(str(folder), "zip", root_dir=folder)
                    archives.append(Path(archive_file))
            return archives if len(archives) > 1 else archives[0]

        return output_path
