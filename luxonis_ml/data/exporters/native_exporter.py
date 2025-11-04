from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import polars as pl

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.export_utils import PreparedLDF


class NativeExporter(BaseExporter):
    """Exporter for LDF format."""

    @staticmethod
    def dataset_type() -> str:
        return "NATIVE"

    @staticmethod
    def supported_annotation_types() -> list[str]:
        return [
            "boundingbox",
            "segmentation",
            "keypoints",
            "instance_segmentation",
        ]

    def get_split_names(self) -> dict[str, str]:
        return {"train": "train", "val": "val", "test": "test"}

    def transform(
        self,
        prepared_ldf: PreparedLDF,
        output_path: Path,
        max_partition_size_gb: float | None = None,
    ) -> None:
        annotation_splits = {split: [] for split in self.get_split_names()}
        grouped_image_sources = prepared_ldf.grouped_image_sources

        current_size = 0
        part = 0 if max_partition_size_gb else None
        max_partition_size = (
            max_partition_size_gb * 1024**3 if max_partition_size_gb else None
        )

        grouped_df = prepared_ldf.processed_df.group_by(
            "group_id", maintain_order=True
        )
        copied_files = set()

        for group_id, group_df in grouped_df:
            matched_df = grouped_image_sources.filter(
                pl.col("group_id") == group_id
            )
            group_files = matched_df.get_column("file").to_list()
            group_source_names = matched_df.get_column("source_name").to_list()

            split = next(
                (
                    s
                    for s, group_ids in prepared_ldf.splits.items()
                    if group_id in group_ids
                ),
                None,
            )
            assert split is not None

            annotation_records = []
            for row in group_df.iter_rows(named=True):
                record = self._process_row(
                    row=row,
                    group_source_names=group_source_names,
                    group_files=group_files,
                )
                annotation_records.append(record)

            annotations_size = sum(
                sys.getsizeof(r) for r in annotation_records
            )
            group_total_size = sum(Path(f).stat().st_size for f in group_files)

            if (
                max_partition_size
                and part is not None
                and (current_size + group_total_size + annotations_size)
                > max_partition_size
            ):
                self._dump_annotations(annotation_splits, output_path, part)
                current_size = 0
                part += 1
                annotation_splits = {
                    split: [] for split in self.get_split_names()
                }

            data_path = self._get_data_path(output_path, split, part)
            data_path.mkdir(parents=True, exist_ok=True)

            for file in group_files:
                file_path = Path(file)
                if file_path not in copied_files:
                    copied_files.add(file_path)
                    image_index = self.image_indices[file_path]
                    dest_file = data_path / f"{image_index}{file_path.suffix}"
                    shutil.copy(file_path, dest_file)
                    current_size += file_path.stat().st_size

            annotation_splits[split].extend(annotation_records)

        self._dump_annotations(annotation_splits, output_path, part)

    def _process_row(
        self,
        row: dict[str, Any],
        group_source_names: list[str],
        group_files: list[str],
    ) -> dict[str, Any]:
        task_name = row["task_name"]
        class_name = row["class_name"]
        instance_id = row["instance_id"]
        task_type = row["task_type"]
        ann_str = row["annotation"]

        source_to_file = {}
        for name, f in zip(group_source_names, group_files, strict=True):
            path = Path(f)
            index = self.image_indices.setdefault(
                path, len(self.image_indices)
            )

            new_filename = f"{index}{path.suffix}"
            new_path = Path("images") / new_filename

            source_to_file[name] = str(new_path.as_posix())

        record = {
            "files" if len(group_source_names) > 1 else "file": (
                source_to_file
                if len(group_source_names) > 1
                else source_to_file[group_source_names[0]]
            ),
            "task_name": task_name,
        }

        if ann_str is not None:
            data = json.loads(ann_str)
            annotation_base = {
                "instance_id": instance_id,
                "class": class_name,
            }
            if task_type in self.supported_annotation_types():
                annotation_base[task_type] = data
            elif task_type.startswith("metadata/"):
                annotation_base["metadata"] = {task_type[9:]: data}
            record["annotation"] = annotation_base

        return record

    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        if part is not None:
            return (
                output_path
                / f"{self.dataset_identifier}_part{part}"
                / split
                / "images"
            )
        return output_path / self.dataset_identifier / split / "images"
