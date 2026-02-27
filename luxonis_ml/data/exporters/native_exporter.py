from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import polars as pl

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    exporter_specific_annotation_warning,
    split_of_group,
)
from luxonis_ml.enums import DatasetType


class NativeExporter(BaseExporter):
    """Exporter for LDF format."""

    @staticmethod
    def get_split_names() -> dict[str, str]:
        return {"train": "train", "val": "val", "test": "test"}

    def supported_ann_types(self) -> list[str]:
        return list(DatasetType.NATIVE.supported_annotation_formats)

    def export(self, prepared_ldf: PreparedLDF) -> None:
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        annotation_splits: dict[str, list[dict[str, Any]]] = {
            k: [] for k in self.get_split_names()
        }
        grouped_image_sources = prepared_ldf.grouped_image_sources
        grouped_df = prepared_ldf.processed_df.group_by(
            "group_id", maintain_order=True
        )
        copied_files: set[Path] = set()

        for group_id, group_df in grouped_df:
            split = split_of_group(prepared_ldf, group_id)

            matched_df = grouped_image_sources.filter(
                pl.col("group_id") == group_id
            )
            group_files = matched_df.get_column("file").to_list()
            group_source_names = matched_df.get_column("source_name").to_list()

            records = [
                self._process_row(row, group_source_names, group_files)
                for row in group_df.iter_rows(named=True)
            ]

            ann_size = sum(sys.getsizeof(r) for r in records)
            img_size = sum(Path(f).stat().st_size for f in group_files)
            annotation_splits = self._maybe_roll_partition(
                annotation_splits, ann_size + img_size
            )

            data_path = self._get_data_path(self.output_path, split, self.part)
            data_path.mkdir(parents=True, exist_ok=True)

            for f in group_files:
                p = Path(f)
                if p not in copied_files:
                    copied_files.add(p)
                    idx = self.image_indices[p]
                    shutil.copy(p, data_path / f"{idx}{p.suffix}")
                    self.current_size += p.stat().st_size

            annotation_splits[split].extend(records)

        self._dump_annotations(annotation_splits, self.output_path, self.part)

    def _maybe_roll_partition(
        self,
        annotation_splits: dict[str, list[dict[str, Any]]],
        additional_size: int,
    ) -> dict[str, list[dict[str, Any]]]:
        if (
            self.max_partition_size
            and self.part is not None
            and (self.current_size + additional_size) > self.max_partition_size
        ):
            self._dump_annotations(
                annotation_splits, self.output_path, self.part
            )
            self.current_size = 0
            self.part += 1
            return {k: [] for k in self.get_split_names()}
        return annotation_splits

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

        source_to_file = {
            name: str(
                Path("images")
                / f"{self.image_indices.setdefault(Path(f), len(self.image_indices))}{Path(f).suffix}"
            )
            for name, f in zip(group_source_names, group_files, strict=True)
        }

        multi_source = len(source_to_file) > 1
        record: dict[str, Any] = {
            ("files" if multi_source else "file"): (
                source_to_file
                if multi_source
                else source_to_file[group_source_names[0]]
            ),
            "task_name": task_name,
        }

        if ann_str is not None:
            data = json.loads(ann_str)
            ann: dict[str, Any] = {
                "instance_id": instance_id,
                "class": class_name,
            }
            if task_type in (
                "boundingbox",
                "segmentation",
                "instance_segmentation",
                "keypoints",
            ):
                ann[task_type] = data
            elif task_type.startswith("metadata/"):
                ann["metadata"] = {task_type[9:]: data}
            record["annotation"] = ann

        return record

    def _dump_annotations(
        self,
        annotation_splits: dict[str, list[dict[str, Any]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        for split_name, items in annotation_splits.items():
            save_name = self.get_split_names().get(split_name, split_name)
            base = (
                output_path / f"{self.dataset_identifier}_part{part}"
                if part is not None
                else output_path / self.dataset_identifier
            )
            split_path = base / save_name
            split_path.mkdir(parents=True, exist_ok=True)
            (split_path / "annotations.json").write_text(
                json.dumps(items, indent=4), encoding="utf-8"
            )

    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        return base / split / "images"
