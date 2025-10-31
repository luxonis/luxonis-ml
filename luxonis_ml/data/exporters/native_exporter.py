from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
import polars as pl
from .base_exporter import BaseExporter


class NativeExporter(BaseExporter):
    """Exporter for LDF format."""

    @staticmethod
    def dataset_type() -> str:
        return "NATIVE"

    @staticmethod
    def supported_annotation_types() -> List[str]:
        return ["boundingbox", "segmentation", "keypoints", "metadata"]

    def get_split_names(self) -> Dict[str, str]:
        return {
            "train": "train",
            "val": "val",
            "test": "test"
        }

    def transform(self, prepared_ldf) -> Dict[str, List[Dict[str, Any]]]:
        annotation_splits = {split: [] for split in self.get_split_names()}
        grouped_image_sources = prepared_ldf.grouped_image_sources
        image_indices = prepared_ldf.image_indices

        for group_id, group_df in prepared_ldf.grouped_df:
            matched_df = grouped_image_sources.filter(
                pl.col("group_id") == group_id
            )
            group_files = matched_df.get_column("file").to_list()
            group_source_names = matched_df.get_column("source_name").to_list()

            split = next(
                (s for s, group_ids in prepared_ldf.splits.items() if group_id in group_ids),
                None,
            )
            assert split is not None

            annotation_records = []
            for row in group_df.iter_rows(named=True):
                task_name = row["task_name"]
                class_name = row["class_name"]
                instance_id = row["instance_id"]
                task_type = row["task_type"]
                ann_str = row["annotation"]

                source_to_file = {
                    name: str(
                        (Path("images") / f"{image_indices.setdefault(Path(f), len(image_indices))}{Path(f).suffix}").as_posix()
                    )
                    for name, f in zip(group_source_names, group_files, strict=True)
                }

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
                    if task_type in {"instance_segmentation", "segmentation", "boundingbox", "keypoints"}:
                        annotation_base[task_type] = data
                    elif task_type.startswith("metadata/"):
                        annotation_base["metadata"] = {task_type[9:]: data}
                    record["annotation"] = annotation_base

                annotation_records.append(record)

            annotation_splits[split].extend(annotation_records)

        return annotation_splits

    def save(
        self,
        transformed_data: Dict[str, List[Dict[str, Any]]],
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
            max_partition_size_gb * 1024**3 if max_partition_size_gb else None
        )

        def _dump_annotations(annotations, output_path, identifier, part=None):
            for split_name, annotation_data in annotations.items():
                if part is not None:
                    split_path = output_path / f"{identifier}_part{part}" / split_name
                else:
                    split_path = output_path / identifier / split_name
                split_path.mkdir(parents=True, exist_ok=True)
                with open(split_path / "annotations.json", "w") as f:
                    json.dump(annotation_data, f, indent=4)

        for group_id, group_df in prepared_ldf.grouped_df:
            matched_df = prepared_ldf.grouped_image_sources.filter(
                pl.col("group_id") == group_id
            )
            group_files = matched_df.get_column("file").to_list()
            split = next(
                (s for s, group_ids in prepared_ldf.splits.items() if group_id in group_ids),
                None,
            )
            assert split is not None

            group_total_size = sum(Path(f).stat().st_size for f in group_files)
            annotations_size = sum(sys.getsizeof(r) for r in transformed_data[split])

            if (
                max_partition_size
                and part is not None
                and current_size + group_total_size + annotations_size > max_partition_size
            ):
                _dump_annotations(transformed_data, output_path, self.dataset_identifier, part)
                current_size = 0
                part += 1

            if max_partition_size:
                data_path = output_path / f"{self.dataset_identifier}_part{part}" / split / "images"
            else:
                data_path = output_path / self.dataset_identifier / split / "images"
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