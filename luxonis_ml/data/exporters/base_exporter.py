from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from luxonis_ml.data.exporters.export_utils import PreparedLDF


class BaseExporter(ABC):
    def __init__(self, dataset_identifier: str):
        self.dataset_identifier = dataset_identifier
        self.image_indices = {}

    @staticmethod
    @abstractmethod
    def dataset_type() -> str:
        """Return the dataset type identifier (e.g. 'NATIVE',
        'COCO')."""
        raise NotImplementedError

    @abstractmethod
    def get_split_names(self) -> dict[str, str]:
        """Return mapping from native split names to dataset-appropriate
        split names e.g.: For COCO Roboflow: {"train", "train", "val":
        "valid", "test": "test"}"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supported_annotation_types() -> list[str]:
        """List of annotation types supported by this exporter."""
        raise NotImplementedError

    @abstractmethod
    def transform(
        self, prepared_ldf: PreparedLDF
    ) -> dict[str, list[dict[str, Any]]]:
        """Convert the prepared dataset into the exporter's format."""
        raise NotImplementedError

    @abstractmethod
    def _compute_annotations_size(
        self, transformed_data: dict[str, Any], split: str
    ) -> int:
        """Return size of annotations for this split in bytes.

        Used to decide when to start a new partition
        """
        raise NotImplementedError

    @abstractmethod
    def _get_data_path(
        self, output_path: Path, split: str, part: int | None
    ) -> Path:
        """Return the folder path to store data files for this split."""
        raise NotImplementedError

    def annotations_per_image(self) -> bool:
        """Whether each image has its own annotation file (e.g. VOC) or
        one file per split (e.g. COCO Roboflow)"""
        return False

    def annotation_filename(self, split: str | None = None) -> str:
        """Return the filename for the annotation file for a given
        split.

        Default is 'annotations.json', but can be overridden per
        exporter.
        """
        return "annotations.json"

    def save(
        self,
        transformed_data: dict[str, list[dict[str, Any]]],
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

        for group_id, _group_df in prepared_ldf.grouped_df:
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
            annotations_size = self._compute_annotations_size(
                transformed_data, split
            )

            if (
                max_partition_size
                and part is not None
                and current_size + group_total_size + annotations_size
                > max_partition_size
            ):
                self._dump_annotations(transformed_data, output_path, part)
                current_size = 0
                part += 1

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

        self._dump_annotations(transformed_data, output_path, part)

        if zip_output:
            self._create_zip_output(max_partition_size, output_path, part)

        logger.info(f"Dataset successfully exported to: {output_path}")
        return output_path

    def _dump_annotations(
        self, annotations: dict, output_path: Path, part: int | None = None
    ) -> None:
        for split_name, annotation_data in annotations.items():
            save_name = str(self.get_split_names().get(split_name, split_name))
            if part is not None:
                split_path = (
                    output_path
                    / f"{self.dataset_identifier}_part{part}"
                    / save_name
                )
            else:
                split_path = output_path / self.dataset_identifier / save_name
            split_path.mkdir(parents=True, exist_ok=True)

            if self.annotations_per_image():
                # Each image has its own annotation file
                for item in annotation_data:
                    image_id = item.get("image_id") or item.get("file_name")
                    if not image_id:
                        raise ValueError(
                            "Per-image annotation missing 'image_id' or 'file_name'"
                        )
                    ann_path = (
                        split_path / f"{Path(image_id).stem}.xml"
                    )  # or .json, depending on subclass
                    with open(ann_path, "w") as f:
                        f.write(item["annotation_content"])
            else:
                # Single annotation file for split
                ann_filename = self.annotation_filename(split_name)
                with open(split_path / ann_filename, "w") as f:
                    json.dump(annotation_data, f, indent=4)

    def _create_zip_output(
        self,
        max_partition_size: float | None,
        output_path: Path,
        part: int | None,
    ) -> Path | list[Path]:
        archives: list[Path] = []

        if max_partition_size is not None and part is not None:
            for i in range(part + 1):
                folder = output_path / f"{self.dataset_identifier}_part{i}"
                if folder.exists():
                    archive_file = shutil.make_archive(
                        str(folder), "zip", root_dir=folder
                    )
                    archives.append(Path(archive_file))
        else:
            folder = output_path / self.dataset_identifier
            if folder.exists():
                archive_file = shutil.make_archive(
                    str(folder), "zip", root_dir=folder
                )
                archives.append(Path(archive_file))

        return archives if len(archives) > 1 else archives[0]
