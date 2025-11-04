from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from luxonis_ml.data.exporters.prepared_ldf import PreparedLDF


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
        split names."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supported_annotation_types() -> list[str]:
        """List of annotation types supported by this exporter."""
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        prepared_ldf: PreparedLDF,
        output_path: Path,
        max_partition_size_gb: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Convert the prepared dataset into the exporter's format."""
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
                    )  # or .json depending on subclass
                    with open(ann_path, "w") as f:
                        f.write(item["annotation_content"])
            else:
                # Single annotation file for split (e.g., )annotations.coco.json for train split)
                ann_filename = self.annotation_filename(split_name)
                with open(split_path / ann_filename, "w") as f:
                    json.dump(annotation_data, f, indent=4)

    def create_zip_output(
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
