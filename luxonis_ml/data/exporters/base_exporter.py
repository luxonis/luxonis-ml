from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from luxonis_ml.data.exporters.export_utils import PreparedLDF


class BaseExporter(ABC):
    def __init__(self, dataset_identifier: str):
        self.dataset_identifier = dataset_identifier

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
    def save(
        self,
        transformed_data: dict[str, list[dict[str, Any]]],
        prepared_ldf: PreparedLDF,
        output_path: Path,
        max_partition_size_gb: float | None,
        zip_output: bool,
    ) -> Path | list[Path]:
        """Write transformed data to disk, handle partitioning and
        zipping."""
        raise NotImplementedError
