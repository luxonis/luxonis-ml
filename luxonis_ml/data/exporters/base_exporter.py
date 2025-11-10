from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from luxonis_ml.data.exporters.exporter_utils import PreparedLDF


class BaseExporter(ABC):
    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
    ):
        self.dataset_identifier = dataset_identifier
        self.output_path = output_path
        self.image_indices = {}

        # attributes for intermediate saving
        self.max_partition_size_gb = max_partition_size_gb
        self.max_partition_size = (
            self.max_partition_size_gb * 1024**3
            if self.max_partition_size_gb
            else None
        )
        self.part = 0 if max_partition_size_gb else None
        self.current_size = 0

    @abstractmethod
    def transform(self, prepared_ldf: PreparedLDF) -> None:
        """Convert the prepared dataset into the exporter's format."""
        raise NotImplementedError

    @abstractmethod
    def _get_data_path(self, output_path: Path, split: str) -> Path:
        """Return the folder path to store data files for this split."""
        raise NotImplementedError

    @abstractmethod
    def _dump_annotations(
        self, annotations: dict, output_path: Path, part: int | None = None
    ) -> None:
        raise NotImplementedError
