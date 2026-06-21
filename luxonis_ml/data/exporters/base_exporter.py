from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from luxonis_ml.data.exporters.exporter_utils import PreparedLDF


class BaseExporter(ABC):
    """Base class for dataset exporters.

    Attributes:
        dataset_identifier: Name or identifier used for exported paths.
        output_path: Directory where the export is written.
        image_indices: Per-image export indices used by concrete
            exporters.
        max_partition_size_gb: Optional maximum partition size in GiB.
        max_partition_size: Optional maximum partition size in bytes.
        part: Current partition index, or ``None`` when partitioning is
            disabled.
        current_size: Current partition size in bytes.

    """

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
    def export(self, prepared_ldf: "PreparedLDF") -> None:
        """Convert the prepared dataset into the exporter's format.

        Args:
            prepared_ldf: Dataset data prepared for export.

        Raises:
            NotImplementedError: Always raised by the abstract base
                implementation.

        """
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

    @abstractmethod
    def supported_ann_types(self) -> list[str]:
        """Return task types supported by this exporter.

        Returns:
            Supported annotation task types.

        Raises:
            NotImplementedError: Always raised by the abstract base
                implementation.

        """
        raise NotImplementedError
