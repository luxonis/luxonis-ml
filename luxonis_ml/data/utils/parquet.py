import json
from pathlib import Path
from typing import TypedDict

import polars as pl
from typing_extensions import Self

from luxonis_ml.typing import Params, PathType


class ParquetRecord(TypedDict):
    """Single annotation row written to parquet.

    Attributes:
        file: Image or source file path.
        source_name: Source component name.
        task_name: Task name.
        class_name: Optional class name.
        instance_id: Optional instance identifier.
        task_type: Optional task type.
        annotation: Optional serialized annotation JSON.

    """

    file: str
    source_name: str
    task_name: str
    class_name: str | None
    instance_id: int | None
    task_type: str | None
    annotation: str | None
    metadata: Params | None


class ParquetFileManager:
    """Manage append-style writes across partitioned parquet files.

    Rows are buffered in memory and flushed to the current parquet file.
    A new file is selected every ``num_rows`` writes. Filenames are
    zero-padded numeric counters such as ``0000000000.parquet``.

    Attributes:
        dir: Directory containing parquet files.
        parquet_files: Existing parquet files discovered in ``dir``.
        num_rows: Maximum rows written to one parquet file.
        num: Current parquet file index.
        current_file: Path to the current parquet file.
        buffer: In-memory column buffer for the current file.
        row_count: Number of rows currently buffered or loaded from the
            current file.

    """

    def __init__(self, directory: PathType, num_rows: int = 100_000) -> None:
        """Manage writing rows into partitioned parquet files.

        Args:
            directory: Local directory where parquet files are stored.
            num_rows: Maximum rows per parquet file before a new file is
                created.

        Example:
            >>> record = {
            ...     "file": "image.jpg",
            ...     "source_name": "image",
            ...     "task_name": "detection",
            ...     "class_name": "car",
            ...     "instance_id": 0,
            ...     "task_type": "boundingbox",
            ...     "annotation": "{}",
            ... }
            >>> manager = ParquetFileManager("/tmp/ldf-parquet-example", num_rows=2)
            >>> manager.num_rows
            2

        """
        self.dir = Path(directory)
        self.parquet_files = list(self.dir.glob("*.parquet"))
        self.num_rows = num_rows

        self.num = self._find_num() if self.parquet_files else 0
        self.current_file = self._generate_filename(self.num)

        self._read()

    def _find_num(self) -> int:
        return max(int(file.stem) for file in self.parquet_files)

    def _generate_filename(self, num: int) -> Path:
        return self.dir / f"{num:010d}.parquet"

    def _read(self) -> None:
        if self.current_file.exists():
            df = pl.read_parquet(self.current_file)
            self.buffer = df.to_dict(as_series=False)
            self.row_count = len(df)
        else:
            self.row_count = 0
            self.buffer = {}

    def _initialize_data(self, data: ParquetRecord) -> None:
        for key in data:
            self.buffer[key] = []
        self.buffer["uuid"] = []
        self.buffer["group_id"] = []

    def write(self, uuid: str, data: ParquetRecord, group_id: str) -> None:
        """Write a row to the current parquet file.

        Args:
            uuid: Unique row identifier.
            data: Annotation row data.
            group_id: Unique identifier for the sample group.

        """
        if not self.buffer:
            self._initialize_data(data)

        for key in data:
            self.buffer[key].append(data[key])

        self.buffer["uuid"].append(uuid)
        self.buffer["group_id"].append(group_id)

        self.row_count += 1
        if self.row_count % self.num_rows == 0:
            self._flush()
            self.num += 1
            self.current_file = self._generate_filename(self.num)
            self._read()

    def remove_duplicate_uuids(self, overwrite_uuids: set[str]) -> None:
        self._flush()

        for parquet_file in self.dir.glob("*.parquet"):
            if parquet_file.is_file():
                df = pl.read_parquet(parquet_file)
                df = df.filter(~pl.col("uuid").is_in(list(overwrite_uuids)))
                parquet_file.unlink()
                df.write_parquet(parquet_file)

        self._read()

    def _flush(self) -> None:
        """Write buffered data to parquet."""
        if self.buffer:
            buffer = self.buffer.copy()
            if "metadata" in buffer:
                buffer["metadata"] = [
                    json.dumps(value) if isinstance(value, dict) else value
                    for value in buffer["metadata"]
                ]
            df = pl.DataFrame(buffer)
            df.write_parquet(self.current_file)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self._flush()
