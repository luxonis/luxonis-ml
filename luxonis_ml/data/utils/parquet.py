from pathlib import Path
from typing import TypedDict

import polars as pl

from luxonis_ml.typing import PathType


class ParquetRecord(TypedDict):
    file: str
    source_name: str
    task_name: str
    class_name: str | None
    instance_id: int | None
    task_type: str | None
    annotation: str | None


class ParquetFileManager:
    def __init__(self, directory: PathType, num_rows: int = 100_000) -> None:
        """Manages the insertion of data into parquet files.

        @type directory: str
        @param directory: The local directory in which parquet files are
            stored.
        @type num_rows: int
        @param num_rows: The maximum number of rows permitted in a
            parquet file before another file is created.
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

    def write(self, uuid: str, data: ParquetRecord) -> None:
        """Writes a row to the current working parquet file.

        @type data: Dict
        @param data: A dictionary representing annotations, mapping
            annotation types to values.
        """

        if not self.buffer:
            self._initialize_data(data)

        for key in data:
            self.buffer[key].append(data[key])

        self.buffer["uuid"].append(uuid)

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
        """Writes buffered data to parquet."""

        if self.buffer:
            df = pl.DataFrame(self.buffer)
            df.write_parquet(self.current_file)

    def __enter__(self) -> "ParquetFileManager":
        return self

    def __exit__(self, *_) -> None:
        self._flush()
