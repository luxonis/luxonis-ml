import os
from typing import Dict, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetFileManager:
    def __init__(
        self,
        directory: str,
        num_rows: int = 100000,
    ) -> None:
        """Manages the insertion of data into parquet files.

        @type directory: str
        @param directory: The local directory in which parquet files are stored.
        @type num_rows: int
        @param num_rows: The maximum number of rows permitted in a parquet file before
            another file is created.
        """

        self.dir = directory
        self.files = os.listdir(self.dir)
        self.num_rows = num_rows

        self.num = self._find_num() if len(self.files) else 0
        self.current_file = self._generate_filename(self.num)

        self._read()

    def _find_num(self) -> int:
        nums = [
            int(os.path.splitext(file)[0])
            for file in self.files
            if os.path.splitext(file)[1] == ".parquet"
        ]
        return max(nums)

    def _generate_filename(self, num: int) -> Tuple[str, str]:
        filename = f"{str(num).zfill(10)}.parquet"
        path = os.path.join(self.dir, filename)
        return path

    def _read(self) -> Dict:
        if os.path.exists(self.current_file):
            df = pd.read_parquet(self.current_file)
            self.data = df.to_dict()
            self.data = {k: list(v.values()) for k, v in self.data.items()}
            self.row_count = len(df)
        else:
            self.row_count = 0
            self.data = {}

    def _initialize_data(self, data: Dict) -> None:
        for key in data:
            self.data[key] = []

    def write(self, add_data: Dict) -> None:
        """Writes a row to the current working parquet file.

        @type add_data: Dict
        @param add_data: A dictionary representing annotations, mapping annotation types
            to values.
        """

        if len(self.data) == 0:
            self._initialize_data(add_data)
        for key in add_data:
            if key not in self.data:
                raise Exception(f"Key {key} Not Found")
            self.data[key].append(add_data[key])

        self.row_count += 1
        if self.row_count % self.num_rows == 0:
            self.close()
            self.num += 1
            self.current_file = self._generate_filename(self.num)
            self._read()

    def close(self) -> None:
        """Ensures all data is written to parquet."""

        df = pd.DataFrame(self.data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.current_file)
