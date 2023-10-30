import os, io
import numpy as np
from typing import Tuple, Optional, Dict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetFileManager:
    def __init__(
        self,
        directory: str,
        file_size_mb: int = 256,
        row_check: int = 100000,
        # num_rows: Optional[int] = None
    ) -> None:
        self.dir = directory
        self.files = os.listdir(self.dir)
        self.file_size = file_size_mb
        self.row_check = row_check

        if len(self.files):
            self.num = self._find_num()
            self.current_file = self._get_current_parquet_file()
        else:
            self.num = 0
            new_filename, self.current_file = self._generate_filename(self.num)
            self.files = [new_filename]

        self._read()

    def _get_current_parquet_file(self) -> str:
        """Finds the best parquet file to edit based on the file size and most last write time"""

        path = self._generate_filename(self.num)[1]
        current_size = os.path.getsize(path) / (1024 * 1024)
        print(current_size)
        if current_size < self.file_size:
            return path
        else:
            for file in self.files:
                path = os.path.join(self.dir, file) / (1024 * 1024)
                sizes = []
                # times = []
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    if size < self.file_size:
                        sizes.append(size)
                        # times.append(os.path.getmtime(path))
            if len(sizes):
                idx = np.argmin(sizes)
                filename = self.files[idx]
                path = os.path.join(self.dir, filename)
                return path
            else:
                self.num += 1
                return self._generate_filename(self.num)[1]

    def _find_num(self) -> int:
        nums = [int(filename.split(".parquet")[0]) for filename in self.files]
        return max(nums)

    def _generate_filename(self, num: int) -> Tuple[str, str]:
        filename = f"{str(num).zfill(10)}.parquet"
        path = os.path.join(self.dir, filename)
        return filename, path

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

    def _estimate_file_size(self, df: pd.DataFrame) -> float:
        with io.BytesIO() as buffer:
            df.to_parquet(buffer)
            return buffer.tell() / (1024 * 1024)

    def write(self, add_data: Dict) -> None:
        """Writes a row to the current working parquet file"""
        if len(self.data) == 0:
            self._initialize_data(add_data)
        for key in add_data:
            if key not in self.data:
                raise Exception(f"Key {key} Not Found")
            self.data[key].append(add_data[key])
            self.row_count += 1

            if self.row_count % self.row_check == 0:
                df = pd.DataFrame(self.data)
                estimated_size = self._estimate_file_size(df)
                if estimated_size > self.file_size:
                    self.close()
                    self.num += 1
                    self.current_file = self._generate_filename(self.num)[1]
                    self._read()

    def close(self) -> None:
        """Ensure all data is written to parquet"""

        df = pd.DataFrame(self.data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.current_file)
