import os
import numpy as np
from typing import Tuple, Optional, Dict


class ParquetFileManager:
    def __init__(
        self,
        directory: str,
        file_size_mb: int = 256,
        row_check: int = 10000,
        # num_rows: Optional[int] = None
    ) -> None:
        self.dir = directory
        self.files = os.listdir(self.dir)
        self.file_size = file_size_mb
        self.row_check = row_check
        self.data = {}
        self.df = None
        if len(self.files):
            self.num = self._find_num()
            self.current_file = self._get_current_parquet_file()
        else:
            self.num = 0
            new_filename, self.current_file = self._generate_filename(self.num)
            self.files = [new_filename]

        # TODO: call _read() to load self.data

    def _get_current_parquet_file(self) -> str:
        """Finds the best parquet file to edit based on the file size and most last write time"""

        path = self._generate_filename(self.num)[1]
        current_size = os.path.getsize(path)
        print(current_size)
        if current_size < self.file_size:
            return path
        else:
            for file in self.files:
                path = os.join(self.dir, file)
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
                # New file
                self.num += 1
                return self._generate_filename(self.num)[1]

    def _find_num(self) -> int:
        nums = [int(filename.split(".parquet")) for filename in self.files]
        return max(nums)

    def _generate_filename(self, num: int) -> Tuple[str, str]:
        filename = f"{str(num).zfill(10)}.parquet"
        path = os.path.join(self.dir, filename)
        return filename, path

    def _read(self) -> Dict:
        pass

    def _initialize_data(self, data: Dict) -> None:
        for key in data:
            self.data[key] = []

    def write(self, data: Dict) -> None:
        """Writes a row to the current working parquet file"""
        if len(data) == 0:
            self._initialize_data()
        for key in data:
            if key not in self.data:
                raise Exception(f"Key {key} Not Found")

    def close(self) -> None:
        """Ensure all data is written to parquet"""

        pass
