import shutil
from pathlib import Path
from types import ModuleType
from typing import Literal, Optional

import polars as pl
from typing_extensions import overload

from luxonis_ml.typing import PathType, PosixPathType
from luxonis_ml.utils.filesystem import LuxonisFileSystem


@overload
def get_file(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_path: PathType,
    mlflow_instance: Optional[ModuleType] = ...,
    default: Literal[None] = ...,
) -> Optional[Path]: ...


@overload
def get_file(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_path: PathType,
    mlflow_instance: Optional[ModuleType] = ...,
    default: PathType = ...,
) -> Path: ...


def get_file(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_path: PathType,
    mlflow_instance: Optional[ModuleType] = None,
    default: Optional[PathType] = None,
) -> Optional[Path]:
    try:
        return fs.get_file(remote_path, local_path, mlflow_instance)
    except shutil.SameFileError:
        return Path(local_path, Path(remote_path).name)
    except Exception:
        return Path(default) if default is not None else None


@overload
def find_filepath_uuid(
    filepath: PathType,
    index: Optional[pl.DataFrame],
    *,
    raise_on_missing: Literal[False] = ...,
) -> Optional[str]: ...


@overload
def find_filepath_uuid(
    filepath: PathType,
    index: Optional[pl.DataFrame],
    *,
    raise_on_missing: Literal[True] = ...,
) -> str: ...


def find_filepath_uuid(
    filepath: PathType,
    index: Optional[pl.DataFrame],
    *,
    raise_on_missing: bool = False,
) -> Optional[str]:
    if index is None:
        return None

    abs_path = str(Path(filepath).absolute().resolve())
    matched = index.filter(pl.col("original_filepath") == abs_path)

    if len(matched):
        return next(iter(matched.select("uuid")))[0]
    if raise_on_missing:
        raise ValueError(f"File {abs_path} not found in index")
    return None


@overload
def get_dir(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_dir: PathType,
    mlflow_instance: Optional[ModuleType] = ...,
    *,
    default: Literal[None] = None,
) -> Optional[Path]: ...


@overload
def get_dir(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_dir: PathType,
    mlflow_instance: Optional[ModuleType] = ...,
    *,
    default: Path = ...,
) -> Path: ...


def get_dir(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_dir: PathType,
    mlflow_instance: Optional[ModuleType] = None,
    *,
    default: Optional[PathType] = None,
) -> Optional[Path]:
    try:
        return fs.get_dir(remote_path, local_dir, mlflow_instance)
    except shutil.SameFileError:
        return Path(local_dir, Path(remote_path).name)
    except Exception:
        return Path(default) if default is not None else None
