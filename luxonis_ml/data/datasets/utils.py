import shutil
from pathlib import Path
from types import ModuleType
from typing import Literal

import polars as pl
from typing_extensions import overload

from luxonis_ml.typing import PathType, PosixPathType
from luxonis_ml.utils.filesystem import LuxonisFileSystem


@overload
def get_file(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_path: PathType,
    mlflow_instance: ModuleType | None = ...,
    default: None = ...,
) -> Path | None: ...


@overload
def get_file(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_path: PathType,
    mlflow_instance: ModuleType | None = ...,
    default: PathType = ...,
) -> Path: ...


def get_file(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_path: PathType,
    mlflow_instance: ModuleType | None = None,
    default: PathType | None = None,
) -> Path | None:
    try:
        return fs.get_file(remote_path, local_path, mlflow_instance)
    except shutil.SameFileError:
        return Path(local_path, Path(remote_path).name)
    except Exception:
        return Path(default) if default is not None else None


@overload
def find_filepath_uuid(
    filepath: PathType,
    index: pl.DataFrame | None,
    *,
    raise_on_missing: Literal[False] = ...,
) -> str | None: ...


@overload
def find_filepath_uuid(
    filepath: PathType,
    index: pl.DataFrame | None,
    *,
    raise_on_missing: Literal[True] = ...,
) -> str: ...


def find_filepath_uuid(
    filepath: PathType,
    index: pl.DataFrame | None,
    *,
    raise_on_missing: bool = False,
) -> str | None:
    """Find the UUID assigned to an original file path.

    Args:
        filepath: Original file path to look up.
        index: Optional dataframe containing ``original_filepath`` and
            ``uuid`` columns.
        raise_on_missing: Whether to raise when the path is not in the
            index.

    Returns:
        Matching UUID, or ``None`` when ``index`` is ``None`` or the path
        is missing and ``raise_on_missing`` is ``False``.

    Raises:
        ValueError: If ``raise_on_missing`` is ``True`` and the path is
            not in the index.

    Examples:
        >>> import polars as pl
        >>> path = Path("image.jpg").absolute().resolve()
        >>> index = pl.DataFrame({
        ...     "original_filepath": [str(path)],
        ...     "uuid": ["abc"],
        ... })
        >>> find_filepath_uuid(path, index)
        'abc'
        >>> find_filepath_uuid("missing.jpg", index, raise_on_missing=True)
        Traceback (most recent call last):
        ...
        ValueError: File ...missing.jpg not found in index

    """
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
def find_filepath_group_id(
    filepath: PathType,
    index: pl.DataFrame | None,
    *,
    raise_on_missing: Literal[False] = ...,
) -> str | None: ...


@overload
def find_filepath_group_id(
    filepath: PathType,
    index: pl.DataFrame | None,
    *,
    raise_on_missing: Literal[True] = ...,
) -> str: ...


def find_filepath_group_id(
    filepath: PathType,
    index: pl.DataFrame | None,
    *,
    raise_on_missing: bool = False,
) -> str | None:
    """Find the group ID assigned to an original file path.

    Args:
        filepath: Original file path to look up.
        index: Optional dataframe containing ``original_filepath`` and
            ``group_id`` columns.
        raise_on_missing: Whether to raise when the path is not in the
            index.

    Returns:
        Matching group ID, or ``None`` when ``index`` is ``None`` or the
        path is missing and ``raise_on_missing`` is ``False``.

    Raises:
        ValueError: If ``raise_on_missing`` is ``True`` and the path is
            not in the index.

    Examples:
        >>> import polars as pl
        >>> path = Path("image.jpg").absolute().resolve()
        >>> index = pl.DataFrame({
        ...     "original_filepath": [str(path)],
        ...     "group_id": ["group-1"],
        ... })
        >>> find_filepath_group_id(path, index)
        'group-1'
        >>> find_filepath_group_id(None, None) is None
        True

    """
    if index is None:
        return None

    abs_path = str(Path(filepath).absolute().resolve())
    matched = index.filter(pl.col("original_filepath") == abs_path)

    if len(matched):
        return next(iter(matched.select("group_id")))[0]
    if raise_on_missing:
        raise ValueError(f"File {abs_path} not found in index")
    return None


@overload
def get_dir(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_dir: PathType,
    mlflow_instance: ModuleType | None = ...,
    *,
    default: None = None,
) -> Path | None: ...


@overload
def get_dir(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_dir: PathType,
    mlflow_instance: ModuleType | None = ...,
    *,
    default: Path = ...,
) -> Path: ...


def get_dir(
    fs: LuxonisFileSystem,
    remote_path: PosixPathType,
    local_dir: PathType,
    mlflow_instance: ModuleType | None = None,
    *,
    default: PathType | None = None,
) -> Path | None:
    try:
        return fs.get_dir(remote_path, local_dir, mlflow_instance)
    except shutil.SameFileError:
        return Path(local_dir, Path(remote_path).name)
    except Exception:
        return Path(default) if default is not None else None
