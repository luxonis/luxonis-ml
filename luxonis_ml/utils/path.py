from pathlib import Path, PurePosixPath, PureWindowsPath

from luxonis_ml.typing import PathType


def parse_manifest_path(value: PathType) -> Path:
    """Parse a path string from a dataset manifest on the current OS."""

    raw = str(value)
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path(PureWindowsPath(raw).as_posix())


def resolve_manifest_path(base_dir: Path, value: PathType) -> Path:
    """Resolve a manifest path relative to the directory that contains
    it."""

    raw = str(value)
    windows_path = PureWindowsPath(raw)
    if windows_path.is_absolute() and not Path(raw).is_absolute():
        return Path(windows_path.as_posix())

    path = parse_manifest_path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def path_to_posix(value: PathType) -> str:
    """Serialize a path with forward slashes for portable manifests."""

    raw = str(value)
    if "\\" in raw:
        return PureWindowsPath(raw).as_posix()
    return PurePosixPath(raw).as_posix()
