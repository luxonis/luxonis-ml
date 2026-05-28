from pathlib import Path, PurePosixPath, PureWindowsPath

from luxonis_ml.typing import PathType


def parse_manifest_path(value: PathType) -> Path:
    """Parse a path string from a dataset manifest on the current OS."""

    path = Path(value)
    if path.is_absolute():
        return path
    return Path(PureWindowsPath(str(value)))


def resolve_manifest_path(base_dir: Path, value: PathType) -> Path:
    """Resolve a manifest path relative to the directory that contains
    it."""

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
