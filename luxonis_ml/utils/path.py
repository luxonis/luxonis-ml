from pathlib import Path, PurePosixPath, PureWindowsPath

from luxonis_ml.typing import PathType


def parse_manifest_path(value: PathType) -> Path:
    r"""Parse a path string from a dataset manifest on the current OS.

    Args:
        value: Path value read from a manifest.

    Returns:
        Parsed path. Relative Windows paths are converted to POSIX-style
        components before constructing the current-platform ``Path``.

    Examples:
        >>> parse_manifest_path("images/cat.jpg").as_posix()
        'images/cat.jpg'
        >>> parse_manifest_path(r"images\\cat.jpg").as_posix()
        'images/cat.jpg'

    """

    raw = str(value)
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path(PureWindowsPath(raw).as_posix())


def resolve_manifest_path(base_dir: Path, value: PathType) -> Path:
    """Resolve a manifest path relative to the directory that contains
    it.

    Args:
        base_dir: Directory relative paths are resolved against.
        value: Path value read from a manifest.

    Returns:
        Absolute resolved path.

    Examples:
        >>> base = Path("/dataset")
        >>> resolve_manifest_path(base, "images/cat.jpg").as_posix()
        '/dataset/images/cat.jpg'
        >>> resolve_manifest_path(base, "/tmp/cat.jpg").as_posix()
        '/tmp/cat.jpg'

    """

    raw = str(value)
    windows_path = PureWindowsPath(raw)
    if windows_path.is_absolute() and not Path(raw).is_absolute():
        return Path(windows_path.as_posix())

    path = parse_manifest_path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def path_to_posix(value: PathType) -> str:
    r"""Serialize a path with forward slashes for portable manifests.

    Args:
        value: Path value to serialize.

    Returns:
        POSIX-style path string.

    Examples:
        >>> path_to_posix(Path("images") / "cat.jpg")
        'images/cat.jpg'
        >>> path_to_posix(r"images\\cat.jpg")
        'images/cat.jpg'

    """

    raw = str(value)
    if "\\" in raw:
        return PureWindowsPath(raw).as_posix()
    return PurePosixPath(raw).as_posix()
