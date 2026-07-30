import string
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from luxonis_ml.utils.path import (
    parse_manifest_path,
    path_to_posix,
    resolve_manifest_path,
)

# Segments exclude "." and ".." and anything pathlib treats differently
# across platforms, so the parts always survive a roundtrip unchanged.
segments = st.text(
    alphabet=string.ascii_letters + string.digits + "_-",
    min_size=1,
    max_size=6,
)
relative_paths = st.lists(segments, min_size=1, max_size=4)


@given(parts=relative_paths)
def test_separators_parse_identically(parts: list[str]):
    posix = "/".join(parts)
    windows = "\\".join(parts)

    assert parse_manifest_path(windows) == parse_manifest_path(posix)
    assert parse_manifest_path(windows) == Path(*parts)
    assert path_to_posix(windows) == path_to_posix(posix) == posix


@given(parts=relative_paths)
def test_parse_posix_roundtrip(parts: list[str]):
    posix = "/".join(parts)

    assert path_to_posix(parse_manifest_path(posix)) == posix
    assert parse_manifest_path(path_to_posix(posix)) == Path(*parts)


@given(parts=relative_paths)
def test_relative_paths_stay_under_base_dir(parts: list[str]):
    base_dir = Path.cwd().resolve()

    resolved = resolve_manifest_path(base_dir, "\\".join(parts))

    assert resolved.is_absolute()
    assert resolved.relative_to(base_dir) == Path(*parts)


def test_path_to_posix_normalizes_windows_separators():
    assert path_to_posix("images\\train\\0.png") == "images/train/0.png"
    assert path_to_posix(Path("images/val/1.png")) == "images/val/1.png"


def test_parse_and_resolve_manifest_path(tempdir: Path):
    base_dir = tempdir / "dataset"
    base_dir.mkdir()

    parsed = parse_manifest_path("nested\\images\\0.png")
    resolved = resolve_manifest_path(base_dir, "nested\\images\\0.png")

    assert parsed == Path("nested/images/0.png")
    assert resolved == (base_dir / "nested" / "images" / "0.png").resolve()


def test_resolve_manifest_path_preserves_windows_absolute_path(tempdir: Path):
    base_dir = tempdir / "dataset"
    base_dir.mkdir()

    resolved = resolve_manifest_path(base_dir, r"C:\data\img.png")

    assert str(resolved).replace("\\", "/") == "C:/data/img.png"
