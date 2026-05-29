from pathlib import Path

from luxonis_ml.utils.path import (
    parse_manifest_path,
    path_to_posix,
    resolve_manifest_path,
)


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
