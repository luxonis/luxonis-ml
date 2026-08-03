"""Packaging coverage for the lightweight LDF extra."""

from pathlib import Path


def test_ldf_extra_declares_its_imports() -> None:
    requirements = (
        Path(__file__).parents[2] / "luxonis_ml" / "ldf" / "requirements.txt"
    ).read_text()

    # `numpy` and `pycocotools` are imported at module level, the decoders
    # lazily from the mask and polyline validators.
    for package in ("numpy", "pycocotools", "opencv-python", "pillow"):
        assert package in requirements
