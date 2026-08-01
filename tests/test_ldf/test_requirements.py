"""Packaging coverage for the lightweight LDF extra."""

from pathlib import Path


def test_ldf_extra_declares_segmentation_decoders() -> None:
    requirements = (
        Path(__file__).parents[2] / "luxonis_ml" / "ldf" / "requirements.txt"
    ).read_text()

    assert "opencv-python" in requirements
    assert "pillow" in requirements
