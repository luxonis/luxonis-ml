from pathlib import Path

import pytest


@pytest.fixture
def image(tempdir: Path) -> Path:
    path = tempdir / "image.png"
    path.touch()
    return path
