import shutil
from pathlib import Path

import pytest

from luxonis_ml.utils import setup_logging
from luxonis_ml.utils.environ import environ

setup_logging(use_rich=True, rich_print=True, configure_warnings=True)


@pytest.fixture(autouse=True, scope="module")
def set_paths():
    environ.LUXONISML_BASE_PATH = Path.cwd() / "tests/data/luxonisml_base_path"
    if environ.LUXONISML_BASE_PATH.exists():
        shutil.rmtree(environ.LUXONISML_BASE_PATH)
