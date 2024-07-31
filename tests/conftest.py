import platform
import shutil
import sys
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


@pytest.fixture(scope="session")
def python_version():
    version = sys.version_info
    formatted_version = f"{version.major}{version.minor}"
    return formatted_version


@pytest.fixture(scope="session")
def platform_name():
    os_name = platform.system().lower()
    if "darwin" in os_name:
        return "mac"
    elif "linux" in os_name:
        return "lin"
    elif "windows" in os_name:
        return "win"
    else:
        raise ValueError(f"Unsupported operating system: {os_name}")
