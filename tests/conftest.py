import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from _pytest.fixtures import SubRequest
from pytest import FixtureRequest, Function, Metafunc, Parser

from luxonis_ml.data import BucketStorage
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
def platform_name():  # pragma: no cover
    os_name = platform.system().lower()
    if "darwin" in os_name:
        return "mac"
    elif "linux" in os_name:
        return "lin"
    elif "windows" in os_name:
        return "win"
    else:
        raise ValueError(f"Unsupported operating system: {os_name}")


@pytest.fixture(scope="function")
def dataset_name(
    request: SubRequest, platform_name: str, python_version: str
) -> str:
    node = request.node
    if isinstance(node, Function):
        prefix = node.function.__name__
    else:  # pragma: no cover
        prefix = node.name
    return f"{prefix}-{platform_name}-{python_version}"


@pytest.fixture(scope="session")
def height() -> int:
    return 480


@pytest.fixture(scope="session")
def width() -> int:
    return 640


@pytest.fixture
def augmentation_data(
    height: int, width: int, request: FixtureRequest
) -> Dict[str, List[np.ndarray]]:
    batch_size: int = request.param
    return {
        "image": [
            np.random.rand(height, width, 3) * 255 for _ in range(batch_size)
        ],
        "bboxes": [
            np.array([[0.3 + i * 0.1, 0.3 + i * 0.1, 0.1, 0.1, 0, 0]])
            for i in range(batch_size)
        ],
        "keypoints": [
            np.array([[64.0 + i * 10, 150.0 + i * 10, 0.0, 0.0, 2.0]])
            for i in range(batch_size)
        ],
        "mask": [
            np.random.randint(0, 2, (height, width)) for _ in range(batch_size)
        ],
    }


@pytest.fixture(scope="function")
def tempdir():
    path = Path("tests/data/tempdir")
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree("tests/data/tempdir")


def pytest_addoption(parser: Parser):
    parser.addoption(
        "--only-local",
        action="store_true",
        default=False,
        help="Run tests only for local storage",
    )


def pytest_generate_tests(metafunc: Metafunc):
    if "bucket_storage" in metafunc.fixturenames:
        only_local = metafunc.config.getoption("--only-local")
        storage_options = (
            [BucketStorage.LOCAL]
            if only_local
            else [BucketStorage.LOCAL, BucketStorage.GCS]
        )
        metafunc.parametrize("bucket_storage", storage_options)
