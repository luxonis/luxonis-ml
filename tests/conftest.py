import builtins
import platform
import random
import shutil
import sys
import time
from collections.abc import Generator
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest
from _pytest.fixtures import SubRequest
from rich import print as rich_print

from luxonis_ml.data import BucketStorage, LuxonisDataset
from luxonis_ml.typing import Params
from luxonis_ml.utils.environ import environ

CREATED_DATASETS = []


@pytest.fixture(autouse=True, scope="session")
def setup():
    builtins.print = rich_print

    randint = random.randint(0, 100000)
    base = Path.cwd() / f"tests/data/luxonisml_base_path/{randint}"
    environ.LUXONISML_BASE_PATH = base
    if base.exists():  # pragma: no cover
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    yield

    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def randint() -> int:
    # Use a fresh, time-seeded RNG so pytest's global seed doesn't influence this value.
    rng = random.Random()
    rng.seed(time.time())
    return rng.randint(0, 100_000)


@pytest.fixture(autouse=True, scope="session")
def fix_seed(worker_id: str):
    np.random.seed(hash(worker_id) % 2**32)
    random.seed(hash(worker_id) % 2**32)


@pytest.fixture(scope="session")
def python_version():
    version = sys.version_info
    return f"{version.major}{version.minor}"


@pytest.fixture(scope="session")
def platform_name():  # pragma: no cover
    os_name = platform.system().lower()
    if "darwin" in os_name:
        return "mac"
    if "linux" in os_name:
        return "lin"
    if "windows" in os_name:
        return "win"
    raise ValueError(f"Unsupported operating system: {os_name}")


@pytest.fixture
def dataset_name(
    request: SubRequest, randint: int
) -> Generator[str, None, None]:
    name = f"{get_caller_name(request)}_{randint}"
    yield name
    with suppress(Exception):
        LuxonisDataset(name, bucket_storage=BucketStorage.GCS).delete_dataset(
            delete_remote=True,
            delete_local=True,
        )


@pytest.fixture(scope="session")
def height() -> int:
    return 480


@pytest.fixture(scope="session")
def width() -> int:
    return 640


@pytest.fixture
def augmentation_data(
    height: int, width: int, request: pytest.FixtureRequest
) -> dict[str, list[np.ndarray]]:
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
            np.random.randint(0, 2, (height, width, 1))
            for _ in range(batch_size)
        ],
    }


@pytest.fixture(scope="session")
def augmentation_config() -> list[Params]:
    return [
        {
            "name": "Mosaic4",
            "params": {"out_width": 416, "out_height": 416, "p": 1.0},
        },
        {"name": "MixUp", "params": {"p": 1.0}},
        {"name": "Defocus", "params": {"p": 1.0}},
        {"name": "Sharpen", "params": {"p": 1.0}},
        {"name": "Flip", "params": {"p": 1.0}},
        {"name": "RandomRotate90", "params": {"p": 1.0}},
    ]


@pytest.fixture(scope="session")
def base_tempdir(worker_id: str):
    path = Path("tests", "data", "tempdir", worker_id)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def tempdir(base_tempdir: Path, randint: int) -> Generator[Path, None, None]:
    t = time.time()
    unique_id = randint
    while True:
        path = base_tempdir / str(unique_id)
        if not path.exists():
            break
        if time.time() - t > 5:  # pragma: no cover
            raise TimeoutError(
                "Could not create a unique tempdir. Something is wrong."
            )
        # regenerate a new random suffix
        unique_id = random.randint(0, 100_000)

    path.mkdir(exist_ok=True)

    yield path

    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def storage_url() -> str:
    return "gs://luxonis-test-bucket/luxonis-ml-test-data/"


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--only-local",
        action="store_true",
        default=False,
        help="Run tests only for local storage",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc):
    if "bucket_storage" in metafunc.fixturenames:
        only_local = metafunc.config.getoption("--only-local")
        storage_options = (
            [BucketStorage.LOCAL]
            if only_local
            else [BucketStorage.LOCAL, BucketStorage.GCS]
        )
        metafunc.parametrize("bucket_storage", storage_options)


def get_caller_name(request: SubRequest) -> str:  # pragma: no cover
    node = request.node
    if isinstance(node, pytest.Function):
        return node.function.__name__
    return node.name
