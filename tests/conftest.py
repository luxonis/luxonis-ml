import builtins
import platform
import random
import shutil
import sys
import time
import zipfile
from collections.abc import Generator
from contextlib import suppress
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from _pytest.fixtures import SubRequest
from rich import print as rich_print

from luxonis_ml.data import BucketStorage, LuxonisDataset
from luxonis_ml.typing import Params
from luxonis_ml.utils import LuxonisFileSystem
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


class CocoSplitConfig(Enum):
    ALL_SPLITS = "all_splits"  # train + validation + test
    TRAIN_VAL = "train_val"  # train + validation only
    TRAIN_TEST = "train_test"  # train + test only (no validation)
    TRAIN_ONLY = "train_only"  # train only


def _download_and_extract(url: str, dest: Path) -> Path:
    zip_path = LuxonisFileSystem.download(url, dest)
    extract_dir = zip_path.parent / zip_path.stem
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()
    return extract_dir


@pytest.fixture(scope="session")
def coco_2017_source(storage_url: str, base_tempdir: Path) -> Path:
    url = f"{storage_url.rstrip('/')}/coco-2017.zip"
    return _download_and_extract(url, base_tempdir / "coco_2017_source")


@pytest.fixture(scope="session")
def imagenet_sample_source(storage_url: str, base_tempdir: Path) -> Path:
    url = f"{storage_url.rstrip('/')}/imagenet-sample.zip"
    return _download_and_extract(url, base_tempdir / "imagenet_sample_source")


def _make_coco_variant(
    source: Path, dest_dir: Path, config: CocoSplitConfig
) -> Path:
    """Creates a coco-2017 directory with only the requested splits.

    Always copies the raw/ folder (needed for keypoint annotations) and
    info.json.
    """
    dest = dest_dir / "coco-2017"
    shutil.copytree(source, dest)

    if config is CocoSplitConfig.TRAIN_ONLY:
        shutil.rmtree(dest / "validation", ignore_errors=True)
        shutil.rmtree(dest / "test", ignore_errors=True)
    elif config is CocoSplitConfig.TRAIN_VAL:
        shutil.rmtree(dest / "test", ignore_errors=True)
    elif config is CocoSplitConfig.TRAIN_TEST:
        shutil.rmtree(dest / "validation", ignore_errors=True)

    return dest


@pytest.fixture
def coco_2017_all_splits(coco_2017_source: Path, tempdir: Path) -> Path:
    return _make_coco_variant(
        coco_2017_source, tempdir, CocoSplitConfig.ALL_SPLITS
    )


@pytest.fixture
def coco_2017_train_val(coco_2017_source: Path, tempdir: Path) -> Path:
    return _make_coco_variant(
        coco_2017_source, tempdir, CocoSplitConfig.TRAIN_VAL
    )


@pytest.fixture
def coco_2017_train_test(coco_2017_source: Path, tempdir: Path) -> Path:
    return _make_coco_variant(
        coco_2017_source, tempdir, CocoSplitConfig.TRAIN_TEST
    )


@pytest.fixture
def coco_2017_train_only(coco_2017_source: Path, tempdir: Path) -> Path:
    return _make_coco_variant(
        coco_2017_source, tempdir, CocoSplitConfig.TRAIN_ONLY
    )


@pytest.fixture
def coco_2017(
    request: pytest.FixtureRequest,
    coco_2017_source: Path,
    tempdir: Path,
) -> Path:
    """Parametrisable coco-2017 fixture.

    Use with ``@pytest.mark.parametrize("coco_2017", [...], indirect=True)``
    where the parameter values are :class:`CocoSplitConfig` members.
    """
    config: CocoSplitConfig = request.param
    return _make_coco_variant(coco_2017_source, tempdir, config)


@pytest.fixture
def imagenet_sample_dir(imagenet_sample_source: Path, tempdir: Path) -> Path:
    dest = tempdir / "imagenet-sample"
    shutil.copytree(imagenet_sample_source, dest)
    return dest
