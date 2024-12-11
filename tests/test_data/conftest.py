import pytest
from _pytest.fixtures import SubRequest
from pytest import Function, Metafunc, Parser

from luxonis_ml.data import BucketStorage


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


@pytest.fixture(scope="function")
def dataset_name(
    request: SubRequest, platform_name: str, python_version: str
) -> str:
    node = request.node
    if isinstance(node, Function):  # pragma: no cover
        prefix = node.function.__name__
    else:
        prefix = node.name
    return f"{prefix}-{platform_name}-{python_version}"
