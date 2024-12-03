import pytest

from luxonis_ml.data import BucketStorage


def pytest_addoption(parser):
    parser.addoption(
        "--only-local",
        action="store_true",
        default=False,
        help="Run tests only for local storage",
    )


@pytest.fixture
def only_local(request):
    return request.config.getoption("--only-local")


@pytest.fixture
def storage_options(only_local: bool):
    if only_local:
        return [BucketStorage.LOCAL]
    return [BucketStorage.LOCAL, BucketStorage.GCS, BucketStorage.S3]


def pytest_generate_tests(metafunc):
    if "bucket_storage" in metafunc.fixturenames:
        only_local = metafunc.config.getoption("--only-local")
        storage_options = (
            [BucketStorage.LOCAL]
            if only_local
            else [BucketStorage.LOCAL, BucketStorage.S3, BucketStorage.GCS]
        )
        metafunc.parametrize("bucket_storage", storage_options)
