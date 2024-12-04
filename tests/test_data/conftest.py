from pytest import Metafunc, Parser

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
            else [BucketStorage.LOCAL, BucketStorage.S3, BucketStorage.GCS]
        )
        metafunc.parametrize("bucket_storage", storage_options)
