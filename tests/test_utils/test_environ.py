from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import SecretStr

from luxonis_ml.utils.environ import Environ


@pytest.fixture
def dotenv_file(tempdir: Path) -> Generator[Path]:
    path = tempdir / "dotenv"
    path.write_text(
        "AWS_ACCESS_KEY_ID=example_access_key\n"
        "AWS_SECRET_ACCESS_KEY=example_secret_key\n"
        "AWS_S3_ENDPOINT_URL=http://example.com\n"
    )
    yield path
    path.unlink(missing_ok=True)


def test_environ(dotenv_file: Path, monkeypatch: pytest.MonkeyPatch):
    environ = Environ(
        POSTGRES_USER="user",
        _env_file=dotenv_file,  # type: ignore
    )
    assert environ.POSTGRES_USER == "user"
    assert isinstance(environ.AWS_ACCESS_KEY_ID, SecretStr)
    assert isinstance(environ.AWS_SECRET_ACCESS_KEY, SecretStr)
    assert (
        environ.AWS_SECRET_ACCESS_KEY.get_secret_value()
        == "example_secret_key"
    )
    assert environ.AWS_S3_ENDPOINT_URL == "http://example.com"
    assert environ.AWS_ACCESS_KEY_ID.get_secret_value() == "example_access_key"

    assert environ.model_dump() == {}
