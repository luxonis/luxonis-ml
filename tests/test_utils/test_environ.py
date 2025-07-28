from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import SecretStr

from luxonis_ml.utils.environ import Environ


@pytest.fixture
def dotenv_file(tempdir: Path) -> Generator[Path]:
    path = tempdir / "dotenv"
    path.write_text("POSTGRES_USER=test\nPOSTGRES_PASSWORD=pass\n")
    yield path
    path.unlink(missing_ok=True)


def test_environ(dotenv_file: Path):
    environ = Environ(
        POSTGRES_USER="user",
        _env_file=dotenv_file,  # type: ignore
    )
    assert environ.POSTGRES_USER == "user"
    assert isinstance(environ.POSTGRES_PASSWORD, SecretStr)
    assert environ.POSTGRES_PASSWORD.get_secret_value() == "pass"

    assert environ.model_dump() == {}
