from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import model_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict

from luxonis_ml.typing import Params

__all__ = ["Environ", "environ"]


class Environ(BaseSettings):
    """A L{BaseSettings} subclass for storing environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_S3_ENDPOINT_URL: str | None = None

    MLFLOW_CLOUDFLARE_ID: str | None = None
    MLFLOW_CLOUDFLARE_SECRET: str | None = None
    MLFLOW_S3_BUCKET: str | None = None
    MLFLOW_S3_ENDPOINT_URL: str | None = None
    MLFLOW_TRACKING_URI: str | None = None

    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: str | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: str | None = None
    POSTGRES_DB: str | None = None

    LUXONISML_BUCKET: str | None = None
    LUXONISML_BASE_PATH: Path = Path.home() / "luxonis_ml"
    LUXONISML_TEAM_ID: str = "offline"
    LUXONISML_DISABLE_SETUP_LOGGING: bool = False

    ROBOFLOW_API_KEY: str | None = None

    GOOGLE_APPLICATION_CREDENTIALS: str | None = None

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )

    @model_serializer(when_used="always", mode="plain")
    def _serialize_environ(self) -> Params:
        return {}


@lru_cache(maxsize=1)
def _load_environ() -> Environ:
    """Return a cached Environ instance, reading .env and os.environ
    once."""
    return Environ()


class _EnvironProxy:
    def __getattr__(self, name: str) -> Any:
        _load_environ.cache_clear()
        real = _load_environ()
        return getattr(real, name)

    def __repr__(self) -> str:
        return "<EnvironProxy loading from .env>"


environ = _EnvironProxy()
