from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import NonNegativeInt, SecretStr, model_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict

from luxonis_ml.typing import Params

__all__ = ["Environ", "environ"]


class Environ(BaseSettings):
    """Environment-backed configuration for Luxonis ML.

    Values are read from ``.env`` and the process environment. Secret
    values are represented as `pydantic.SecretStr`_ so they are not exposed by
    default string conversion.

    Attributes:
        AWS_ACCESS_KEY_ID: AWS access key used by S3-compatible storage.
        AWS_SECRET_ACCESS_KEY: AWS secret key used by S3-compatible
            storage.
        AWS_S3_ENDPOINT_URL: Optional custom S3 endpoint.
        MLFLOW_CLOUDFLARE_ID: Optional Cloudflare access client ID for
            MLflow.
        MLFLOW_CLOUDFLARE_SECRET: Optional Cloudflare access client
            secret for MLflow.
        MLFLOW_S3_BUCKET: Optional S3 bucket used by MLflow artifacts.
        MLFLOW_S3_ENDPOINT_URL: Optional S3 endpoint used by MLflow
            artifacts.
        MLFLOW_TRACKING_URI: Optional MLflow tracking URI.
        POSTGRES_USER: Optional PostgreSQL user.
        POSTGRES_PASSWORD: Optional PostgreSQL password.
        POSTGRES_HOST: Optional PostgreSQL host.
        POSTGRES_PORT: Optional PostgreSQL port. Must be
            non-negative when provided.
        POSTGRES_DB: Optional PostgreSQL database name.
        LUXONISML_BUCKET: Optional cloud bucket for datasets.
        LUXONISML_BASE_PATH: Local base path for offline datasets and
            cache files.
        LUXONISML_TEAM_ID: Team identifier used by dataset storage.
        LUXONISML_DISABLE_SETUP_LOGGING: Whether automatic logging setup
            is disabled.
        ROBOFLOW_API_KEY: Optional Roboflow API key used by
            `LuxonisParser`.
        GOOGLE_APPLICATION_CREDENTIALS: Optional Google credentials path.
        LOG_LEVEL: Logging level. One of ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``, or ``"CRITICAL"``.

    .. _pydantic.SecretStr:
        https://pydantic.dev/docs/validation/2.0/usage/types/secrets/

    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    AWS_ACCESS_KEY_ID: SecretStr | None = None
    AWS_SECRET_ACCESS_KEY: SecretStr | None = None
    AWS_S3_ENDPOINT_URL: str | None = None

    MLFLOW_CLOUDFLARE_ID: str | None = None
    MLFLOW_CLOUDFLARE_SECRET: SecretStr | None = None
    MLFLOW_S3_BUCKET: str | None = None
    MLFLOW_S3_ENDPOINT_URL: str | None = None
    MLFLOW_TRACKING_URI: str | None = None

    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: NonNegativeInt | None = None
    POSTGRES_DB: str | None = None

    LUXONISML_BUCKET: str | None = None
    LUXONISML_BASE_PATH: Path = Path.home() / "luxonis_ml"
    LUXONISML_TEAM_ID: str = "offline"
    LUXONISML_DISABLE_SETUP_LOGGING: bool = False

    ROBOFLOW_API_KEY: SecretStr | None = None
    ULTRALYTICS_API_KEY: SecretStr | None = None

    GOOGLE_APPLICATION_CREDENTIALS: SecretStr | None = None

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )

    @model_serializer(when_used="always", mode="plain")
    def _serialize_environ(self) -> Params:
        return {}


@lru_cache(maxsize=1)
def _load_environ() -> Environ:
    """Return a cached Environ instance, reading .env and os.environ
    once.
    """
    return Environ()


class _EnvironProxy:
    def __getattr__(self, name: str) -> Any:
        _load_environ.cache_clear()
        real = _load_environ()
        return getattr(real, name)

    def __repr__(self) -> str:
        return "<EnvironProxy loading from .env>"


_proxy = _EnvironProxy()
environ: Environ = cast(Environ, _proxy)
