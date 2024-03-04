from pathlib import Path
from typing import Literal, Optional

from pydantic import model_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Environ", "environ"]


class Environ(BaseSettings):
    """A L{BaseSettings} subclass for storing environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_ENDPOINT_URL: Optional[str] = None

    MLFLOW_CLOUDFLARE_ID: Optional[str] = None
    MLFLOW_CLOUDFLARE_SECRET: Optional[str] = None
    MLFLOW_S3_BUCKET: Optional[str] = None
    MLFLOW_S3_ENDPOINT_URL: Optional[str] = None
    MLFLOW_TRACKING_URI: Optional[str] = None

    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: Optional[str] = None
    POSTGRES_DB: Optional[str] = None

    LUXONISML_BUCKET: Optional[str] = None
    LUXONISML_BASE_PATH: str = str(Path.home() / "luxonis_ml")
    LUXONISML_TEAM_ID: str = "offline"
    LUXONISML_TEAM_NAME: str = "offline"

    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @model_serializer(when_used="always", mode="plain", return_type=str)
    def _serialize_environ(self) -> str:
        return "[Content hidden ...]"


environ = Environ()
