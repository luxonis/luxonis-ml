"""Runtime libraries bundled with an NN Archive model."""

from pathlib import PurePosixPath, PureWindowsPath

from pydantic import Field, field_validator

from luxonis_ml.typing import BaseModelExtraForbid


class RuntimeImplementation(BaseModelExtraForbid):
    """Implementation library for one SNPE runtime."""

    runtime: str = Field(min_length=1, description="SNPE runtime, e.g. 'dsp'.")
    path: str = Field(
        description="Archive path of the implementation library."
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        """Validate the implementation library's archive path."""
        return _validate_archive_path(path)


class SNPEUDOPackage(BaseModelExtraForbid):
    """Registration library and implementations for one SNPE UDO package."""

    registration: str = Field(
        description="Archive path of the registration library."
    )
    implementations: list[RuntimeImplementation] = Field(
        min_length=1,
        description="Implementation libraries required by the package.",
    )

    @field_validator("registration")
    @classmethod
    def validate_registration(cls, path: str) -> str:
        """Validate the registration library's archive path."""
        return _validate_archive_path(path)


class RuntimeDependencies(BaseModelExtraForbid):
    """Optional runtime dependencies of the model executable."""

    snpe_udo: list[SNPEUDOPackage] = Field(
        min_length=1, description="SNPE user-defined operation packages."
    )


def _validate_archive_path(path: str) -> str:
    """Keep a dependency path unambiguous and inside the archive root."""
    pure_path = PurePosixPath(path)
    if (
        not path
        or path == "."
        or "\\" in path
        or "\x00" in path
        or pure_path.is_absolute()
        or PureWindowsPath(path).drive
        or ".." in pure_path.parts
        or str(pure_path) != path
        or path == "config.json"
    ):
        raise ValueError(
            "Runtime dependency paths must be normalized, relative POSIX "
            "file paths inside the archive and cannot be config.json"
        )
    return path
