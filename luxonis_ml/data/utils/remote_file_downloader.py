import shutil
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

import requests
from filelock import FileLock
from PIL import Image, UnidentifiedImageError

from luxonis_ml.utils.filesystem import LuxonisFileSystem


class RemoteFileAPIClient(Protocol):
    def download(
        self, url: str, destination: Path, *, timeout: float
    ) -> None: ...


class RequestsRemoteFileAPIClient:
    def download(self, url: str, destination: Path, *, timeout: float) -> None:
        with (
            requests.get(
                url,
                headers={"User-Agent": "luxonis-ml"},
                stream=True,
                timeout=timeout,
            ) as response,
            open(destination, "wb") as output_file,
        ):
            response.raise_for_status()
            shutil.copyfileobj(response.raw, output_file)


class RemoteFileDownloader:
    def __init__(self, api_client: RemoteFileAPIClient | None = None) -> None:
        self.api_client = api_client or RequestsRemoteFileAPIClient()

    def download(
        self,
        url: str,
        destination: Path,
        *,
        timeout: float = 30.0,
        validate_image: bool = False,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock_path = destination.with_suffix(f"{destination.suffix}.lock")
        tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")

        with FileLock(str(lock_path)):
            if destination.exists():
                if validate_image:
                    self._validate_image_format(destination)
                return destination

            self._remove_temporary_file(tmp_path)

            scheme = urlsplit(url).scheme
            if scheme not in ALLOWED_REMOTE_URL_SCHEMES:
                raise ValueError(
                    f"Unsupported remote URL scheme '{scheme}'. "
                    f"Expected one of {sorted(ALLOWED_REMOTE_URL_SCHEMES)}."
                )

            try:
                self._download_to_temporary_path(
                    url,
                    tmp_path,
                    scheme=scheme,
                    timeout=timeout,
                )
                if validate_image:
                    self._validate_image_format(
                        tmp_path, destination=destination
                    )
                tmp_path.replace(destination)
            except Exception:
                self._remove_temporary_file(tmp_path)
                raise

        return destination

    def _download_to_temporary_path(
        self,
        url: str,
        destination: Path,
        *,
        scheme: str,
        timeout: float,
    ) -> None:
        if scheme in FS_REMOTE_URL_SCHEMES:
            downloaded_path = LuxonisFileSystem.download(url, destination)
            if downloaded_path != destination:
                downloaded_path.replace(destination)
            return

        if scheme == "file":
            shutil.copyfile(self._path_from_file_url(url), destination)
            return

        self.api_client.download(url, destination, timeout=timeout)

    @staticmethod
    def _path_from_file_url(url: str) -> Path:
        parsed = urlsplit(url)
        path = unquote(parsed.path)

        if parsed.netloc and parsed.netloc != "localhost":
            return Path(f"//{parsed.netloc}{path}")

        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]

        return Path(path)

    @staticmethod
    def _remove_temporary_file(path: Path) -> None:
        if path.exists():
            path.unlink()

    @staticmethod
    def _validate_image_format(
        path: Path, *, destination: Path | None = None
    ) -> None:
        effective_destination = path if destination is None else destination
        suffix = effective_destination.suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(
                f"Unsupported downloaded image extension "
                f"'{effective_destination.suffix}'. "
                f"Expected one of {sorted(ALLOWED_IMAGE_SUFFIXES)}."
            )

        try:
            with Image.open(path) as image:
                image_format = image.format
                image.verify()
        except (OSError, SyntaxError, UnidentifiedImageError) as exc:
            raise ValueError(
                f"Downloaded file '{effective_destination.name}' "
                "is not a valid image."
            ) from exc

        if image_format is None:
            raise ValueError(
                f"Downloaded image '{effective_destination.name}' "
                "does not expose a format."
            )

        expected_formats = next(
            (
                pillow_formats
                for declared_format, suffixes in ALLOWED_IMAGE_SUFFIXES_BY_FORMAT.items()
                if suffix in suffixes
                for pillow_formats in [
                    ALLOWED_PILLOW_FORMATS_BY_FORMAT[declared_format]
                ]
            ),
            None,
        )
        if expected_formats is None:
            raise ValueError(
                f"Unsupported downloaded image extension "
                f"'{effective_destination.suffix}'. "
                f"Expected one of {sorted(ALLOWED_IMAGE_SUFFIXES)}."
            )

        if image_format.upper() not in expected_formats:
            raise ValueError(
                f"Downloaded image '{effective_destination.name}' "
                f"has format '{image_format}', which is incompatible "
                f"with extension '{effective_destination.suffix}'."
            )


def download_remote_file(
    url: str,
    destination: Path,
    *,
    timeout: float = 30.0,
    validate_image: bool = False,
) -> Path:
    """Downloads a remote file to the requested destination."""

    return RemoteFileDownloader().download(
        url,
        destination,
        timeout=timeout,
        validate_image=validate_image,
    )


ALLOWED_IMAGE_SUFFIXES_BY_FORMAT = {
    "JPEG": frozenset({".jpeg", ".jpg"}),
    "PNG": frozenset({".png"}),
    "WEBP": frozenset({".webp"}),
    "BMP": frozenset({".bmp"}),
    "TIFF": frozenset({".tif", ".tiff"}),
    "HEIC": frozenset({".heic"}),
    "AVIF": frozenset({".avif"}),
    "JP2": frozenset({".jp2"}),
    "DNG": frozenset({".dng"}),
    "MPO": frozenset({".mpo"}),
}
ALLOWED_IMAGE_SUFFIXES = frozenset(
    suffix
    for suffixes in ALLOWED_IMAGE_SUFFIXES_BY_FORMAT.values()
    for suffix in suffixes
)
ALLOWED_PILLOW_FORMATS_BY_FORMAT = {
    "JPEG": frozenset({"JPEG"}),
    "PNG": frozenset({"PNG"}),
    "WEBP": frozenset({"WEBP"}),
    "BMP": frozenset({"BMP"}),
    "TIFF": frozenset({"TIFF"}),
    "HEIC": frozenset({"HEIC", "HEIF"}),
    "AVIF": frozenset({"AVIF"}),
    "JP2": frozenset({"JP2", "JPEG2000"}),
    "DNG": frozenset({"DNG", "TIFF"}),
    "MPO": frozenset({"MPO"}),
}
FS_REMOTE_URL_SCHEMES = frozenset({"gcs", "gs", "s3"})
ALLOWED_REMOTE_URL_SCHEMES = frozenset(
    {"file", "http", "https", *FS_REMOTE_URL_SCHEMES}
)
