import shutil
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from filelock import FileLock
from PIL import Image, UnidentifiedImageError

from luxonis_ml.utils.filesystem import LuxonisFileSystem

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


def download_remote_file(
    url: str,
    destination: Path,
    *,
    timeout: float = 30.0,
    validate_image: bool = False,
) -> Path:
    """Downloads a remote file to the requested destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(f"{destination.suffix}.lock")
    tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")

    with FileLock(str(lock_path)):
        if destination.exists():
            if validate_image:
                _validate_image_format(destination)
            return destination

        if tmp_path.exists():
            tmp_path.unlink()

        scheme = urlsplit(url).scheme
        if scheme not in ALLOWED_REMOTE_URL_SCHEMES:
            raise ValueError(
                f"Unsupported remote URL scheme '{scheme}'. Expected one of "
                f"{sorted(ALLOWED_REMOTE_URL_SCHEMES)}."
            )

        try:
            if scheme in FS_REMOTE_URL_SCHEMES:
                downloaded_path = LuxonisFileSystem.download(url, tmp_path)
                if downloaded_path != tmp_path:
                    downloaded_path.replace(tmp_path)
            else:
                request = (
                    url
                    if scheme == "file"
                    else Request(  # noqa: S310
                        url, headers={"User-Agent": "luxonis-ml"}
                    )
                )
                with (
                    urlopen(request, timeout=timeout) as response,  # nosec B310  # noqa: S310
                    open(tmp_path, "wb") as output_file,
                ):
                    shutil.copyfileobj(response, output_file)

            if validate_image:
                _validate_image_format(tmp_path, destination=destination)

            tmp_path.replace(destination)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    return destination


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
            f"Downloaded file '{effective_destination.name}' is not a valid image."
        ) from exc

    if image_format is None:
        raise ValueError(
            f"Downloaded image '{effective_destination.name}' does not expose a format."
        )

    expected_formats = next(
        (
            pillow_formats
            for declared_format, suffixes in ALLOWED_IMAGE_SUFFIXES_BY_FORMAT.items()
            if suffix in suffixes
            for pillow_formats in [ALLOWED_PILLOW_FORMATS_BY_FORMAT[declared_format]]
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
            f"Downloaded image '{effective_destination.name}' has format "
            f"'{image_format}', which is incompatible with extension "
            f"'{effective_destination.suffix}'."
        )
