from pathlib import Path

import pytest

from luxonis_ml.data.utils import RemoteFileDownloader, download_remote_file

from .utils import create_image


class RecordingAPIClient:
    def __init__(self, source: Path | None = None) -> None:
        self.source = source
        self.calls: list[tuple[str, Path, float]] = []

    def download(self, url: str, destination: Path, *, timeout: float) -> None:
        self.calls.append((url, destination, timeout))
        if self.source is None:
            raise AssertionError("API client should not have been called.")
        destination.write_bytes(self.source.read_bytes())


def assert_no_temporary_artifacts(destination: Path) -> None:
    assert not destination.exists()
    assert not destination.with_suffix(f"{destination.suffix}.tmp").exists()


def test_download_remote_file_accepts_valid_file_url_image(
    tempdir: Path,
) -> None:
    source = create_image(0, tempdir)
    destination = tempdir / "downloads" / "copied.jpg"

    downloaded = download_remote_file(
        source.as_uri(), destination, validate_image=True
    )

    assert downloaded == destination
    assert destination.is_file()
    assert destination.read_bytes() == source.read_bytes()


def test_remote_file_downloader_dispatches_https_urls_to_api_client(
    tempdir: Path,
) -> None:
    source = create_image(1, tempdir)
    destination = tempdir / "downloads" / "copied.jpg"
    api_client = RecordingAPIClient(source=source)
    downloader = RemoteFileDownloader(api_client=api_client)

    downloaded = downloader.download(
        "https://example.com/dataset/copied.jpg",
        destination,
        timeout=12.5,
        validate_image=True,
    )

    assert downloaded == destination
    assert destination.read_bytes() == source.read_bytes()
    assert api_client.calls == [
        (
            "https://example.com/dataset/copied.jpg",
            destination.with_suffix(".jpg.tmp"),
            12.5,
        )
    ]


def test_remote_file_downloader_reuses_existing_destination(
    tempdir: Path,
) -> None:
    destination = create_image(2, tempdir)
    api_client = RecordingAPIClient()
    downloader = RemoteFileDownloader(api_client=api_client)

    downloaded = downloader.download(
        "https://example.com/should-not-be-downloaded.jpg",
        destination,
        validate_image=True,
    )

    assert downloaded == destination
    assert api_client.calls == []


def test_remote_file_downloader_rejects_unsupported_scheme(
    tempdir: Path,
) -> None:
    destination = tempdir / "downloads" / "payload.jpg"
    api_client = RecordingAPIClient()
    downloader = RemoteFileDownloader(api_client=api_client)

    with pytest.raises(ValueError, match="Unsupported remote URL scheme"):
        downloader.download("ftp://example.com/payload.jpg", destination)

    assert api_client.calls == []
    assert_no_temporary_artifacts(destination)


def test_download_remote_file_rejects_non_image_content_and_cleans_up(
    tempdir: Path,
) -> None:
    source = tempdir / "payload.jpg"
    source.write_text("not an image", encoding="utf-8")
    destination = tempdir / "downloads" / "copied.jpg"

    with pytest.raises(ValueError, match="not a valid image"):
        download_remote_file(source.as_uri(), destination, validate_image=True)

    assert_no_temporary_artifacts(destination)


def test_download_remote_file_rejects_extension_mismatch_and_cleans_up(
    tempdir: Path,
) -> None:
    source = create_image(1, tempdir)
    destination = tempdir / "downloads" / "copied.png"

    with pytest.raises(ValueError, match="incompatible with extension"):
        download_remote_file(source.as_uri(), destination, validate_image=True)

    assert_no_temporary_artifacts(destination)


def test_remote_file_downloader_does_not_send_file_urls_to_api_client(
    tempdir: Path,
) -> None:
    source = create_image(3, tempdir)
    destination = tempdir / "downloads" / "copied.jpg"
    api_client = RecordingAPIClient()
    downloader = RemoteFileDownloader(api_client=api_client)

    downloaded = downloader.download(
        source.as_uri(),
        destination,
        validate_image=True,
    )

    assert downloaded == destination
    assert destination.read_bytes() == source.read_bytes()
    assert api_client.calls == []
