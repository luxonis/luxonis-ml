import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import pytest

from luxonis_ml.utils import environ
from luxonis_ml.utils.filesystem import LuxonisFileSystem, _get_protocol_and_path

URL_PATH = "luxonis-test-bucket/luxonis-ml-test-data/fs_test_data"

LOCAL_ROOT = Path("tests/data/test_filesystem")
LOCAL_FILE_PATH = LOCAL_ROOT / "_file.txt"
LOCAL_DIR_PATH = LOCAL_ROOT / "_dir"

skip_if_no_s3_credentials = pytest.mark.skipif(
    environ.AWS_ACCESS_KEY_ID is None
    or environ.AWS_SECRET_ACCESS_KEY is None
    or environ.AWS_S3_ENDPOINT_URL is None,
    reason="S3 credentials not set",
)

skip_if_no_gcs_credentials = pytest.mark.skipif(
    environ.GOOGLE_APPLICATION_CREDENTIALS is None,
    reason="GCS credentials not set",
)


def get_python_version():
    version = sys.version_info
    formatted_version = f"{version.major}{version.minor}"
    return formatted_version


def get_os():
    os_name = platform.system().lower()
    if "darwin" in os_name:
        return "mac"
    elif "linux" in os_name:
        return "lin"
    elif "windows" in os_name:
        return "win"
    else:
        raise ValueError(f"Unsupported operating system: {os_name}")


# NOTE: needed for tests running in GitHub Actions using the matrix strategy
#       to avoid race conditions when running tests in parallel
def get_os_python_specific_url(protocol: str):
    os_name = get_os()
    python_version = get_python_version()
    return f"{protocol}://{URL_PATH}_{os_name}_{python_version}"


@pytest.fixture
def fs(request):
    url_path = get_os_python_specific_url(request.param)
    yield LuxonisFileSystem(url_path)


def parametrize_dependent_fixture(
    name: str = "fs",
    depends: Optional[List[str]] = None,
):
    depends = depends or []
    skips = {
        "gs": skip_if_no_gcs_credentials,
        "s3": skip_if_no_s3_credentials,
    }

    def decorator(func):
        protocols = ["gs", "s3"]
        return pytest.mark.parametrize(
            name,
            [
                pytest.param(
                    protocol,
                    marks=[
                        pytest.mark.dependency(
                            depends=[f"{d}_{protocol}" for d in depends],
                            name=f"{func.__name__}_{protocol}",
                        ),
                        skips[protocol],
                    ],
                )
                for protocol in protocols
            ],
            indirect=name == "fs",
        )(func)

    return decorator


def clean(fs: LuxonisFileSystem, name: str):
    if Path(name).suffix:
        fs.delete_file(name)
    else:
        fs.delete_dir(name)
    assert not fs.exists(name)


@pytest.fixture(scope="function", autouse=True)
def setup_remote_tests():
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_DIR_PATH.mkdir(parents=True, exist_ok=True)
    LOCAL_FILE_PATH.write_text("test 42")

    for i in range(5):
        file_path = LOCAL_DIR_PATH / f"file_{i}.txt"
        file_path.write_text(f"file {i + 21}")

    yield

    shutil.rmtree(LOCAL_ROOT)


def test_protocol():
    assert _get_protocol_and_path("foo://bar/baz") == ("foo", "bar/baz")
    assert _get_protocol_and_path("gs://foo/bar") == ("gcs", "foo/bar")
    assert _get_protocol_and_path("local_path/to/file") == (
        "file",
        "local_path/to/file",
    )

    with pytest.raises(ValueError):
        LuxonisFileSystem("foo://bar")


def test_fail():
    with pytest.raises(ValueError):
        LuxonisFileSystem(None)  # type: ignore
        LuxonisFileSystem(str(LOCAL_FILE_PATH), allow_local=False)


@parametrize_dependent_fixture()
def test_file_download(fs: LuxonisFileSystem):
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = fs.get_file("file.txt", tempdir)
        assert file_path.exists()
        assert file_path.read_text() == "file\n"


@parametrize_dependent_fixture()
def test_bytes(fs: LuxonisFileSystem):
    fs.put_bytes(b"bytes_test", "test_bytes.txt")
    assert fs.exists("test_bytes.txt")
    buffer = fs.read_to_byte_buffer("test_bytes.txt")
    assert buffer.read() == b"bytes_test"

    clean(fs, "test_bytes.txt")


@parametrize_dependent_fixture(depends=["test_file_download"])
def test_file_upload(fs: LuxonisFileSystem):
    fs.put_file(LOCAL_FILE_PATH, "_file_upload_test.txt")
    assert fs.exists("_file_upload_test.txt")

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = fs.get_file("_file_upload_test.txt", tempdir)
        assert file_path.exists()
        assert file_path.read_text() == LOCAL_FILE_PATH.read_text()

    clean(fs, "_file_upload_test.txt")


@parametrize_dependent_fixture()
def test_dir_download(fs: LuxonisFileSystem):
    with tempfile.TemporaryDirectory() as tempdir:
        dir_path = fs.get_dir("dir", tempdir)

        assert dir_path.exists()
        for i in range(1, 6):
            file_path = Path(dir_path, f"file_{i}.txt")
            assert file_path.exists()
            assert file_path.read_text() == f"file_{i}\n"

    with tempfile.TemporaryDirectory() as tempdir:
        dir_path = fs.get_dir(
            [
                "dir/file_1.txt",
                "dir/file_2.txt",
                "dir/file_3.txt",
                "dir/file_4.txt",
                "dir/file_5.txt",
            ],
            tempdir,
        )

        assert dir_path.exists()
        for i in range(1, 6):
            file_path = Path(dir_path, f"file_{i}.txt")
            assert file_path.exists()
            assert file_path.read_text() == f"file_{i}\n"


@parametrize_dependent_fixture(depends=["test_dir_download"])
def test_dir_upload(fs: LuxonisFileSystem):
    if fs.exists("_dir_upload_test"):
        fs.delete_dir("_dir_upload_test")
    assert not fs.exists("_dir_upload_test")

    fs.put_dir(LOCAL_DIR_PATH, "_dir_upload_test")
    assert fs.exists("_dir_upload_test")
    assert fs.is_directory("_dir_upload_test")

    def check_contents(dir_path: Path):
        assert dir_path.exists()
        for i in range(5):
            file_path = Path(dir_path, f"file_{i}.txt")
            assert file_path.exists()
            assert (
                file_path.read_text() == (LOCAL_DIR_PATH / f"file_{i}.txt").read_text()
            )

    with tempfile.TemporaryDirectory() as tempdir:
        dir_path = fs.get_dir("_dir_upload_test", tempdir)
        check_contents(dir_path)

    fs.delete_files([f"_dir_upload_test/file_{i}.txt" for i in range(4)])
    clean(fs, "_dir_upload_test")

    fs.put_dir([str(p) for p in LOCAL_DIR_PATH.iterdir()], "__dir_upload_test")
    assert fs.exists("__dir_upload_test")

    with tempfile.TemporaryDirectory() as tempdir:
        dir_path = fs.get_dir("__dir_upload_test", tempdir)
        check_contents(dir_path)

    clean(fs, "__dir_upload_test")


@parametrize_dependent_fixture()
def test_walk_dir(fs: LuxonisFileSystem):
    walked_files = [Path(file).name for file in fs.walk_dir("dir")]
    assert set(walked_files) == {f"file_{i}.txt" for i in range(1, 6)}


@parametrize_dependent_fixture(
    "protocol",
    [
        "test_file_download",
        "test_dir_download",
    ],
)
def test_static_download(protocol: str):
    url_root = get_os_python_specific_url(protocol)
    with tempfile.TemporaryDirectory() as tempdir:
        url = f"{url_root}/file.txt"
        path = LuxonisFileSystem.download(url, tempdir)
        assert path.exists()
        assert path.read_text() == "file\n"

        url = f"{url_root}/dir"
        path = LuxonisFileSystem.download(url, tempdir)
        assert path.exists()
        for i in range(1, 6):
            file_path = Path(path, f"file_{i}.txt")
            assert file_path.exists()
            assert file_path.read_text() == f"file_{i}\n"


@parametrize_dependent_fixture(
    "protocol",
    [
        "test_file_upload",
        "test_dir_upload",
        "test_static_download",
    ],
)
def test_static_upload(protocol: str):
    url_root = get_os_python_specific_url(protocol)
    with tempfile.TemporaryDirectory() as tempdir:
        url = f"{url_root}/_file_upload_test.txt"
        LuxonisFileSystem.upload(LOCAL_FILE_PATH, url)

        file_path = LuxonisFileSystem.download(url, tempdir)
        assert file_path.exists()
        assert file_path.read_text() == LOCAL_FILE_PATH.read_text()

        url = f"{url_root}/_dir_upload_test"
        LuxonisFileSystem.upload(LOCAL_DIR_PATH, url)

        dir_path = LuxonisFileSystem.download(url, tempdir)
        assert dir_path.exists()
        for i in range(5):
            file_path = Path(dir_path, f"file_{i}.txt")
            assert file_path.exists()
            assert (
                file_path.read_text() == (LOCAL_DIR_PATH / f"file_{i}.txt").read_text()
            )

    fs = LuxonisFileSystem(url_root)
    clean(fs, "_file_upload_test.txt")
    clean(fs, "_dir_upload_test")
