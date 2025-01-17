import random
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest
from pytest_subtests import SubTests

from luxonis_ml.utils.filesystem import (
    LuxonisFileSystem,
    _get_protocol_and_path,
)

URL_PATH = "luxonis-test-bucket/luxonis-ml-test-data/fs_test_data"

PROTOCOLS = ["gcs", "s3"]


@pytest.fixture
def protocol_tempdir(tempdir: Path, request: SubRequest) -> Path:
    protocol = request.getfixturevalue("protocol")
    rand = random.randint(0, 10000)
    tempdir = tempdir / protocol / str(rand)
    tempdir.mkdir(parents=True)
    return tempdir


@pytest.fixture
def local_file(protocol_tempdir: Path) -> Path:
    rand = random.randint(0, 10000)
    file_path = protocol_tempdir / f"file_{rand}.txt"
    file_path.write_text(f"test {rand}")

    return file_path


@pytest.fixture
def local_dir(protocol_tempdir: Path) -> Path:
    rand = random.randint(0, 10000)

    dir_path = protocol_tempdir / f"dir_{rand}"
    dir_path.mkdir()

    for i in range(5):
        file_path = dir_path / f"file_{i}.txt"
        file_path.write_text(f"file {rand}")

    (dir_path / "nested").mkdir()
    (dir_path / "nested" / "file.txt").write_text(f"nested file {rand}")

    return dir_path


@pytest.fixture(params=PROTOCOLS)
def protocol(request: SubRequest) -> str:
    return request.param


@pytest.fixture
def fs(protocol: str, python_version: str, platform_name: str):
    url_path = get_platform_specific_url(
        protocol, platform_name, python_version
    )
    yield LuxonisFileSystem(url_path)


def test_protocol():
    assert _get_protocol_and_path("foo://bar/baz") == ("foo", "bar/baz")
    assert _get_protocol_and_path("gs://foo/bar") == ("gcs", "foo/bar")
    assert _get_protocol_and_path("local_path/to/file") == (
        "file",
        "local_path/to/file",
    )

    with pytest.raises(ValueError):
        LuxonisFileSystem("foo://bar")


def test_fail(tempdir: Path):
    file = tempdir / f"file_{random.randint(0, 10000)}.txt"
    file.write_text("test")
    with pytest.raises(ValueError):
        LuxonisFileSystem(str(local_file), allow_local=False)


def test_single_file(
    fs: LuxonisFileSystem,
    local_file: Path,
    subtests: SubTests,
    protocol_tempdir: Path,
):
    uploaded_file = f"test_upload_{local_file.name}"
    if fs.exists(uploaded_file):  # pragma: no cover
        fs.delete_file(uploaded_file)

    assert not fs.exists(uploaded_file)

    with subtests.test("upload"):
        fs.put_file(local_file, uploaded_file)
        assert fs.exists(uploaded_file)

    with subtests.test("download"):
        file_path = fs.get_file(
            uploaded_file, protocol_tempdir / f"copy_{uploaded_file}"
        )
        assert file_path.exists()
        assert file_path.read_text() == local_file.read_text()

    with subtests.test("delete"):
        fs.delete_file(uploaded_file)
        assert not fs.exists(uploaded_file)


def test_directory(
    fs: LuxonisFileSystem,
    local_dir: Path,
    subtests: SubTests,
    protocol_tempdir: Path,
):
    uploaded_dir = f"dir_upload_{local_dir.name}"
    if fs.exists(uploaded_dir):  # pragma: no cover
        fs.delete_dir(uploaded_dir)

    assert not fs.exists(uploaded_dir)

    with subtests.test("upload_name"):
        fs.put_dir(local_dir, uploaded_dir)
        assert fs.exists(uploaded_dir)
        assert fs.is_directory(uploaded_dir)

    with subtests.test("upload_list"):
        fs.put_dir(
            [str(p) for p in local_dir.iterdir()],
            f"{uploaded_dir}_list",
        )
        assert fs.exists(f"{uploaded_dir}_list")
        assert fs.is_directory(f"{uploaded_dir}_list")

    with subtests.test("download_name"):
        dir_path = fs.get_dir(
            uploaded_dir, protocol_tempdir / f"dowload_{uploaded_dir}_name"
        )
        compare_directories(dir_path, local_dir)

    with subtests.test("download_list"):
        dir_path = fs.get_dir(
            [f"{uploaded_dir}/file_{i}.txt" for i in range(5)]
            + [f"{uploaded_dir}/nested/file.txt"],
            protocol_tempdir / f"dowload_{uploaded_dir}_list",
        )
        compare_directories(dir_path, local_dir, flat=True)


def test_walk_dir(fs: LuxonisFileSystem, local_dir: Path, subtests: SubTests):
    uploaded_dir = f"dir_upload_{local_dir.name}"

    if fs.exists(uploaded_dir):  # pragma: no cover
        fs.delete_dir(uploaded_dir)

    fs.put_dir(local_dir, uploaded_dir)

    def walk(**kwargs):
        return sorted(
            [
                path.lstrip(f"{uploaded_dir}/")
                for path in fs.walk_dir(uploaded_dir, **kwargs)
            ]
        )

    with subtests.test("recursive"):
        files = walk(recursive=True, typ="file")

        assert files == sorted(
            [f"file_{i}.txt" for i in range(5)] + ["nested/file.txt"]
        )

    with subtests.test("all"):
        files = walk(recursive=False, typ="all")

        assert files == sorted(
            [f"file_{i}.txt" for i in range(5)] + ["nested"]
        )

    with subtests.test("directories"):
        files = walk(recursive=False, typ="directory")
        assert files == ["nested"]


def test_bytes(fs: LuxonisFileSystem):
    rand = random.randint(0, 10000)
    bytes_file = f"bytes_test_{rand}.txt"
    fs.put_bytes(f"bytes test {rand}".encode(), bytes_file)
    assert fs.exists(bytes_file)
    buffer = fs.read_to_byte_buffer(bytes_file)
    assert buffer.read() == f"bytes test {rand}".encode()


def test_static(
    fs: LuxonisFileSystem,
    local_file: Path,
    local_dir: Path,
    protocol_tempdir: Path,
    subtests: SubTests,
):
    rand = random.randint(0, 10000)
    with subtests.test("file"):
        file_name = f"static_file_upload_{rand}.txt"
        url = f"{fs.url}/{file_name}"
        with subtests.test("upload"):
            assert not fs.exists(file_name)
            LuxonisFileSystem.upload(local_file, url)
            assert fs.exists(file_name)

        with subtests.test("download"):
            assert fs.exists(file_name)
            file = LuxonisFileSystem.download(url, protocol_tempdir)
            assert file.exists()
            assert file.read_text() == local_file.read_text()

    with subtests.test("directory"):
        dir_name = f"static_dir_upload_{rand}"
        url = f"{fs.url}/{dir_name}"
        with subtests.test("upload"):
            assert not fs.exists(dir_name)
            LuxonisFileSystem.upload(local_dir, url)
            assert fs.exists(dir_name)

        with subtests.test("download"):
            assert fs.exists(dir_name)
            dir = LuxonisFileSystem.download(url, protocol_tempdir)
            compare_directories(dir, local_dir)


def compare_directories(
    dir_path: Path, orig_dir_path: Path, flat: bool = False
):
    assert dir_path.exists()
    for i in range(5):
        file_path = dir_path / f"file_{i}.txt"
        assert file_path.exists()
        assert (
            file_path.read_text()
            == (orig_dir_path / f"file_{i}.txt").read_text()
        )
    if flat:
        file_path = "file.txt"
    else:
        file_path = "nested/file.txt"
    nested_file = dir_path / file_path

    assert nested_file.exists()
    assert (
        nested_file.read_text()
        == (orig_dir_path / "nested" / "file.txt").read_text()
    )


def get_platform_specific_url(
    protocol: str, platform: str, python_version: str
):
    return f"{protocol}://{URL_PATH}_{platform}_{python_version}"
