import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import closing, suppress
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from luxonis_ml.utils.filesystem import (
    LuxonisFileSystem,
    _get_protocol_and_path,
)

MLFLOW = pytest.importorskip("mlflow")


def _get_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_mlflow_server(
    tracking_uri: str, process: subprocess.Popen, log_path: Path
) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(log_path.read_text())
        try:
            with urlopen(  # noqa: S310
                f"{tracking_uri}/health", timeout=0.5
            ) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, socket.timeout):
            time.sleep(0.2)
    process.terminate()
    raise TimeoutError(log_path.read_text())


@pytest.fixture(scope="module")
def mlflow_tracking_uri(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    server_dir = tmp_path_factory.mktemp("mlflow-server")
    backend_store_uri = f"sqlite:///{(server_dir / 'tracking.db').as_posix()}"
    artifacts_dir = server_dir / "artifacts"
    artifacts_dir.mkdir()
    port = _get_free_port()
    tracking_uri = f"http://127.0.0.1:{port}"
    log_path = server_dir / "server.log"
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--backend-store-uri",
                backend_store_uri,
                "--default-artifact-root",
                "mlflow-artifacts:/",
                "--artifacts-destination",
                artifacts_dir.resolve().as_uri(),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        _wait_for_mlflow_server(tracking_uri, process, log_path)

        yield tracking_uri

        MLFLOW.end_run()
        process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.fixture(autouse=True)
def configure_mlflow_tracking_uri(mlflow_tracking_uri: str) -> Iterator[None]:
    MLFLOW.set_tracking_uri(mlflow_tracking_uri)

    yield

    MLFLOW.end_run()


@pytest.fixture
def mlflow_run(mlflow_tracking_uri: str, tempdir: Path, randint: int):
    experiment_id = MLFLOW.create_experiment(f"fs-test-{randint}")
    client = MLFLOW.MlflowClient(tracking_uri=mlflow_tracking_uri)
    run = client.create_run(experiment_id)

    yield experiment_id, run.info.run_id

    client.set_terminated(run.info.run_id)


def test_mlflow_split_full_path():
    assert LuxonisFileSystem._split_mlflow_path("0") == ["0", None, None]
    assert _get_protocol_and_path("mlflow://0/run/nested/file.txt") == (
        "mlflow",
        "0/run/nested/file.txt",
    )
    assert LuxonisFileSystem.split_full_path(
        "mlflow://0/run/nested/file.txt"
    ) == ("mlflow://0/run", "nested/file.txt")
    assert LuxonisFileSystem.split_full_path("mlflow://0/run") == (
        "mlflow://0/run",
        "",
    )
    assert LuxonisFileSystem.split_full_path("mlflow://") == (
        "mlflow://",
        "",
    )


def test_mlflow_file_operations(
    tempdir: Path,
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    experiment_id, run_id = mlflow_run
    local_file = tempdir / "source.txt"
    local_file.write_text("mlflow file contents")

    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )

    remote_path = "nested/renamed.txt"
    assert fs.put_file(local_file, remote_path) == (
        f"mlflow://{experiment_id}/{run_id}/{remote_path}"
    )
    assert fs.exists(remote_path)
    assert not fs.is_directory(remote_path)
    assert fs.read_text(remote_path) == local_file.read_text()
    assert (
        fs.read_to_byte_buffer(remote_path).read() == local_file.read_bytes()
    )

    expected_uuid = uuid.uuid5(
        uuid.NAMESPACE_URL, local_file.read_bytes().hex()
    )
    assert fs.get_file_uuid(remote_path) == str(expected_uuid)
    assert fs.get_file_uuids([remote_path]) == {
        remote_path: str(expected_uuid)
    }

    downloaded_file = fs.get_file(remote_path, tempdir / "downloaded.txt")
    assert downloaded_file.read_text() == local_file.read_text()

    explicit_download = LuxonisFileSystem.download(
        f"mlflow://{experiment_id}/{run_id}/{remote_path}",
        tempdir / "explicit.txt",
        tracking_uri=mlflow_tracking_uri,
    )
    assert explicit_download.read_text() == local_file.read_text()

    static_download_dir = tempdir / "static-download"
    LuxonisFileSystem.download(
        f"mlflow://{experiment_id}/{run_id}/{remote_path}",
        static_download_dir,
        tracking_uri=mlflow_tracking_uri,
    )
    assert (
        static_download_dir / "nested" / "renamed.txt"
    ).read_text() == local_file.read_text()


def test_mlflow_directory_operations(
    tempdir: Path,
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    experiment_id, run_id = mlflow_run
    local_dir = tempdir / "local-dir"
    local_dir.mkdir()
    (local_dir / "file_0.txt").write_text("file 0")
    (local_dir / "nested").mkdir()
    (local_dir / "nested" / "file_1.txt").write_text("file 1")

    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )
    fs.put_dir(local_dir, "remote-dir")

    assert fs.exists("remote-dir")
    assert fs.is_directory("remote-dir")
    assert fs.is_directory("")
    assert fs.exists("")
    assert not fs.exists("missing")
    assert set(fs.walk_dir("remote-dir", recursive=True)) == {
        "remote-dir/file_0.txt",
        "remote-dir/nested/file_1.txt",
    }
    assert set(fs.walk_dir("remote-dir", recursive=False, typ="all")) == {
        "remote-dir/file_0.txt",
        "remote-dir/nested",
    }

    downloaded_dir = fs.get_dir("remote-dir", tempdir / "downloaded-dir")
    assert (downloaded_dir / "file_0.txt").read_text() == "file 0"
    assert (downloaded_dir / "nested" / "file_1.txt").read_text() == "file 1"

    existing_dir = tempdir / "existing-download-root"
    existing_dir.mkdir()
    existing_download = fs.get_dir("remote-dir", existing_dir)
    assert existing_download == existing_dir / "remote-dir"
    assert (existing_download / "file_0.txt").read_text() == "file 0"
    assert (
        existing_download / "nested" / "file_1.txt"
    ).read_text() == "file 1"

    static_source = tempdir / "static-source"
    static_source.mkdir()
    (static_source / "payload.txt").write_text("payload")
    LuxonisFileSystem.upload(
        static_source,
        f"mlflow://{experiment_id}/{run_id}/static-dir",
        tracking_uri=mlflow_tracking_uri,
    )
    static_download = LuxonisFileSystem.download(
        f"mlflow://{experiment_id}/{run_id}/static-dir",
        tempdir / "static-dir-download",
        tracking_uri=mlflow_tracking_uri,
    )
    assert (static_download / "payload.txt").read_text() == "payload"


def test_mlflow_error_paths(
    tempdir: Path,
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    experiment_id, run_id = mlflow_run
    local_file = tempdir / "source.txt"
    local_file.write_text("payload")
    local_dir = tempdir / "local-dir"
    local_dir.mkdir()

    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )
    with pytest.raises(ValueError, match="No relative artifact path"):
        fs.read_to_byte_buffer()
    with pytest.raises(ValueError, match="No relative artifact path"):
        fs.put_file(local_file, "")

    missing_run_fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/missing-run",
        tracking_uri=mlflow_tracking_uri,
    )
    assert not missing_run_fs.exists()
    with pytest.raises(ValueError, match="run_id"):
        LuxonisFileSystem(
            f"mlflow://{experiment_id}", tracking_uri=mlflow_tracking_uri
        )._require_mlflow_run_id()

    artifact_fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}/base",
        tracking_uri=mlflow_tracking_uri,
    )
    assert artifact_fs._get_mlflow_artifact_path("child.txt") == (
        "base/child.txt"
    )
    assert artifact_fs._relative_mlflow_artifact_path("other/file.txt") == (
        "other/file.txt"
    )

    active_fs = LuxonisFileSystem(
        "mlflow://",
        allow_active_mlflow_run=True,
        allow_local=False,
        tracking_uri=mlflow_tracking_uri,
    )
    assert active_fs._get_mlflow_url("fallback.txt") == (
        "mlflow:///fallback.txt"
    )
    with pytest.raises(ValueError, match="Reading to byte buffer"):
        active_fs.read_to_byte_buffer()
    with pytest.raises(ValueError, match="No active mlflow_instance"):
        active_fs.put_file(local_file, "artifact.txt")
    with pytest.raises(ValueError, match="No active mlflow_instance"):
        active_fs.put_dir(local_dir, "artifacts")


def test_mlflow_put_bytes(
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    experiment_id, run_id = mlflow_run
    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )
    fs.put_bytes(b"binary payload", "bytes/payload.bin")

    assert fs.exists("bytes/payload.bin")
    assert fs.read_to_byte_buffer("bytes/payload.bin").read() == (
        b"binary payload"
    )


def test_mlflow_active_run_upload(
    tempdir: Path,
    mlflow_tracking_uri: str,
    randint: int,
):
    experiment_id = MLFLOW.create_experiment(f"active-fs-test-{randint}")
    local_file = tempdir / "active-source.txt"
    local_file.write_text("active run payload")
    local_dir = tempdir / "active-dir-source"
    local_dir.mkdir()
    (local_dir / "active-dir-file.txt").write_text("active dir payload")

    with MLFLOW.start_run(experiment_id=experiment_id) as run:
        fs = LuxonisFileSystem(
            "mlflow://",
            allow_active_mlflow_run=True,
            allow_local=False,
            tracking_uri=mlflow_tracking_uri,
        )
        assert (
            fs.put_file(
                local_file, "active/renamed.txt", mlflow_instance=MLFLOW
            )
            == f"mlflow://{experiment_id}/{run.info.run_id}/active/renamed.txt"
        )
        fs.put_dir(local_dir, "active-dir", mlflow_instance=MLFLOW)

        explicit_fs = LuxonisFileSystem(
            f"mlflow://{experiment_id}/{run.info.run_id}",
            tracking_uri=mlflow_tracking_uri,
        )
        assert explicit_fs.exists("active/renamed.txt")
        assert explicit_fs.read_text("active/renamed.txt") == (
            "active run payload"
        )
        assert explicit_fs.read_text("active-dir/active-dir-file.txt") == (
            "active dir payload"
        )


def test_mlflow_delete_is_unsupported(
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    experiment_id, run_id = mlflow_run
    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )

    with pytest.raises(NotImplementedError, match="cannot be deleted"):
        fs.delete_file("artifact.txt")
    with pytest.raises(NotImplementedError, match="cannot be deleted"):
        fs.delete_files(["artifact.txt"])
    with pytest.raises(NotImplementedError, match="cannot be deleted"):
        fs.delete_dir("artifacts")


def test_tracker_upload_artifact_to_mlflow(
    tempdir: Path,
    mlflow_tracking_uri: str,
    randint: int,
):
    from luxonis_ml.tracker import LuxonisTracker

    experiment_id = MLFLOW.create_experiment(f"tracker-test-{randint}")
    artifact = tempdir / "tracker-source.txt"
    artifact.write_text("tracker payload")
    tracker = LuxonisTracker(
        project_id=experiment_id,
        run_name=f"tracker-run-{randint}",
        save_directory=tempdir / "output",
        is_mlflow=True,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    tracker.upload_artifact(artifact, name="tracker/renamed.txt")
    assert tracker.project_id is not None
    assert tracker.run_id is not None

    fs = LuxonisFileSystem(
        f"mlflow://{tracker.project_id}/{tracker.run_id}",
        tracking_uri=mlflow_tracking_uri,
    )
    assert fs.exists("tracker/renamed.txt")
    assert fs.read_text("tracker/renamed.txt") == "tracker payload"

    tracker.experiment["mlflow"].end_run()
