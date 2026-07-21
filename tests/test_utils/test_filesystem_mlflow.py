import uuid
from pathlib import Path

import pytest

from luxonis_ml.utils.filesystem import (
    LuxonisFileSystem,
    _get_protocol_and_path,
)

MLFLOW = pytest.importorskip("mlflow")


@pytest.fixture
def mlflow_tracking_uri(tempdir: Path, monkeypatch: pytest.MonkeyPatch):
    tracking_dir = (tempdir / "mlflow").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{(tracking_dir / 'tracking.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    MLFLOW.set_tracking_uri(tracking_uri)

    yield tracking_uri

    MLFLOW.end_run()


@pytest.fixture
def mlflow_run(mlflow_tracking_uri: str, tempdir: Path, randint: int):
    artifact_root = (tempdir / "artifacts").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment_id = MLFLOW.create_experiment(
        f"fs-test-{randint}", artifact_location=artifact_root.as_uri()
    )
    client = MLFLOW.MlflowClient(tracking_uri=mlflow_tracking_uri)
    run = client.create_run(experiment_id)

    yield experiment_id, run.info.run_id

    client.set_terminated(run.info.run_id)


def test_mlflow_split_full_path():
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
    monkeypatch: pytest.MonkeyPatch,
):
    experiment_id, run_id = mlflow_run
    local_file = tempdir / "source.txt"
    local_file.write_text("mlflow file contents")

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
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
    artifact_root = (tempdir / "active-artifacts").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment_id = MLFLOW.create_experiment(
        f"active-fs-test-{randint}",
        artifact_location=artifact_root.as_uri(),
    )
    local_file = tempdir / "active-source.txt"
    local_file.write_text("active run payload")

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

        explicit_fs = LuxonisFileSystem(
            f"mlflow://{experiment_id}/{run.info.run_id}",
            tracking_uri=mlflow_tracking_uri,
        )
        assert explicit_fs.exists("active/renamed.txt")
        assert explicit_fs.read_text("active/renamed.txt") == (
            "active run payload"
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

    artifact_root = (tempdir / "tracker-artifacts").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment_id = MLFLOW.create_experiment(
        f"tracker-test-{randint}",
        artifact_location=artifact_root.as_uri(),
    )
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
