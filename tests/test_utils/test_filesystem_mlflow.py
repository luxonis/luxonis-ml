import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import closing, suppress
from pathlib import Path
from typing import NoReturn, cast
from urllib.error import URLError
from urllib.request import urlopen

import mlflow
import pytest

from luxonis_ml.utils import filesystem
from luxonis_ml.utils.filesystem import (
    LuxonisFileSystem,
    _get_protocol_and_path,
)

SERVER_TIMEOUT_ENV = "LUXONISML_TEST_MLFLOW_SERVER_TIMEOUT"
DEFAULT_SERVER_TIMEOUT = 120.0


def _get_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mlflow_server_timeout() -> float:
    """Seconds to wait for the test MLflow server to become healthy.

    The budget has to cover interpreter startup, the whole ``mlflow``
    import and alembic creating the schema of a fresh SQLite database,
    once per ``xdist`` worker in parallel. The environment variable lets
    a slow runner raise it without a code change.
    """
    return float(os.environ.get(SERVER_TIMEOUT_ENV, DEFAULT_SERVER_TIMEOUT))


def _wait_for_mlflow_server(
    tracking_uri: str, process: subprocess.Popen, log_path: Path
) -> None:
    timeout = _mlflow_server_timeout()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(log_path.read_text())
        try:
            with urlopen(  # noqa: S310
                f"{tracking_uri}/health", timeout=0.5
            ) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            time.sleep(0.2)
    process.terminate()
    raise TimeoutError(
        f"MLflow server did not become healthy within {timeout}s. "
        f"Set `{SERVER_TIMEOUT_ENV}` to wait longer.\n"
        f"{log_path.read_text()}"
    )


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

        mlflow.end_run()
        process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.fixture(autouse=True)
def configure_mlflow_tracking_uri(mlflow_tracking_uri: str) -> Iterator[None]:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    yield

    mlflow.end_run()


@pytest.fixture
def mlflow_run(mlflow_tracking_uri: str, tempdir: Path, randint: int):
    experiment_id = mlflow.create_experiment(f"fs-test-{randint}")
    client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)
    run = client.create_run(experiment_id)

    yield experiment_id, run.info.run_id

    client.set_terminated(run.info.run_id)


def _listing(fs: LuxonisFileSystem, remote_dir: str) -> set[str]:
    """List all files under ``remote_dir`` as POSIX paths.

    The ``fsspec`` backend yields native paths, MLflow always yields
    POSIX ones, so listings from the two are only comparable once
    normalized.
    """
    return {
        Path(path).as_posix()
        for path in fs.walk_dir(remote_dir, recursive=True)
    }


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


def test_mlflow_server_timeout_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test for the 30s MLflow server startup budget.

    The server the module fixture spawns has to get through interpreter
    startup, the whole ``mlflow`` import and alembic creating the schema
    of a fresh SQLite database -- once per ``xdist`` worker, all in
    parallel. 30s did not cover that on ``windows-latest``: CI run
    30660638060 killed a server whose log showed it still creating the
    initial tables 24s in, and the timeout failed the entire module.

    Both halves are pinned because both are load bearing: the default is
    what CI actually runs with, and the environment variable is the only
    way to give a slow runner more room without a code change.
    """
    monkeypatch.delenv(SERVER_TIMEOUT_ENV, raising=False)
    assert _mlflow_server_timeout() == 120
    monkeypatch.setenv(SERVER_TIMEOUT_ENV, "7.5")
    assert _mlflow_server_timeout() == 7.5


class _StillStartingProcess:
    """Stand-in for a server process that never finishes starting."""

    def poll(self) -> None:
        return None

    def terminate(self) -> None: ...


def test_mlflow_server_timeout_reports_how_long_it_waited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test for the unattributable startup timeout.

    The failure raised a bare ``TimeoutError(server_log)``, so CI showed
    the server's own log and nothing about the budget that had expired
    -- from the outside a hung server and one that merely needed more
    than the hard-coded 30s looked identical.

    Driving the real wait loop against a port nothing listens on (rather
    than asserting on the message alone) is what proves the configured
    value is honored, not just readable; the tiny budget keeps it
    instant.
    """
    monkeypatch.setenv(SERVER_TIMEOUT_ENV, "0.01")
    log_path = tmp_path / "server.log"
    log_path.write_text("Creating initial MLflow database tables...")

    with pytest.raises(TimeoutError, match=r"within 0\.01s") as error:
        _wait_for_mlflow_server(
            f"http://127.0.0.1:{_get_free_port()}",
            cast(subprocess.Popen, _StillStartingProcess()),
            log_path,
        )

    assert "Creating initial MLflow database tables" in str(error.value)


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
    # NOTE: `is_directory("")` used to return a hard-coded `True`
    # without contacting the server, so this assertion held for any
    # MLflow filesystem regardless of server state. The run-root case is
    # now server-backed; `test_mlflow_root_is_directory_checks_the_run`
    # covers the falsifiable half.
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


def test_mlflow_repeated_download_returns_fresh_data(
    tempdir: Path,
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    """Regression test for ``download`` returning a stale path.

    ``LuxonisFileSystem.download`` called ``get_dir``/``get_file`` and
    discarded their return values, handing back the path it had computed
    itself instead. Both methods deliberately nest the download inside
    ``local_path`` when that path already exists, so downloading the
    same remote directory twice into the same destination wrote the
    fresh data one level deeper (``<dest>/artifacts/artifacts``) while
    ``download`` kept returning ``<dest>/artifacts`` -- the stale copy
    from the first call. Every cache refresh or repeated run silently
    read outdated data.

    Reading a file *through the returned path* is the point of the
    assertion: the data is on disk either way, only the returned path
    tells the caller where.
    """
    experiment_id, run_id = mlflow_run
    source = tempdir / "download-source"
    source.mkdir()
    (source / "payload.txt").write_text("first")

    run_url = f"mlflow://{experiment_id}/{run_id}"
    fs = LuxonisFileSystem(run_url, tracking_uri=mlflow_tracking_uri)
    fs.put_dir(source, "artifacts", copy_contents=True)

    url = f"{run_url}/artifacts"
    dest = tempdir / "download-dest"
    first = LuxonisFileSystem.download(
        url, dest, tracking_uri=mlflow_tracking_uri
    )
    assert (first / "payload.txt").read_text() == "first"

    (source / "payload.txt").write_text("second")
    fs.put_dir(source, "artifacts", copy_contents=True)
    assert fs.read_text("artifacts/payload.txt") == "second"

    second = LuxonisFileSystem.download(
        url, dest, tracking_uri=mlflow_tracking_uri
    )
    assert (second / "payload.txt").read_text() == "second"


def test_mlflow_put_dir_matches_fsspec_layout(
    tempdir: Path,
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    """Regression test for ``copy_contents`` being ignored on MLflow.

    ``log_artifacts`` always uploads the *contents* of a directory, so
    the MLflow branch of ``put_dir`` accepted ``copy_contents`` and then
    behaved as if it were always ``True``. ``fsspec`` copies like
    ``cp -r``: with ``copy_contents=False`` a destination that already
    exists gets the source directory nested inside it. Uploading two
    directories that share file names into one ``remote_dir`` therefore
    kept them apart on every backend except MLflow, where the second
    upload silently overwrote the first.

    The assertion compares the full listings and file contents of a
    ``file://`` filesystem and an ``mlflow://`` one after identical
    calls, instead of hard-coding a layout, so the two backends stay
    pinned to each other against future drift.
    """
    experiment_id, run_id = mlflow_run
    for name, payload in [("dir_a", "a"), ("dir_b", "b")]:
        source = tempdir / name
        (source / "nested").mkdir(parents=True)
        (source / "payload.txt").write_text(payload)
        (source / "nested" / "inner.txt").write_text(f"nested {payload}")

    for copy_contents in (False, True):
        remote_dir = f"parity-{copy_contents}"
        local_root = tempdir / f"local-root-{copy_contents}"
        local_root.mkdir()
        filesystems = [
            LuxonisFileSystem(str(local_root)),
            LuxonisFileSystem(
                f"mlflow://{experiment_id}/{run_id}",
                tracking_uri=mlflow_tracking_uri,
            ),
        ]
        for fs in filesystems:
            for name in ("dir_a", "dir_b"):
                fs.put_dir(
                    tempdir / name, remote_dir, copy_contents=copy_contents
                )

        local_fs, mlflow_fs = filesystems
        listing = _listing(local_fs, remote_dir)
        assert listing == _listing(mlflow_fs, remote_dir)
        assert {file: local_fs.read_text(file) for file in listing} == {
            file: mlflow_fs.read_text(file) for file in listing
        }

        # The listings above would also match if both backends did
        # nothing, so pin down what the shared layout actually is.
        if copy_contents:
            assert listing == {
                f"{remote_dir}/payload.txt",
                f"{remote_dir}/nested/inner.txt",
            }
            assert local_fs.read_text(f"{remote_dir}/payload.txt") == "b"
        else:
            assert listing == {
                f"{remote_dir}/payload.txt",
                f"{remote_dir}/nested/inner.txt",
                f"{remote_dir}/dir_b/payload.txt",
                f"{remote_dir}/dir_b/nested/inner.txt",
            }
            assert local_fs.read_text(f"{remote_dir}/payload.txt") == "a"


def test_mlflow_transfers_are_staged_beside_their_endpoint(
    tempdir: Path,
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test for staging every transfer through ``$TMPDIR``.

    Downloads landed in a bare ``tempfile.TemporaryDirectory()`` and
    were then copied to the destination, and a renamed upload was copied
    into one the same way. ``$TMPDIR`` is ``/tmp``, on most Linux
    distros a RAM-backed tmpfs capped near half of memory, so pulling a
    multi-GB checkpoint onto a roomy ``/data`` volume died with
    ``OSError: [Errno 28] No space left on device`` -- and everything
    that did fit was written to disk twice.

    The ``dir`` argument is the entire fix: staged on the filesystem of
    the file it is about to become, the copy degrades into a rename (a
    hardlink when uploading), which costs nothing and cannot overflow an
    unrelated filesystem. Recording it is also the only evidence
    available, since the staging directory is gone once the call
    returns. Only ``luxonis_ml``'s own uses are recorded -- the patch
    replaces the module's ``tempfile`` global, not the module itself --
    so MLflow's internal temporary directories cannot mask a
    regression.
    """
    experiment_id, run_id = mlflow_run
    staging_dirs: list[Path | None] = []

    class RecordingTempfile:
        """Records where ``filesystem`` asks for its staging dirs."""

        @staticmethod
        def TemporaryDirectory(*args, **kwargs) -> tempfile.TemporaryDirectory:
            staging_dirs.append(kwargs.get("dir"))
            return tempfile.TemporaryDirectory(*args, **kwargs)

    monkeypatch.setattr(filesystem, "tempfile", RecordingTempfile)

    source_dir = tempdir / "staging-source"
    source_dir.mkdir()
    local_file = source_dir / "source.txt"
    local_file.write_text("staging payload")

    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )
    fs.put_file(local_file, "staged/renamed.txt")
    assert staging_dirs == [source_dir]

    # The destination's parent does not exist yet, so this also pins the
    # ordering: it has to be created before a staging dir can go in it.
    destination = tempdir / "staging-destination" / "downloaded.txt"
    downloaded = fs.get_file("staged/renamed.txt", destination)
    assert staging_dirs == [source_dir, destination.parent]
    assert downloaded == destination
    assert downloaded.read_text() == "staging payload"


@pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: 0)() == 0,
    reason="read-only directories are not enforced for Windows or root",
)
def test_mlflow_upload_falls_back_when_the_source_is_read_only(
    tempdir: Path,
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test for staging beside a read-only source.

    Staging next to the source needs write permission there, which a
    read-only mount -- a shared dataset volume, an archived checkpoint
    tree -- does not grant.
    ``tempfile.TemporaryDirectory(dir=local_path.parent)`` then raised
    ``PermissionError`` before the ``hardlink_to``/``copy2`` fallback
    could run, so a renamed upload out of such a directory could not
    succeed at all. That is the ordinary path rather than an edge case:
    ``_put_files`` renames every file to ``<uuid><suffix>`` whenever a
    ``uuid_dict`` is given, so a whole dataset push goes through it.

    ``$TMPDIR`` is the only place left to rename in, which is what the
    recorded attempts pin: beside the source first, the default location
    second. Falling back does reintroduce the tmpfs overflow the ``dir``
    argument exists to avoid, so it has to stay a fallback rather than
    become the default --
    ``test_mlflow_transfers_are_staged_beside_their_endpoint`` pins the
    writable case that must keep bypassing it.
    """
    experiment_id, run_id = mlflow_run
    staging_dirs: list[Path | None] = []

    class RecordingTempfile:
        """Records where ``filesystem`` asks for its staging dirs."""

        @staticmethod
        def TemporaryDirectory(*args, **kwargs) -> tempfile.TemporaryDirectory:
            staging_dirs.append(kwargs.get("dir"))
            return tempfile.TemporaryDirectory(*args, **kwargs)

    monkeypatch.setattr(filesystem, "tempfile", RecordingTempfile)

    source_dir = tempdir / "read-only-source"
    source_dir.mkdir()
    local_file = source_dir / "source.txt"
    local_file.write_text("read-only payload")

    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )
    source_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        fs.put_file(local_file, "read-only/renamed.txt")
    finally:
        # Restored before `tempdir` is torn down, which cannot remove a
        # directory it is not allowed to write to.
        source_dir.chmod(stat.S_IRWXU)

    assert staging_dirs == [source_dir, None]
    assert fs.read_text("read-only/renamed.txt") == "read-only payload"


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
    # NOTE: This used to assert that `put_file(local_file, "")` raises
    # `ValueError("No relative artifact path specified.")`. The run's
    # artifact root is a perfectly valid destination, and it is the one
    # `split_full_path("mlflow://<experiment>/<run>")` produces, so
    # refusing it broke every `LuxonisFileSystem.upload` to a run URL.
    # `test_mlflow_put_file_to_run_root` covers the behaviour.
    assert fs.put_file(local_file, "") == (
        f"mlflow://{experiment_id}/{run_id}/{local_file.name}"
    )

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
    # NOTE: This used to assert `_get_mlflow_url("fallback.txt") ==
    # "mlflow:///fallback.txt"` -- an unparsable URL that
    # `LuxonisFileSystem.download` reads back as
    # experiment="fallback.txt", run_id=None. Refusing to mint it is the
    # only correct answer when no run is active; the happy path is
    # covered by `test_mlflow_active_run_url_round_trips`.
    with pytest.raises(ValueError, match="No active MLflow run"):
        active_fs._get_mlflow_url("fallback.txt")
    # NOTE: This used to assert a `ValueError("Reading to byte buffer
    # not available for active mlflow runs.")`. Reads on active runs
    # work now, so the only genuine error left is the missing artifact
    # path -- the same one the explicit-run filesystem raises above.
    with pytest.raises(ValueError, match="No relative artifact path"):
        active_fs.read_to_byte_buffer()
    with pytest.raises(ValueError, match="No active mlflow_instance"):
        active_fs.put_file(local_file, "artifact.txt")
    with pytest.raises(ValueError, match="No active mlflow_instance"):
        active_fs.put_dir(local_dir, "artifacts")


def test_mlflow_put_file_to_run_root(
    tempdir: Path,
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    """Regression test for uploads to a run's artifact root.

    ``put_file`` required a non-empty artifact path and raised
    ``ValueError("No relative artifact path specified.")`` otherwise.
    But ``split_full_path("mlflow://<experiment>/<run>")`` returns an
    empty remote path by design, so ``LuxonisFileSystem.upload(file,
    "mlflow://<experiment>/<run>")`` -- the most natural way to attach a
    file to a run, and what ``LuxonisTracker`` builds on -- could never
    work. The file belongs at the run's artifact root under its own base
    name, which is what MLflow's own ``log_artifact`` does.

    The returned URL is fed straight back into ``download`` because a
    URL that cannot be resolved again is no better than no URL at all.
    """
    experiment_id, run_id = mlflow_run
    uploaded = tempdir / "uploaded.txt"
    uploaded.write_text("uploaded payload")
    attached = tempdir / "attached.txt"
    attached.write_text("attached payload")

    run_url = f"mlflow://{experiment_id}/{run_id}"
    LuxonisFileSystem.upload(
        uploaded, run_url, tracking_uri=mlflow_tracking_uri
    )

    fs = LuxonisFileSystem(run_url, tracking_uri=mlflow_tracking_uri)
    assert set(fs.walk_dir("", recursive=True)) == {uploaded.name}
    assert fs.read_text(uploaded.name) == "uploaded payload"

    url = fs.put_file(attached, "")
    assert url == f"{run_url}/{attached.name}"
    downloaded = LuxonisFileSystem.download(
        url, tempdir / "root-download", tracking_uri=mlflow_tracking_uri
    )
    assert downloaded.read_text() == "attached payload"


def test_mlflow_root_is_directory_checks_the_run(
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    """Regression test for the run root being a directory by fiat.

    ``is_directory`` short-circuited the run root with a hard-coded
    ``return True`` and never contacted the tracking server, so a
    nonexistent run reported its root as a directory while ``exists("")``
    on the very same object reported ``False`` -- two predicates
    disagreeing about one path. It also mattered downstream:
    ``LuxonisFileSystem.download`` branches on ``is_directory``, so a bad
    run was routed into ``get_dir`` and failed deep inside
    ``download_artifacts`` instead of at the lookup.

    Raising for the missing run is the intended behaviour and matches
    the ``fsspec`` branch, where ``info`` raises ``FileNotFoundError``.
    """
    from mlflow.exceptions import MlflowException

    experiment_id, run_id = mlflow_run
    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )
    assert fs.is_directory("")
    assert fs.exists("")

    missing_run_fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/missing-run",
        tracking_uri=mlflow_tracking_uri,
    )
    assert not missing_run_fs.exists("")
    with pytest.raises(MlflowException, match="missing-run"):
        missing_run_fs.is_directory("")


def test_mlflow_exists_only_swallows_missing_resources(
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test for ``exists`` reporting outages as absence.

    The MLflow branch of ``exists`` wrapped the whole lookup in a bare
    ``except Exception: return False``, so a connection failure, an
    expired token or a 500 from a remote tracking server was
    indistinguishable from a missing artifact. That turns an
    ``if not fs.exists(p): fs.put_file(p)`` guard into an endless
    re-upload and makes a real outage look like missing data.

    Only MLflow's ``RESOURCE_DOES_NOT_EXIST`` answer -- what the server
    returns for a missing run or artifact -- may become ``False``.
    """
    from mlflow.exceptions import MlflowException, RestException

    experiment_id, run_id = mlflow_run
    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )

    # A genuine "not found" must still be reported as `False`, both for
    # a missing artifact under a live run and for a missing run.
    assert not fs.exists("definitely/missing.txt")
    missing_run_fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/missing-run",
        tracking_uri=mlflow_tracking_uri,
    )
    assert not missing_run_fs.exists()
    assert not missing_run_fs.exists("some/artifact.txt")

    class BrokenClient:
        """Client whose every call fails the way an outage would."""

        def __init__(self, error: Exception):
            self.error = error

        def get_run(self, *_args, **_kwargs) -> NoReturn:
            raise self.error

        def list_artifacts(self, *_args, **_kwargs) -> NoReturn:
            raise self.error

    # MLflow wraps `requests` failures into a plain `MlflowException`
    # with the default `INTERNAL_ERROR` code, and reports auth failures
    # as a `RestException` with the server's error code.
    transport_error = MlflowException(
        f"API request to {mlflow_tracking_uri}/api/2.0/mlflow/runs/get "
        "failed with exception ConnectionError"
    )
    assert transport_error.error_code == "INTERNAL_ERROR"
    auth_error = RestException(
        {"error_code": "PERMISSION_DENIED", "message": "token expired"}
    )

    for error in (transport_error, auth_error):
        monkeypatch.setattr(
            fs, "_get_mlflow_client", lambda error=error: BrokenClient(error)
        )
        # The run root goes through `get_run`, a nested path through
        # `list_artifacts` -- neither may be reported as "not there".
        with pytest.raises(MlflowException) as root_error:
            fs.exists()
        assert root_error.value is error
        with pytest.raises(MlflowException) as artifact_error:
            fs.exists("some/artifact.txt")
        assert artifact_error.value is error


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
    experiment_id = mlflow.create_experiment(f"active-fs-test-{randint}")
    local_file = tempdir / "active-source.txt"
    local_file.write_text("active run payload")
    local_dir = tempdir / "active-dir-source"
    local_dir.mkdir()
    (local_dir / "active-dir-file.txt").write_text("active dir payload")
    iterable_files = []
    for index in range(2):
        file = tempdir / f"active-iterable-{index}.txt"
        file.write_text(f"active iterable payload {index}")
        iterable_files.append(file)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        fs = LuxonisFileSystem(
            "mlflow://",
            allow_active_mlflow_run=True,
            allow_local=False,
            tracking_uri=mlflow_tracking_uri,
        )
        assert (
            fs.put_file(
                local_file, "active/renamed.txt", mlflow_instance=mlflow
            )
            == f"mlflow://{experiment_id}/{run.info.run_id}/active/renamed.txt"
        )
        fs.put_dir(local_dir, "active-dir", mlflow_instance=mlflow)
        assert fs.put_dir(
            iterable_files, "active-iterable", mlflow_instance=mlflow
        ) == {
            str(file): f"active-iterable/{file.name}"
            for file in iterable_files
        }

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
        for file in iterable_files:
            assert (
                explicit_fs.read_text(f"active-iterable/{file.name}")
                == file.read_text()
            )


def test_mlflow_active_run_read_round_trip(
    tempdir: Path,
    mlflow_tracking_uri: str,
    randint: int,
):
    """Regression test for reads never working on active-run filesystems.

    Writes handled the active run (``put_file``/``put_dir`` delegate to
    the passed ``mlflow_instance``), but every read funnelled through
    ``self._run_id``, which is ``None`` in active-run mode. So
    ``walk_dir``, ``get_file``, ``get_dir``, ``is_directory``, ``exists``
    and ``read_to_byte_buffer`` all died with ``ValueError: `run_id`
    cannot be `None` when using `mlflow` `` on the very object whose
    ``put_file`` had just succeeded -- a filesystem you can write to but
    never read back.

    The reads deliberately take no ``mlflow_instance``:
    ``mlflow.active_run()`` is process-global, so resolution has to work
    off the ambient run alone.
    """
    experiment_id = mlflow.create_experiment(f"active-read-test-{randint}")
    local_file = tempdir / "active-read-source.txt"
    local_file.write_text("active run read payload")

    with mlflow.start_run(experiment_id=experiment_id):
        fs = LuxonisFileSystem(
            "mlflow://",
            allow_active_mlflow_run=True,
            allow_local=False,
            tracking_uri=mlflow_tracking_uri,
        )
        fs.put_file(
            local_file, "active-read/renamed.txt", mlflow_instance=mlflow
        )

        assert fs.exists("active-read/renamed.txt")
        assert not fs.exists("active-read/missing.txt")
        assert fs.exists("")
        assert fs.is_directory("")
        assert fs.is_directory("active-read")
        assert not fs.is_directory("active-read/renamed.txt")
        assert set(fs.walk_dir("", recursive=True)) == {
            "active-read/renamed.txt"
        }
        assert fs.read_text("active-read/renamed.txt") == (
            "active run read payload"
        )
        assert fs.read_to_byte_buffer("active-read/renamed.txt").read() == (
            local_file.read_bytes()
        )

        downloaded_file = fs.get_file(
            "active-read/renamed.txt", tempdir / "active-read-download.txt"
        )
        assert downloaded_file.read_text() == local_file.read_text()

        downloaded_dir = fs.get_dir("active-read", tempdir / "active-read-dir")
        assert (downloaded_dir / "renamed.txt").read_text() == (
            local_file.read_text()
        )


def test_mlflow_active_run_url_round_trips(
    tempdir: Path,
    mlflow_tracking_uri: str,
    randint: int,
):
    """Regression test for the malformed ``mlflow:///...`` URL.

    In active-run mode ``_get_mlflow_url`` only looked at an injected
    ``mlflow_instance``; without one it fell back to a bare
    ``"mlflow://"`` base and returned e.g. ``"mlflow:///fallback.txt"``.
    That string cannot be parsed back -- ``split_full_path`` reads the
    artifact name as the experiment and leaves ``run_id`` unset -- so
    feeding the URL a caller was just handed back into
    ``LuxonisFileSystem.download`` fails.

    The active run object carries both IDs, so the URL can always name
    the real experiment and run. Asserting the ``download`` round trip
    (rather than only the string) is the point: it proves the URL is
    usable, not merely well-formed.
    """
    experiment_id = mlflow.create_experiment(f"active-url-test-{randint}")
    local_file = tempdir / "active-url-source.txt"
    local_file.write_text("active run url payload")

    with mlflow.start_run(experiment_id=experiment_id) as run:
        fs = LuxonisFileSystem(
            "mlflow://",
            allow_active_mlflow_run=True,
            allow_local=False,
            tracking_uri=mlflow_tracking_uri,
        )
        fs.put_file(
            local_file, "active-url/renamed.txt", mlflow_instance=mlflow
        )

        run_url = f"mlflow://{experiment_id}/{run.info.run_id}"
        url = fs._get_mlflow_url("active-url/renamed.txt")
        assert url == f"{run_url}/active-url/renamed.txt"
        assert LuxonisFileSystem.split_full_path(url) == (
            run_url,
            "active-url/renamed.txt",
        )

        downloaded = LuxonisFileSystem.download(
            url,
            tempdir / "active-url-download.txt",
            tracking_uri=mlflow_tracking_uri,
        )
        assert downloaded.read_text() == local_file.read_text()


def test_mlflow_delete_is_unsupported(
    mlflow_tracking_uri: str,
    mlflow_run: tuple[str, str],
):
    """Pins the MLflow guard the delete methods start with.

    Each method rejects MLflow up front and then goes straight into the
    ``fsspec`` branch. That guard is the only thing between an MLflow
    filesystem and ``self._fs``, which such an instance never gets, so
    losing it turns a clear "MLflow artifacts cannot be deleted" into an
    ``AttributeError`` from the internals.

    It used to be followed by an ``if self.is_fsspec: ... else: raise
    NotImplementedError`` whose ``else`` could not be reached --
    ``FSType`` has exactly two members and the MLflow one has already
    returned by then -- which read as if a third backend existed.
    Deleting that branch is only safe while these three assertions hold;
    the ``fsspec`` half is covered by the delete cases in
    ``test_filesystem.py``.
    """
    experiment_id, run_id = mlflow_run
    fs = LuxonisFileSystem(
        f"mlflow://{experiment_id}/{run_id}",
        tracking_uri=mlflow_tracking_uri,
    )

    message = "^MLflow artifacts cannot be deleted\\.$"
    with pytest.raises(NotImplementedError, match=message):
        fs.delete_file("artifact.txt")
    with pytest.raises(NotImplementedError, match=message):
        fs.delete_files(["artifact.txt"])
    with pytest.raises(NotImplementedError, match=message):
        fs.delete_dir("artifacts")


def test_tracker_upload_artifact_to_mlflow(
    tempdir: Path,
    mlflow_tracking_uri: str,
    randint: int,
):
    """Regression test for artifacts being nested under local paths.

    ``luxonis-train`` uploads exported files with
    ``tracker.upload_artifact(f.name, name=f.name, typ="export")``,
    where ``f`` is an open file object, so ``name`` is a *full local
    path* such as ``output/<run-name>/export/model.yaml``. Interpolating
    it into the ``mlflow://`` URL stored the artifact under that whole
    directory tree, leaking the local layout into the artifact store.
    Only the base name may be used, so the artifact always lands at the
    run's artifact root -- hence the assertion on the full artifact
    listing rather than a plain ``exists`` check.
    """
    from luxonis_ml.tracker import LuxonisTracker

    experiment_id = mlflow.create_experiment(f"tracker-test-{randint}")
    export_dir = tempdir / "export" / f"tracker-run-{randint}"
    export_dir.mkdir(parents=True)
    artifact = export_dir / "model.yaml"
    artifact.write_text("tracker payload")

    tracker = LuxonisTracker(
        project_id=experiment_id,
        run_name=f"tracker-run-{randint}",
        save_directory=tempdir / "output",
        is_mlflow=True,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    with artifact.open() as file:
        tracker.upload_artifact(file.name, name=file.name)

    assert tracker.project_id is not None
    assert tracker.run_id is not None

    fs = LuxonisFileSystem(
        f"mlflow://{tracker.project_id}/{tracker.run_id}",
        tracking_uri=mlflow_tracking_uri,
    )
    assert set(fs.walk_dir("", recursive=True)) == {"model.yaml"}
    assert fs.read_text("model.yaml") == "tracker payload"

    tracker.upload_artifact(artifact)
    assert set(fs.walk_dir("", recursive=True)) == {"model.yaml"}

    mlflow_module = tracker.experiment.get("mlflow")
    assert mlflow_module is not None
    mlflow_module.end_run()
