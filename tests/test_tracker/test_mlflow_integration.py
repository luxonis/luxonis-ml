"""End-to-end tests against a real MLflow tracking server.

Unlike the rest of the tracker tests, these start an actual
``mlflow server`` subprocess and talk to it over HTTP. The test is
skipped when the server cannot be started so that environments without a
usable MLflow installation do not fail the suite.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import suppress
from pathlib import Path

import mlflow
import numpy as np
import pytest
import requests

from luxonis_ml.tracker import LuxonisTracker

SERVER_STARTUP_TIMEOUT = 120.0
SERVER_SHUTDOWN_TIMEOUT = 60.0
SERVER_POLL_INTERVAL = 0.5


class MlflowServer:
    def __init__(self, uri: str, process: subprocess.Popen):
        self.uri = uri
        self.process = process

    def is_ready(self) -> bool:
        if self.process.poll() is not None:
            return False
        try:
            requests.get(f"{self.uri}/health", timeout=1)
        except requests.RequestException:
            return False
        return True

    def wait_until_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return False
            if self.is_ready():
                return True
            time.sleep(SERVER_POLL_INTERVAL)
        return False

    def stop(self) -> None:
        """Stop the server and wait until the port stops answering.

        ``mlflow server`` serves through a separate worker process, so
        terminating only the process we launched leaves the port open.
        """
        if self.process.poll() is None:
            self._terminate_tree()
            try:
                self.process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self.process.kill()
                self.process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT)

        deadline = time.monotonic() + SERVER_SHUTDOWN_TIMEOUT
        while self.is_ready() and time.monotonic() < deadline:
            time.sleep(SERVER_POLL_INTERVAL)  # pragma: no cover

    def _terminate_tree(self) -> None:
        if sys.platform == "win32":  # pragma: no cover
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],  # noqa: S607
                check=False,
                capture_output=True,
            )
        else:
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def mlflow_server(tempdir: Path) -> Generator[MlflowServer, None, None]:
    store = tempdir / "mlflow"
    store.mkdir(parents=True, exist_ok=True)
    port = _free_port()

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{store / 'mlflow.db'}",
            "--default-artifact-root",
            str(store / "artifacts"),
            "--workers",
            "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # own process group, so the worker can be killed along with it
        start_new_session=True,
    )
    server = MlflowServer(f"http://127.0.0.1:{port}", process)

    if not server.wait_until_ready(SERVER_STARTUP_TIMEOUT):  # pragma: no cover
        server.stop()
        pytest.skip("a local MLflow server could not be started")

    yield server

    server.stop()
    mlflow.disable_system_metrics_logging()
    if mlflow.active_run() is not None:  # pragma: no cover
        mlflow.end_run()


def test_logging_to_a_real_mlflow_server(
    mlflow_server: MlflowServer, tempdir: Path
) -> None:
    image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    matrix = np.arange(6).reshape(2, 3)
    artifact_path = tempdir / "model.onnx"
    artifact_path.write_text("weights")

    with LuxonisTracker(
        project_name="integration-project",
        run_name="integration-run",
        save_directory=tempdir / "output",
        is_mlflow=True,
        mlflow_tracking_uri=mlflow_server.uri,
    ) as tracker:
        tracker.log_hyperparams({"lr": 0.001, "batch_size": 32})
        tracker.log_metric("loss", 0.25, 1)
        tracker.log_metrics({"accuracy": 0.9, "f1": 0.8}, 2)
        tracker.log_image("val/vis", image, 3)
        tracker.log_matrix(matrix, "confusion", 4)
        tracker.upload_artifact(artifact_path)

        run_id = tracker.run_id
        experiment_id = tracker.project_id

    assert not tracker.local_logs
    assert not (tracker.run_directory / "local_logs.json").exists()
    assert run_id is not None
    assert experiment_id is not None

    client = mlflow.MlflowClient(tracking_uri=mlflow_server.uri)
    assert client.get_experiment(experiment_id).name == "integration-project"

    run = client.get_run(run_id)
    assert run.info.run_name == "integration-run"
    assert run.info.status == "FINISHED"
    assert run.data.params == {"lr": "0.001", "batch_size": "32"}
    assert run.data.metrics["loss"] == 0.25
    assert run.data.metrics["accuracy"] == 0.9
    assert run.data.metrics["f1"] == 0.8

    artifacts = {file.path for file in client.list_artifacts(run_id)}
    assert {"model.onnx", "confusion.json"} <= artifacts

    images = {file.path for file in client.list_artifacts(run_id, "val/3")}
    assert "val/3/vis.png" in images


def test_logs_survive_a_server_outage(
    mlflow_server: MlflowServer,
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")

    tracker = LuxonisTracker(
        project_name="outage-project",
        run_name="outage-run",
        save_directory=tempdir / "output",
        is_mlflow=True,
        mlflow_tracking_uri=mlflow_server.uri,
    )
    tracker.log_metric("loss", 1.0, 0)
    assert tracker.mlflow_initialized

    mlflow_server.stop()
    assert not mlflow_server.is_ready()

    tracker.log_metric("loss", 0.5, 1)
    tracker.log_hyperparams({"lr": 0.001})
    assert len(tracker.local_logs) == 2

    # closing must not raise even though the server is gone
    tracker.close()

    saved = json.loads((tracker.run_directory / "local_logs.json").read_text())
    assert saved["metric"] == [{"name": "loss", "value": 0.5, "step": 1}]
    assert saved["params"] == {"lr": 0.001}
