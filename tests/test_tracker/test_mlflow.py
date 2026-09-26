import json
import socket
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    REQUEST_LIMIT_EXCEEDED,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mlflow_tags import (
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_SOURCE_NAME,
    MLFLOW_USER,
)

import luxonis_ml.tracker.backends.mlflow as mlflow_module
from luxonis_ml.tracker import LuxonisTracker, MLflowBackend, RunContext


def http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"status {status}", response=response)


@pytest.fixture(scope="module")
def mlflow_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("mlflow")


@pytest.fixture
def tracking_uri(mlflow_dir: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Return a local sqlite store.

    MLflow puts the artifacts of a new experiment under the working
    directory, so the test runs in the store directory.
    """
    monkeypatch.chdir(mlflow_dir)
    return f"sqlite:///{mlflow_dir / 'mlflow.db'}"


@pytest.fixture
def client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri)


@pytest.fixture
def project(request: pytest.FixtureRequest) -> str:
    """Name a project of its own for each test, as the store is
    shared.
    """
    return request.node.name


def make_run(tmp_path: Path, project: str | None, **kwargs: Any) -> RunContext:
    return RunContext("0-test", tmp_path, project_name=project, **kwargs)


@pytest.fixture
def backend(tmp_path: Path, project: str, tracking_uri: str) -> MLflowBackend:
    backend = MLflowBackend(
        make_run(tmp_path, project), tracking_uri=tracking_uri
    )
    backend.start()
    return backend


def test_the_options_are_checked(
    tmp_path: Path, tracking_uri: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        MLflowBackend(make_run(tmp_path, "project"))
    with pytest.raises(ValueError, match="project_name"):
        MLflowBackend(make_run(tmp_path, None), tracking_uri=tracking_uri)


def test_the_tracking_uri_defaults_to_the_environment(
    tmp_path: Path, tracking_uri: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    tracker = LuxonisTracker(
        project_name="project", save_directory=tmp_path, mlflow=True
    )

    backend = tracker.get_backend(MLflowBackend)
    assert backend.tracking_uri == tracking_uri
    assert backend.parent_run_id is None


def test_start_creates_the_experiment_and_the_run(
    backend: MLflowBackend, client: MlflowClient, project: str
):
    experiment = client.get_experiment_by_name(project)
    assert experiment is not None
    assert backend.experiment_id == experiment.experiment_id
    assert backend.run_id is not None
    run = client.get_run(backend.run_id)
    assert run.info.run_name == "0-test"
    assert run.info.status == "RUNNING"
    assert MLFLOW_PARENT_RUN_ID not in run.data.tags
    # the context tags that `mlflow.start_run` adds as well
    assert run.data.tags[MLFLOW_USER] == run.info.user_id != "unknown"
    assert MLFLOW_SOURCE_NAME in run.data.tags
    assert mlflow_module._open_runs == [backend.run_id]


def test_start_reuses_an_experiment_of_the_same_name(
    tmp_path: Path, project: str, tracking_uri: str, client: MlflowClient
):
    experiment_id = client.create_experiment(project)
    backend = MLflowBackend(
        make_run(tmp_path, project), tracking_uri=tracking_uri
    )

    backend.start()

    assert backend.experiment_id == experiment_id


def test_the_project_id_selects_the_experiment(
    tmp_path: Path, project: str, tracking_uri: str, client: MlflowClient
):
    experiment_id = client.create_experiment(project)
    run = make_run(tmp_path, "unused name", project_id=experiment_id)
    backend = MLflowBackend(run, tracking_uri=tracking_uri)

    backend.start()

    assert backend.experiment_id == experiment_id
    assert client.get_experiment_by_name("unused name") is None


def test_a_run_id_continues_the_run(
    tmp_path: Path, project: str, tracking_uri: str, client: MlflowClient
):
    experiment_id = client.create_experiment(project)
    run_id = client.create_run(experiment_id).info.run_id
    client.set_terminated(run_id, "FAILED")
    run = make_run(tmp_path, project, project_id=experiment_id, run_id=run_id)
    backend = MLflowBackend(run, tracking_uri=tracking_uri)

    backend.start()
    assert client.get_run(run_id).info.status == "RUNNING"
    backend.close("success")

    assert client.get_run(run_id).info.status == "FINISHED"
    assert len(client.search_runs([experiment_id])) == 1


def test_the_retry_count_is_only_a_default(
    backend: MLflowBackend, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "5")
    backend.start()
    assert mlflow_module.os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "5"

    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES")
    backend.start()
    assert mlflow_module.os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "2"


def test_a_sweep_trial_nests_under_the_open_run(
    backend: MLflowBackend,
    tmp_path: Path,
    project: str,
    tracking_uri: str,
    client: MlflowClient,
):
    sweep_run = make_run(tmp_path, project, is_sweep=True)
    trial = MLflowBackend(sweep_run, tracking_uri=tracking_uri)
    trial.start()

    assert trial.run_id is not None
    tags = client.get_run(trial.run_id).data.tags
    assert tags[MLFLOW_PARENT_RUN_ID] == backend.run_id
    assert mlflow_module._open_runs == [backend.run_id]

    backend.close("success")
    trial.close("success")
    late_trial = MLflowBackend(sweep_run, tracking_uri=tracking_uri)
    late_trial.start()

    assert late_trial.run_id is not None
    tags = client.get_run(late_trial.run_id).data.tags
    assert MLFLOW_PARENT_RUN_ID not in tags
    assert mlflow_module._open_runs == []


def test_an_explicit_parent_wins(
    backend: MLflowBackend,
    tmp_path: Path,
    project: str,
    tracking_uri: str,
    client: MlflowClient,
):
    assert backend.experiment_id is not None
    parent = client.create_run(backend.experiment_id).info.run_id
    trial = MLflowBackend(
        make_run(tmp_path, project, is_sweep=True),
        tracking_uri=tracking_uri,
        parent_run_id=parent,
    )

    trial.start()

    assert trial.run_id is not None
    tags = client.get_run(trial.run_id).data.tags
    assert tags[MLFLOW_PARENT_RUN_ID] == parent


def artifact_root(client: MlflowClient, run_id: str) -> Path:
    uri = client.get_run(run_id).info.artifact_uri
    assert uri is not None
    # a plain prefix cut breaks `file:///C:/...` on Windows
    return Path(local_file_uri_to_path(uri))


def test_params_and_metrics_reach_the_run(
    backend: MLflowBackend, client: MlflowClient
):
    backend.log_hyperparams({"lr": 0.1, "note": None})
    backend.log_metrics({"loss": 0.5}, 1)
    backend.log_metrics({"loss": 0.25}, 2)
    backend.close("success")

    assert backend.run_id is not None
    run = client.get_run(backend.run_id)
    assert run.data.params == {"lr": "0.1", "note": "None"}
    history = client.get_metric_history(backend.run_id, "loss")
    assert [(m.step, m.value) for m in history] == [(1, 0.5), (2, 0.25)]
    assert run.info.status == "FINISHED"


def test_an_image_goes_under_its_step(
    backend: MLflowBackend,
    client: MlflowClient,
    image: npt.NDArray[np.uint8],
):
    backend.log_image("val/image", image, 3)
    backend.log_image("plain", image, 4)

    assert backend.run_id is not None
    artifacts = artifact_root(client, backend.run_id)
    assert (artifacts / "val" / "3" / "image.png").is_file()
    assert (artifacts / "4" / "plain.png").is_file()


def test_a_matrix_goes_to_a_json_file(
    backend: MLflowBackend, client: MlflowClient
):
    backend.log_matrix(np.eye(2), "matrix", 5, {"labels": ["a", "b"]})

    assert backend.run_id is not None
    matrix_file = artifact_root(client, backend.run_id) / "matrix.json"
    assert json.loads(matrix_file.read_text()) == {
        "flat_array": [1.0, 0.0, 0.0, 1.0],
        "shape": [2, 2],
        "labels": ["a", "b"],
    }


@pytest.mark.parametrize(
    ("name", "stored_as"),
    [(None, "model.txt"), ("output/export/final.txt", "final.txt")],
)
def test_an_artifact_goes_to_the_artifact_root(
    backend: MLflowBackend,
    client: MlflowClient,
    tmp_path: Path,
    name: str | None,
    stored_as: str,
):
    artifact = tmp_path / "model.txt"
    artifact.write_text("weights")

    backend.upload_artifact(artifact, name, "weights")

    assert backend.run_id is not None
    artifacts = artifact_root(client, backend.run_id)
    assert [path.name for path in artifacts.iterdir()] == [stored_as]
    assert (artifacts / stored_as).read_text() == "weights"


def test_a_run_that_fails_to_close_is_no_longer_a_parent(
    backend: MLflowBackend, monkeypatch: pytest.MonkeyPatch
):
    def unreachable(*_: object) -> None:
        raise MlflowException("connection refused")

    monkeypatch.setattr(backend.experiment, "set_terminated", unreachable)

    with pytest.raises(MlflowException):
        backend.close("success")

    assert mlflow_module._open_runs == []


def test_a_failed_run_is_marked_failed(
    backend: MLflowBackend, client: MlflowClient
):
    backend.close("failed")

    assert backend.run_id is not None
    assert client.get_run(backend.run_id).info.status == "FAILED"
    assert mlflow_module._open_runs == []


def test_the_client_exists_only_after_start(tmp_path: Path):
    """The client opens a local store at once, which the constructor
    must not do on every rank.
    """
    store = tmp_path / "fresh.db"
    backend = MLflowBackend(
        make_run(tmp_path, "project"), tracking_uri=f"sqlite:///{store}"
    )

    with pytest.raises(RuntimeError, match="not started"):
        _ = backend.experiment
    assert not store.exists()


def test_a_run_in_a_missing_experiment_is_rejected(
    tmp_path: Path, tracking_uri: str
):
    run = make_run(tmp_path, "project", project_id="999999")
    backend = MLflowBackend(run, tracking_uri=tracking_uri)

    with pytest.raises(MlflowException) as error:
        backend.start()
    assert not backend.is_transient(error.value)
    with pytest.raises(RuntimeError, match="not started"):
        backend.log_metrics({"loss": 0.5}, 1)


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (MlflowException("bad value", INVALID_PARAMETER_VALUE), False),
        (MlflowException("no such run", RESOURCE_DOES_NOT_EXIST), False),
        (MlflowException("slow down", REQUEST_LIMIT_EXCEEDED), True),
        (MlflowException("connection refused"), True),
        (MlflowException("server error", INTERNAL_ERROR), True),
        (FileNotFoundError("model.txt"), False),
        (ConnectionError("reset"), True),
        (requests.ConnectionError("refused"), True),
        (http_error(503), True),
        (http_error(429), True),
        (http_error(413), False),
        (requests.HTTPError("no response"), True),
        (ValueError("pixel values out of range"), False),
        (TypeError("unsupported dtype"), False),
    ],
)
def test_an_outage_is_told_from_a_rejected_call(
    tmp_path: Path, tracking_uri: str, error: Exception, transient: bool
):
    backend = MLflowBackend(
        make_run(tmp_path, "project"), tracking_uri=tracking_uri
    )

    assert backend.is_transient(error) is transient


class FakeMonitor:
    instances: list["FakeMonitor"] = []
    error: Exception | None = None

    def __init__(
        self, run_id: str, resume_logging: bool, tracking_uri: str
    ) -> None:
        if self.error is not None:
            raise self.error
        self.run_id = run_id
        self.resume_logging = resume_logging
        self.tracking_uri = tracking_uri
        self.running = False
        FakeMonitor.instances.append(self)

    def start(self) -> None:
        self.running = True

    def finish(self) -> None:
        self.running = False


@pytest.fixture
def monitor(monkeypatch: pytest.MonkeyPatch) -> type[FakeMonitor]:
    """Pretend that ``psutil`` is installed, which MLflow needs to
    monitor the system.
    """
    module = ModuleType("system_metrics_monitor")
    module.SystemMetricsMonitor = FakeMonitor  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "mlflow.system_metrics.system_metrics_monitor", module
    )
    monkeypatch.setattr(mlflow_module, "find_spec", lambda _: object())
    monkeypatch.setattr(FakeMonitor, "instances", [])
    monkeypatch.setattr(FakeMonitor, "error", None)
    return FakeMonitor


def test_system_metrics_follow_the_run(
    monitor: type[FakeMonitor], backend: MLflowBackend, tracking_uri: str
):
    (instance,) = monitor.instances
    assert instance.run_id == backend.run_id
    assert instance.tracking_uri == tracking_uri
    assert not instance.resume_logging
    assert instance.running

    backend.close("success")

    assert not instance.running


def test_a_broken_monitor_does_not_stop_the_run(
    monitor: type[FakeMonitor],
    tmp_path: Path,
    tracking_uri: str,
    warnings_log: list[str],
):
    monitor.error = RuntimeError("no sensors")
    backend = MLflowBackend(
        make_run(tmp_path, "project"), tracking_uri=tracking_uri
    )

    backend.start()
    backend.close("success")

    assert any("no sensors" in m for m in warnings_log)


def test_system_metrics_need_psutil(
    backend: MLflowBackend,
    monkeypatch: pytest.MonkeyPatch,
    warnings_log: list[str],
):
    monkeypatch.setattr(mlflow_module, "find_spec", lambda _: None)

    backend.start()

    assert any("Install `psutil`" in m for m in warnings_log)


def test_a_missing_sdk_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "mlflow", None)

    backend = MLflowBackend(make_run(tmp_path, "project"), tracking_uri="uri")

    with pytest.raises(ImportError, match=r"luxonis-ml\[mlflow\]"):
        backend.start()


@pytest.fixture
def unreachable_uri(monkeypatch: pytest.MonkeyPatch) -> str:
    """Return a tracking URI on a port that nothing listens on."""
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_an_unreachable_server_does_not_stop_the_training(
    tmp_path: Path,
    unreachable_uri: str,
    image: npt.NDArray[np.uint8],
    warnings_log: list[str],
):
    with LuxonisTracker(
        project_name="project",
        run_name="0-test",
        save_directory=tmp_path,
        mlflow={"tracking_uri": unreachable_uri},
    ) as tracker:
        tracker.log_hyperparams({"lr": 0.1})
        tracker.log_metric("loss", 0.5, 1)
        tracker.log_image("val/image", image, 1)

    path = tmp_path / "0-test" / "unsent_logs" / "mlflow" / "calls.jsonl"
    calls = [
        json.loads(line)["call"] for line in path.read_text().splitlines()
    ]
    assert calls == ["log_hyperparams", "log_metrics", "log_image"]
    assert any("mlflow is unavailable" in m for m in warnings_log)
