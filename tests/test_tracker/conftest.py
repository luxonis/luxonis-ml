from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import luxonis_ml.tracker.backends.mlflow as mlflow_module
from luxonis_ml.tracker import (
    TRACKER_BACKENDS,
    RunContext,
    RunStatus,
    TrackerBackend,
)
from luxonis_ml.tracker.tracker import RUN_NAME_ENV
from luxonis_ml.typing import ParamValue

MAX_RETRIES_ENV = "MLFLOW_HTTP_REQUEST_MAX_RETRIES"


class Rejected(Exception):
    """An error that `FakeBackend.is_transient` does not retry."""


class FakeBackend(TrackerBackend):
    """Record each call. While `error` is set, `start` and every
    logging call raise it.
    """

    def __init__(self, run: RunContext, *, option: str = "default") -> None:
        super().__init__(run)
        self.option = option
        self.calls: list[tuple[Any, ...]] = []
        self.starts = 0
        self.status: RunStatus | None = None
        self.error: Exception | None = None

    @property
    def experiment(self) -> "FakeBackend":
        return self

    def start(self) -> None:
        self._raise()
        self.starts += 1

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        self._record("log_hyperparams", dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self._record("log_metrics", dict(metrics), step)

    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        self._record("log_image", name, image, step)

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        self._record("log_matrix", matrix, name, step, dict(extra_data))

    def upload_artifact(self, path: Path, name: str | None, typ: str) -> None:
        self._raise()
        self._record("upload_artifact", path.read_text(), name, typ)

    def close(self, status: RunStatus) -> None:
        self._raise()
        self.status = status

    def is_transient(self, error: Exception) -> bool:
        return not isinstance(error, Rejected)

    def _record(self, *call: Any) -> None:
        self._raise()
        self.calls.append(call)

    def _raise(self) -> None:
        if self.error is not None:
            raise self.error


class BufferedFakeBackend(FakeBackend):
    buffered = True


TRACKER_BACKENDS.register(module=FakeBackend, name="fake", force=True)
TRACKER_BACKENDS.register(module=FakeBackend, name="other_fake", force=True)
TRACKER_BACKENDS.register(
    module=BufferedFakeBackend, name="buffered_fake", force=True
)


@pytest.fixture(autouse=True)
def isolated_process_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo what a tracker leaves in the process: the exported run
    name, the MLflow retry count, and the open MLflow runs.
    """
    monkeypatch.setenv(RUN_NAME_ENV, "")
    # `setenv` records the original state, so that the teardown also
    # removes a value that the MLflow backend sets during the test
    monkeypatch.setenv(MAX_RETRIES_ENV, "")
    monkeypatch.delenv(MAX_RETRIES_ENV)
    monkeypatch.setattr(mlflow_module, "_open_runs", [])


@pytest.fixture
def run(tmp_path: Path) -> RunContext:
    return RunContext(
        run_name="0-test",
        save_directory=tmp_path,
        project_name="project",
    )


@pytest.fixture
def image() -> npt.NDArray[np.uint8]:
    return np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
