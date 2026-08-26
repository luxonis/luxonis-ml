"""Stand-ins for the tracker's optional logging backends.

The tracker imports `torch`, `wandb`, and `mlflow` lazily, so the fakes
below only need to be registered in `sys.modules` before the backend is
first used. They record every call so the tests can assert on what the
tracker sent, and can be told to fail in order to exercise the fallback
paths.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TypeAlias

import numpy as np
import pytest

from luxonis_ml.tracker.tracker import LogValue, MLflowArg

Scalar: TypeAlias = int | float | np.generic
"""A cell of a logged matrix, which numpy hands over as its own scalar."""

WandbImage: TypeAlias = dict[str, np.ndarray | str]
"""What `FakeWandb.Image` hands back."""


class FakeSummaryWriter:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scalars: list[tuple[str, float, int]] = []
        self.images: list[tuple[str, np.ndarray, int, str]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.hparams: list[dict[str, LogValue]] = []
        self.closed = False
        self.fail_on_close = False

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars.append((name, value, step))

    def add_image(
        self, name: str, img: np.ndarray, step: int, dataformats: str
    ) -> None:
        self.images.append((name, img, step, dataformats))

    def add_text(self, name: str, text: str, step: int) -> None:
        self.texts.append((name, text, step))

    def add_hparams(
        self, hparams: dict[str, LogValue], metrics: dict[str, float]
    ) -> None:
        self.hparams.append(hparams)
        self.hparam_metrics = metrics

    def close(self) -> None:
        if self.fail_on_close:
            raise RuntimeError("the writer is broken")
        self.closed = True


class FakeConfig:
    def __init__(self):
        self.params: dict[str, LogValue] = {}

    def update(self, params: dict[str, LogValue]) -> None:
        self.params.update(params)


class FakeArtifact:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, local_path: str) -> None:
        self.files.append(local_path)


class FakeTable:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.rows: list[tuple[Scalar, ...]] = []

    def add_data(self, *row: Scalar) -> None:
        self.rows.append(row)


# Defined here, because it names `FakeTable`.
WandbLogValue: TypeAlias = float | WandbImage | FakeTable
"""A value the tracker logs through a WandB run."""


class FakeWandbRun:
    def __init__(self, **init_kwargs: str | Path | None):
        self.init_kwargs = init_kwargs
        self.logged: list[dict[str, WandbLogValue]] = []
        self.artifacts: list[FakeArtifact] = []
        self.config = FakeConfig()
        self.exit_code: int | None = None
        self.fail_on_finish = False

    # `log` deliberately takes no `step`; passing one would be an error
    def log(self, data: dict[str, WandbLogValue]) -> None:
        self.logged.append(data)

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self, exit_code: int) -> None:
        if self.fail_on_finish:
            raise RuntimeError("wandb is broken")
        self.exit_code = exit_code


class FakeWandb:
    Table = FakeTable
    Artifact = FakeArtifact

    def __init__(self):
        self.run: FakeWandbRun | None = None
        self.init_attempts = 0
        self.fail_init = False

    def Image(self, img: np.ndarray, caption: str) -> WandbImage:
        return {"img": img, "caption": caption}

    def init(self, **kwargs: str | Path | None) -> FakeWandbRun:
        self.init_attempts += 1
        if self.fail_init:
            raise RuntimeError("wandb is unreachable")
        self.run = FakeWandbRun(**kwargs)
        return self.run


class FakeMlflow:
    def __init__(self):
        self.tracking_uri: str | None = None
        self.system_metrics = False
        self.experiment_kwargs: dict[str, str | None] | None = None
        self.run_kwargs: dict[str, str | bool | None] | None = None
        self.calls: list[tuple[str, tuple[MLflowArg, ...]]] = []
        self.init_attempts = 0
        self.end_run_calls: list[str] = []

        self.fail_init = False
        self.fail_after: int | None = None
        self.fail_end_run = False
        # calls the server rejects for good, whatever else it accepts
        self.reject_kinds: set[str] = set()

    def enable_system_metrics_logging(self) -> None:
        self.system_metrics = True

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(
        self,
        experiment_name: str | None = None,
        experiment_id: str | None = None,
    ) -> SimpleNamespace:
        self.init_attempts += 1
        if self.fail_init:
            raise RuntimeError("mlflow is unreachable")
        self.experiment_kwargs = {
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
        }
        return SimpleNamespace(experiment_id=experiment_id or "exp-1")

    def start_run(
        self,
        run_id: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
    ) -> SimpleNamespace:
        self.run_kwargs = {
            "run_id": run_id,
            "run_name": run_name,
            "nested": nested,
        }
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id or "run-1"),
        )

    def _record(self, name: str, *args: MLflowArg) -> None:
        if name in self.reject_kinds:
            raise RuntimeError(f"mlflow rejects {name}")
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("mlflow is unreachable")
        self.calls.append((name, args))

    def log_metric(self, *args: MLflowArg) -> None:
        self._record("log_metric", *args)

    def log_metrics(self, *args: MLflowArg) -> None:
        self._record("log_metrics", *args)

    def log_params(self, *args: MLflowArg) -> None:
        self._record("log_params", *args)

    def log_image(self, *args: MLflowArg) -> None:
        self._record("log_image", *args)

    def log_dict(self, *args: MLflowArg) -> None:
        self._record("log_dict", *args)

    def log_artifact(self, *args: MLflowArg) -> None:
        self._record("log_artifact", *args)

    def end_run(self, status: str) -> None:
        if self.fail_end_run:
            raise RuntimeError("mlflow is unreachable")
        self.end_run_calls.append(status)

    @property
    def kinds(self) -> list[str]:
        return [name for name, _ in self.calls]


@pytest.fixture
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Register fake TensorBoard, WandB, and MLflow modules."""
    writer = SimpleNamespace(SummaryWriter=FakeSummaryWriter)
    tensorboard = SimpleNamespace(writer=writer)
    utils = SimpleNamespace(tensorboard=tensorboard)
    torch = SimpleNamespace(utils=utils)
    wandb = FakeWandb()
    mlflow = FakeMlflow()

    modules = {
        "torch": torch,
        "torch.utils": utils,
        "torch.utils.tensorboard": tensorboard,
        "torch.utils.tensorboard.writer": writer,
        "wandb": wandb,
        "mlflow": mlflow,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(wandb=wandb, mlflow=mlflow)
