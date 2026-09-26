import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import wandb

from luxonis_ml.tracker import RunContext, WandbBackend


class RunSpy:
    """Record what the backend hands to the methods of a WandB run."""

    def __init__(self, run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.logged: list[dict[str, Any]] = []
        self.artifacts: list[Any] = []
        self.exit_codes: list[int | None] = []
        monkeypatch.setattr(run, "log", self.logged.append)
        monkeypatch.setattr(run, "log_artifact", self.artifacts.append)
        monkeypatch.setattr(
            run,
            "finish",
            lambda exit_code=None: self.exit_codes.append(exit_code),
        )


@pytest.fixture
def init_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Run WandB in disabled mode, which needs no account, and record
    the arguments of `wandb.init`.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    calls: list[dict[str, Any]] = []
    init = wandb.init

    def recording_init(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return init(**kwargs)

    monkeypatch.setattr(wandb, "init", recording_init)
    return calls


@pytest.fixture
def backend(
    run: RunContext, init_kwargs: list[dict[str, Any]]
) -> Iterator[WandbBackend]:
    backend = WandbBackend(run, entity="team")
    backend.start()
    yield backend
    # `RunSpy` may still replace `finish` on the instance
    type(backend.experiment).finish(backend.experiment)


@pytest.fixture
def spy(backend: WandbBackend, monkeypatch: pytest.MonkeyPatch) -> RunSpy:
    return RunSpy(backend.experiment, monkeypatch)


def test_start_opens_a_run_in_the_project(
    backend: WandbBackend, run: RunContext, init_kwargs: list[dict[str, Any]]
):
    assert init_kwargs == [
        {
            "project": "project",
            "entity": "team",
            "dir": run.save_directory / "wandb_logs",
            "name": run.run_name,
        }
    ]
    assert (run.save_directory / "wandb_logs").is_dir()


def test_the_project_id_names_a_project_without_a_name(tmp_path: Path):
    run = RunContext("0-test", tmp_path, project_id="1234")

    assert WandbBackend(run).project == "1234"


def test_a_project_is_required(tmp_path: Path):
    with pytest.raises(ValueError, match="project_name"):
        WandbBackend(RunContext("0-test", tmp_path))


def test_hyperparameters_go_to_the_config(backend: WandbBackend):
    backend.log_hyperparams({"lr": 0.1, "layers": [1, 2]})

    assert backend.experiment.config["lr"] == 0.1
    assert backend.experiment.config["layers"] == [1, 2]


def test_calls_leave_the_step_to_wandb(
    backend: WandbBackend, spy: RunSpy, image: npt.NDArray[np.uint8]
):
    backend.log_metrics({"loss": 0.5}, 7)
    backend.log_image("val/image", image, 8)
    backend.log_matrix(np.array([[1, 2], [3, 4]]), "matrix", 9, {})
    backend.log_matrix(np.array([5, 6, 7]), "vector", 9, {})

    assert spy.logged[0] == {"loss": 0.5}
    wandb_image = spy.logged[1]["val/image"]
    assert isinstance(wandb_image, wandb.Image)
    assert wandb_image._caption == "val/image"
    table = spy.logged[2]["matrix_table"]
    assert table.columns == ["Row Index", "Col 0", "Col 1"]
    assert table.data == [[0, 1, 2], [1, 3, 4]]
    assert spy.logged[3]["vector_table"].data == [[0, 5, 6, 7]]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, "model"),
        ("final.onnx", "final.onnx"),
        ("output/run/export/final.onnx", "final.onnx"),
    ],
)
def test_an_artifact_takes_a_valid_name(
    backend: WandbBackend,
    spy: RunSpy,
    tmp_path: Path,
    name: str | None,
    expected: str,
):
    path = tmp_path / "model.onnx"
    path.write_text("weights")

    backend.upload_artifact(path, name, "export")

    (artifact,) = spy.artifacts
    assert artifact.name == expected
    assert artifact.type == "export"
    assert list(artifact.manifest.entries) == ["model.onnx"]


@pytest.mark.parametrize(
    ("status", "exit_code"), [("success", 0), ("failed", 1)]
)
def test_close_finishes_the_run(
    backend: WandbBackend, spy: RunSpy, status: Any, exit_code: int
):
    backend.close(status)

    assert spy.exit_codes == [exit_code]


def test_the_run_exists_only_after_start(run: RunContext):
    with pytest.raises(RuntimeError, match="not started"):
        _ = WandbBackend(run).experiment


def test_a_missing_sdk_names_the_extra(
    run: RunContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ImportError, match=r"luxonis-ml\[wandb\]"):
        WandbBackend(run).start()
