import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from luxonis_ml.tracker import tracker as tracker_module
from luxonis_ml.tracker.tracker import LuxonisTracker

from .conftest import FakeSummaryWriter


def rgb_image() -> np.ndarray:
    return np.full((4, 4, 3), 128, dtype=np.uint8)


@pytest.fixture
def tb_tracker(
    tempdir: Path, fake_backends: SimpleNamespace
) -> LuxonisTracker:
    return LuxonisTracker(
        run_name="run", save_directory=tempdir, is_tensorboard=True
    )


@pytest.fixture
def wandb_tracker(
    tempdir: Path, fake_backends: SimpleNamespace
) -> LuxonisTracker:
    return LuxonisTracker(
        project_name="project",
        run_name="run",
        save_directory=tempdir,
        is_wandb=True,
        wandb_entity="entity",
    )


@pytest.fixture
def mlflow_tracker(
    tempdir: Path, fake_backends: SimpleNamespace
) -> LuxonisTracker:
    return LuxonisTracker(
        project_name="project",
        run_name="run",
        save_directory=tempdir,
        is_mlflow=True,
        mlflow_tracking_uri="http://mlflow",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "At least one integration"),
        ({"is_wandb": True}, "Either project_name or project_id"),
        ({"is_mlflow": True}, "Either project_name or project_id"),
        (
            {"is_wandb": True, "project_name": "project"},
            "Must specify wandb_entity",
        ),
        (
            {"is_mlflow": True, "project_name": "project"},
            "Must specify mlflow_tracking_uri",
        ),
    ],
)
def test_invalid_configuration(
    tempdir: Path, kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LuxonisTracker(save_directory=tempdir, **kwargs)


def test_run_names_are_numbered(tempdir: Path) -> None:
    first = LuxonisTracker(save_directory=tempdir, is_tensorboard=True)
    second = LuxonisTracker(save_directory=tempdir, is_tensorboard=True)

    assert first.run_name.startswith("0-")
    assert second.run_name.startswith("1-")
    assert first.name == first.run_name
    assert (first.version, second.version) == (0, 1)
    assert first.run_directory == tempdir / first.run_name
    assert second.run_directory.is_dir()


def test_explicit_run_name_has_no_version(tempdir: Path) -> None:
    tracker = LuxonisTracker(
        run_name="baseline", save_directory=tempdir, is_tensorboard=True
    )
    assert tracker.run_name == "baseline"
    assert tracker.version == 0


def test_other_ranks_join_the_latest_run(tempdir: Path) -> None:
    LuxonisTracker(save_directory=tempdir, is_tensorboard=True)
    main = LuxonisTracker(save_directory=tempdir, is_tensorboard=True)
    worker = LuxonisTracker(
        save_directory=tempdir, is_tensorboard=True, rank=1
    )

    assert worker.run_name == main.run_name


def test_other_ranks_wait_for_the_run_directory(
    tempdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = LuxonisTracker(
        run_name="explicit", save_directory=tempdir, is_tensorboard=True
    )

    def create_run_directory(_: float) -> None:
        (tempdir / "3-late").mkdir()

    monkeypatch.setattr(tracker_module.time, "sleep", create_run_directory)

    assert tracker._get_latest_run_name(timeout=5) == "3-late"


def test_other_ranks_give_up_eventually(tempdir: Path) -> None:
    tracker = LuxonisTracker(
        run_name="explicit", save_directory=tempdir, is_tensorboard=True
    )
    with pytest.raises(RuntimeError, match="No run directory found"):
        tracker._get_latest_run_name(timeout=0)


def test_other_ranks_never_log(
    tempdir: Path, fake_backends: SimpleNamespace
) -> None:
    worker = LuxonisTracker(
        project_name="project",
        run_name="run",
        save_directory=tempdir,
        is_mlflow=True,
        mlflow_tracking_uri="http://mlflow",
        rank=1,
    )

    assert worker.experiment == {}
    assert worker.log_metric("loss", 1.0, 0) is None
    assert worker.close() is None
    assert fake_backends.mlflow.calls == []


def test_mlflow_retry_count_is_only_a_default(
    tempdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "7")
    LuxonisTracker(
        project_name="project",
        save_directory=tempdir,
        is_mlflow=True,
        mlflow_tracking_uri="http://mlflow",
    )
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "7"

    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES")
    LuxonisTracker(
        project_name="project",
        save_directory=tempdir,
        is_mlflow=True,
        mlflow_tracking_uri="http://mlflow",
    )
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "2"


def test_no_mlflow_environment_without_mlflow(
    tempdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    LuxonisTracker(save_directory=tempdir, is_tensorboard=True)
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ


def test_tensorboard_logging(
    tb_tracker: LuxonisTracker, tempdir: Path
) -> None:
    writer = tb_tracker.experiment["tensorboard"]
    assert isinstance(writer, FakeSummaryWriter)
    assert writer.log_dir == tempdir / "tensorboard_logs" / "run"

    tb_tracker.log_metric("loss", 0.5, 1)
    tb_tracker.log_metrics({"acc": 0.9, "f1": 0.8}, 2)
    tb_tracker.log_image("val/vis", rgb_image(), 3)
    tb_tracker.log_images({"val/a": rgb_image(), "val/b": rgb_image()}, 4)
    tb_tracker.log_hyperparams({"lr": 0.1, "frozen": True, "scheduler": None})
    tb_tracker.close()

    assert writer.scalars == [
        ("loss", 0.5, 1),
        ("acc", 0.9, 2),
        ("f1", 0.8, 2),
    ]
    assert [name for name, *_ in writer.images] == [
        "val/vis",
        "val/a",
        "val/b",
    ]
    assert writer.images[0][3] == "HWC"
    # TensorBoard rejects non-scalar hyperparameters such as ``None``
    assert writer.hparams == [{"lr": 0.1, "frozen": True, "scheduler": "None"}]
    assert writer.closed


def test_tensorboard_logs_the_whole_matrix(tb_tracker: LuxonisTracker) -> None:
    matrix = np.arange(2000).reshape(40, 50)
    tb_tracker.log_matrix(matrix, "confusion", 1)

    name, text, step = tb_tracker.experiment["tensorboard"].texts[0]
    assert (name, step) == ("confusion", 1)
    # numpy summarizes arrays over 1000 elements unless told otherwise
    assert "..." not in text
    assert "1999" in text


@pytest.mark.usefixtures("fake_backends")
def test_sweep_runs_get_their_own_trial_directory(tempdir: Path) -> None:
    for trial in range(3):
        tracker = LuxonisTracker(
            run_name="sweep",
            save_directory=tempdir,
            is_tensorboard=True,
            is_sweep=True,
        )
        writer = tracker.experiment["tensorboard"]
        assert writer.log_dir.name == f"trial_{trial}"


def test_wandb_logging(
    wandb_tracker: LuxonisTracker,
    fake_backends: SimpleNamespace,
    tempdir: Path,
) -> None:
    run = wandb_tracker.experiment["wandb_run"]
    assert run.init_kwargs["project"] == "project"
    assert run.init_kwargs["entity"] == "entity"
    assert run.init_kwargs["name"] == "run"
    assert run is fake_backends.wandb.run

    artifact_path = tempdir / "model.onnx"
    artifact_path.write_text("weights")

    wandb_tracker.log_hyperparams({"lr": 0.1})
    wandb_tracker.log_metric("loss", 0.5, 1)
    wandb_tracker.log_metrics({"acc": 0.9}, 2)
    wandb_tracker.log_image("vis", rgb_image(), 3)
    wandb_tracker.log_matrix(np.arange(4).reshape(2, 2), "confusion", 4)
    wandb_tracker.upload_artifact(artifact_path, typ="model")
    wandb_tracker.close()

    assert run.config.params == {"lr": 0.1}
    assert run.logged[0] == {"loss": 0.5}
    assert run.logged[1] == {"acc": 0.9}
    assert run.logged[2]["vis"] == {
        "img": pytest.approx(rgb_image()),
        "caption": "vis",
    }
    assert run.logged[3]["confusion_table"].rows == [(0, 0, 1), (1, 2, 3)]
    assert run.artifacts[0].name == "model"
    assert run.artifacts[0].type == "model"
    assert run.artifacts[0].files == [str(artifact_path)]
    assert run.exit_code == 0


@pytest.mark.usefixtures("fake_backends")
def test_wandb_falls_back_to_the_project_id(tempdir: Path) -> None:
    tracker = LuxonisTracker(
        project_id="42",
        run_name="run",
        save_directory=tempdir,
        is_wandb=True,
        wandb_entity="entity",
    )
    assert tracker.experiment["wandb_run"].init_kwargs["project"] == "42"


def test_mlflow_logging(
    mlflow_tracker: LuxonisTracker,
    fake_backends: SimpleNamespace,
    tempdir: Path,
) -> None:
    mlflow = fake_backends.mlflow
    artifact_path = tempdir / "model.onnx"
    artifact_path.write_text("weights")

    mlflow_tracker.log_hyperparams({"lr": 0.1})
    mlflow_tracker.log_metric("loss", 0.5, 1)
    mlflow_tracker.log_metrics({"acc": 0.9}, 2)
    mlflow_tracker.log_image("val/vis", rgb_image(), 3)
    mlflow_tracker.log_matrix(
        np.arange(4).reshape(2, 2), "confusion", 4, extra_data={"n": 2}
    )
    mlflow_tracker.upload_artifact(artifact_path)
    mlflow_tracker.close()

    assert mlflow.tracking_uri == "http://mlflow"
    assert mlflow.experiment_kwargs == {
        "experiment_name": "project",
        "experiment_id": None,
    }
    assert mlflow.run_kwargs == {
        "run_id": None,
        "run_name": "run",
        "nested": False,
    }
    # whether system metrics are enabled depends on psutil being installed,
    # so it is asserted in the two `find_spec` tests instead
    # the resolved IDs are stored, the project name is left alone
    assert mlflow_tracker.project_id == "exp-1"
    assert mlflow_tracker.run_id == "run-1"
    assert mlflow_tracker.project_name == "project"

    assert mlflow.kinds == [
        "log_params",
        "log_metric",
        "log_metrics",
        "log_image",
        "log_dict",
        "log_artifact",
    ]
    assert mlflow.calls[1][1] == ("loss", 0.5, 1)
    assert mlflow.calls[3][1][1] == "val/3/vis.png"
    assert mlflow.calls[4][1] == (
        {"flat_array": [0, 1, 2, 3], "shape": (2, 2), "n": 2},
        "confusion.json",
    )
    assert mlflow.calls[5][1] == (str(artifact_path),)
    assert mlflow.end_run_calls == ["FINISHED"]
    assert not mlflow_tracker.local_logs


def test_mlflow_image_name_without_a_directory(
    mlflow_tracker: LuxonisTracker, fake_backends: SimpleNamespace
) -> None:
    mlflow_tracker.log_image("loss_curve", rgb_image(), 7)
    assert fake_backends.mlflow.calls[0][1][1] == "7/loss_curve.png"


def test_mlflow_resumes_a_run(
    tempdir: Path, fake_backends: SimpleNamespace
) -> None:
    tracker = LuxonisTracker(
        project_id="7",
        project_name="project",
        run_id="previous-run",
        run_name="run",
        save_directory=tempdir,
        is_mlflow=True,
        mlflow_tracking_uri="http://mlflow",
        is_sweep=True,
    )
    tracker.log_metric("loss", 0.5, 1)

    mlflow = fake_backends.mlflow
    assert mlflow.experiment_kwargs == {
        "experiment_name": None,
        "experiment_id": "7",
    }
    assert mlflow.run_kwargs == {
        "run_id": "previous-run",
        "run_name": "run",
        "nested": True,
    }
    assert tracker.run_id == "previous-run"


def test_system_metrics_need_psutil(
    mlflow_tracker: LuxonisTracker,
    fake_backends: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker_module, "find_spec", lambda _: None)
    mlflow_tracker.log_metric("loss", 0.5, 1)
    assert fake_backends.mlflow.system_metrics is False


def test_gpu_metrics_need_pynvml(
    mlflow_tracker: LuxonisTracker,
    fake_backends: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker_module, "find_spec", lambda _: object())
    mlflow_tracker.log_metric("loss", 0.5, 1)
    assert fake_backends.mlflow.system_metrics is True


def test_unreachable_mlflow_does_not_break_logging(
    mlflow_tracker: LuxonisTracker, fake_backends: SimpleNamespace
) -> None:
    mlflow = fake_backends.mlflow
    mlflow.fail_init = True

    for step in range(5):
        mlflow_tracker.log_metric("loss", 1.0, step)

    assert not mlflow_tracker.mlflow_initialized
    assert len(mlflow_tracker.local_logs) == 5
    # a failed connection is retried once per interval, not once per log
    assert mlflow.init_attempts == 1


def test_mlflow_reconnects_and_replays_in_order(
    mlflow_tracker: LuxonisTracker, fake_backends: SimpleNamespace
) -> None:
    mlflow = fake_backends.mlflow
    mlflow.fail_init = True
    for step in range(3):
        mlflow_tracker.log_metric("loss", 1.0, step)

    mlflow.fail_init = False
    mlflow_tracker._mlflow_retry_at = 0.0
    mlflow_tracker.log_metric("loss", 1.0, 3)

    assert mlflow.init_attempts == 2
    assert mlflow_tracker.mlflow_initialized
    assert not mlflow_tracker.local_logs
    assert [args[2] for _, args in mlflow.calls] == [0, 1, 2, 3]


def test_missing_mlflow_package_does_not_break_logging(
    tempdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a ``None`` entry makes ``import mlflow`` raise ``ImportError``
    monkeypatch.setitem(sys.modules, "mlflow", None)
    tracker = LuxonisTracker(
        project_name="project",
        run_name="run",
        save_directory=tempdir,
        is_mlflow=True,
        mlflow_tracking_uri="http://mlflow",
    )

    tracker.log_metric("loss", 0.5, 1)
    tracker.close()

    assert (tracker.run_directory / "local_logs.json").exists()


def test_failing_calls_are_buffered_and_replayed(
    mlflow_tracker: LuxonisTracker, fake_backends: SimpleNamespace
) -> None:
    mlflow = fake_backends.mlflow
    mlflow_tracker.log_metric("loss", 1.0, 0)
    assert mlflow.kinds == ["log_metric"]

    mlflow.fail_after = 1
    mlflow_tracker.log_metric("loss", 1.0, 1)
    assert [call.args[2] for call in mlflow_tracker.local_logs] == [1]

    mlflow.fail_after = None
    mlflow_tracker.log_metric("loss", 1.0, 2)
    assert not mlflow_tracker.local_logs
    assert [args[2] for _, args in mlflow.calls] == [0, 1, 2]


def test_replaying_stops_at_the_first_failure(
    mlflow_tracker: LuxonisTracker, fake_backends: SimpleNamespace
) -> None:
    mlflow = fake_backends.mlflow
    mlflow.fail_init = True
    for step in range(3):
        mlflow_tracker.log_metric("loss", 1.0, step)

    mlflow.fail_init = False
    mlflow.fail_after = 2
    mlflow_tracker._mlflow_retry_at = 0.0
    assert mlflow_tracker.experiment["mlflow"] is mlflow

    mlflow_tracker.log_stored_logs_to_mlflow()

    assert [args[2] for _, args in mlflow.calls] == [0, 1]
    assert [call.args[2] for call in mlflow_tracker.local_logs] == [2]


def test_nothing_to_replay(
    mlflow_tracker: LuxonisTracker, fake_backends: SimpleNamespace
) -> None:
    fake_backends.mlflow.fail_init = True
    # not initialized
    mlflow_tracker.log_stored_logs_to_mlflow()

    fake_backends.mlflow.fail_init = False
    mlflow_tracker._mlflow_retry_at = 0.0
    assert mlflow_tracker.experiment["mlflow"] is fake_backends.mlflow
    # initialized, but nothing buffered
    mlflow_tracker.log_stored_logs_to_mlflow()

    assert fake_backends.mlflow.calls == []


def test_buffer_is_bounded(
    mlflow_tracker: LuxonisTracker,
    fake_backends: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracker_module, "MAX_BUFFERED_IMAGES", 2)
    monkeypatch.setattr(tracker_module, "MAX_BUFFERED_LOGS", 4)
    fake_backends.mlflow.fail_init = True

    for step in range(5):
        mlflow_tracker.log_image("vis", rgb_image(), step)
    assert len(mlflow_tracker.local_logs) == 2

    for step in range(5):
        mlflow_tracker.log_metric("loss", 1.0, step)
    assert len(mlflow_tracker.local_logs) == 4
    assert [call.kind for call in mlflow_tracker.local_logs] == [
        "log_metric"
    ] * 4


def test_buffered_logs_are_saved_locally(
    mlflow_tracker: LuxonisTracker,
    fake_backends: SimpleNamespace,
    tempdir: Path,
) -> None:
    fake_backends.mlflow.fail_init = True
    artifact_path = tempdir / "model.onnx"
    artifact_path.write_text("weights")
    deleted_path = tempdir / "deleted.txt"
    deleted_path.write_text("gone")

    mlflow_tracker.log_hyperparams({"lr": 0.1})
    mlflow_tracker.log_hyperparams({"batch_size": 32})
    # metrics often arrive as NumPy scalars, which JSON cannot serialize
    mlflow_tracker.log_metric("loss", np.float32(0.5), 1)  # pyright: ignore[reportArgumentType]
    mlflow_tracker.log_metrics({"acc": 0.9}, 2)
    mlflow_tracker.log_matrix(
        np.arange(4).reshape(2, 2),
        "confusion",
        3,
        extra_data={"raw": np.arange(3), "output": tempdir},
    )
    mlflow_tracker.log_image("val/vis", rgb_image(), 4)
    mlflow_tracker.upload_artifact(artifact_path)
    mlflow_tracker.upload_artifact_to_mlflow(deleted_path)
    deleted_path.unlink()

    mlflow_tracker.close("failed")

    saved = json.loads(
        (mlflow_tracker.run_directory / "local_logs.json").read_text()
    )
    assert saved["params"] == {"lr": 0.1, "batch_size": 32}
    # NumPy scalars keep their value instead of breaking serialization
    assert saved["metric"] == [{"name": "loss", "value": 0.5, "step": 1}]
    assert saved["metrics"] == [{"metrics": {"acc": 0.9}, "step": 2}]
    assert saved["matrices"][0]["matrix"]["raw"] == [0, 1, 2]
    assert saved["matrices"][0]["matrix"]["output"] == str(tempdir)

    image_path = Path(saved["images"][0]["image_data"])
    assert image_path.name == "0_val_4_vis.png"
    assert image_path.exists()

    assert saved["artifacts"][0]["path"] == str(
        mlflow_tracker.run_directory / "artifacts" / "model.onnx"
    )
    assert Path(saved["artifacts"][0]["path"]).read_text() == "weights"
    # the artifact that disappeared is reported but does not break the save
    assert saved["artifacts"][1]["path"] == str(deleted_path)

    # the buffer is cleared, so closing again cannot duplicate anything
    assert not mlflow_tracker.local_logs


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 3), (4, 4, 4)])
def test_images_of_any_shape_are_saved(
    tempdir: Path, shape: tuple[int, ...]
) -> None:
    path = LuxonisTracker._save_image_locally(
        tempdir, 0, "val/vis", np.zeros(shape, dtype=np.uint8)
    )
    assert path is not None
    assert Path(path).exists()


def test_unnamed_image_gets_a_fallback_name(tempdir: Path) -> None:
    path = LuxonisTracker._save_image_locally(
        tempdir, 3, "///", np.zeros((4, 4, 3), dtype=np.uint8)
    )
    assert path == str(tempdir / "3_image.png")


def test_unencodable_image_is_skipped(tempdir: Path) -> None:
    unencodable = np.array([{"not": "an image"}], dtype=object)
    assert (
        LuxonisTracker._save_image_locally(tempdir, 0, "vis", unencodable)
        is None
    )


def test_close_flushes_the_buffer_when_mlflow_recovers(
    mlflow_tracker: LuxonisTracker, fake_backends: SimpleNamespace
) -> None:
    fake_backends.mlflow.fail_init = True
    mlflow_tracker.log_metric("loss", 0.5, 1)

    fake_backends.mlflow.fail_init = False
    mlflow_tracker._mlflow_retry_at = 0.0
    mlflow_tracker.close()

    assert fake_backends.mlflow.kinds == ["log_metric"]
    assert not (mlflow_tracker.run_directory / "local_logs.json").exists()


def test_close_is_idempotent(
    tempdir: Path, fake_backends: SimpleNamespace
) -> None:
    tracker = LuxonisTracker(
        project_name="project",
        run_name="run",
        save_directory=tempdir,
        is_tensorboard=True,
        is_wandb=True,
        is_mlflow=True,
        wandb_entity="entity",
        mlflow_tracking_uri="http://mlflow",
    )
    tracker.log_metric("loss", 0.5, 1)

    tracker.close()
    tracker.close()

    assert fake_backends.mlflow.end_run_calls == ["FINISHED"]
    assert tracker.experiment["wandb_run"].exit_code == 0
    assert tracker.experiment["tensorboard"].closed


def test_close_survives_broken_backends(
    tempdir: Path, fake_backends: SimpleNamespace
) -> None:
    tracker = LuxonisTracker(
        project_name="project",
        run_name="run",
        save_directory=tempdir,
        is_tensorboard=True,
        is_wandb=True,
        is_mlflow=True,
        wandb_entity="entity",
        mlflow_tracking_uri="http://mlflow",
    )
    tracker.log_metric("loss", 0.5, 1)

    tracker.experiment["tensorboard"].fail_on_close = True
    tracker.experiment["wandb_run"].fail_on_finish = True
    fake_backends.mlflow.fail_end_run = True

    tracker.close()

    assert tracker._closed


def test_context_manager_reports_success(
    tempdir: Path, fake_backends: SimpleNamespace
) -> None:
    with LuxonisTracker(
        project_name="project",
        run_name="run",
        save_directory=tempdir,
        is_mlflow=True,
        mlflow_tracking_uri="http://mlflow",
    ) as tracker:
        tracker.log_metric("loss", 0.5, 1)

    assert fake_backends.mlflow.end_run_calls == ["FINISHED"]


def test_context_manager_reports_failure(
    mlflow_tracker: LuxonisTracker, fake_backends: SimpleNamespace
) -> None:
    def failing_run() -> None:
        with mlflow_tracker:
            mlflow_tracker.log_metric("loss", 0.5, 1)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        failing_run()

    assert fake_backends.mlflow.end_run_calls == ["FAILED"]
