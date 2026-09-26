import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import luxonis_ml.tracker as tracker_package
import luxonis_ml.tracker.tracker as tracker_module
from luxonis_ml.tracker import (
    TRACKER_BACKENDS,
    BufferedBackend,
    LuxonisTracker,
    MLflowBackend,
    RunContext,
    TensorBoardBackend,
    WandbBackend,
)
from luxonis_ml.tracker.tracker import RUN_NAME_ENV

from .conftest import BufferedFakeBackend, FakeBackend


def make_tracker(save_directory: Path, **kwargs: Any) -> LuxonisTracker:
    kwargs.setdefault("run_name", "0-test")
    kwargs.setdefault("fake", True)
    return LuxonisTracker(save_directory=save_directory, **kwargs)


def fake(tracker: LuxonisTracker, name: str = "fake") -> FakeBackend:
    backend = tracker.backends[name]
    assert isinstance(backend, FakeBackend)
    return backend


class FakeClock:
    """Stands in for the `time` module. `sleep` advances the clock and
    runs ``on_sleep``.
    """

    def __init__(self, on_sleep: Any = None) -> None:
        self.now = 0.0
        self.on_sleep = on_sleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


def test_at_least_one_backend_is_required(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one backend"):
        make_tracker(tmp_path, fake=False, other_fake=None)


def test_an_unknown_keyword_is_rejected(tmp_path: Path):
    with pytest.raises(TypeError, match="'run_nme', and no tracker backend"):
        make_tracker(tmp_path, run_nme="typo")

    assert not (tmp_path / "0-test").exists()


def test_backend_options_reach_the_backend(tmp_path: Path):
    tracker = make_tracker(tmp_path, fake={"option": "set"})

    assert fake(tracker).option == "set"
    assert fake(tracker).run.run_directory == tmp_path / "0-test"


def test_true_turns_a_backend_on_with_its_defaults(tmp_path: Path):
    tracker = make_tracker(tmp_path, fake=True, other_fake={})

    assert fake(tracker).option == "default"
    assert fake(tracker, "other_fake").option == "default"


def test_the_built_in_backends_have_keywords(tmp_path: Path):
    tracker = make_tracker(
        tmp_path,
        project_name="project",
        fake=False,
        tensorboard=True,
        wandb={"entity": "team"},
        mlflow={"tracking_uri": "sqlite:///unused.db"},
    )

    assert list(tracker.backends) == ["tensorboard", "wandb", "mlflow"]
    assert tracker.get_backend(WandbBackend).entity == "team"


def test_a_rejected_option_creates_no_run_directory(tmp_path: Path):
    with pytest.raises(TypeError):
        make_tracker(tmp_path, fake={"unknown": 1})

    assert not (tmp_path / "0-test").exists()


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"is_tensorboard": True}, {"tensorboard": TensorBoardBackend}),
        (
            {"is_wandb": True, "wandb_entity": "team"},
            {"wandb": WandbBackend},
        ),
        (
            {"is_mlflow": True, "mlflow_tracking_uri": "sqlite:///unused.db"},
            {"mlflow": BufferedBackend},
        ),
    ],
)
def test_deprecated_flags_still_enable_backends(
    tmp_path: Path, flags: dict[str, Any], expected: dict[str, type]
):
    with pytest.deprecated_call(match="Use `"):
        tracker = LuxonisTracker(
            project_name="project",
            run_name="0-test",
            save_directory=tmp_path,
            **flags,
        )

    assert {name: type(b) for name, b in tracker.backends.items()} == expected


def test_deprecated_flags_keep_their_options(tmp_path: Path):
    with pytest.deprecated_call():
        tracker = LuxonisTracker(
            project_name="project",
            save_directory=tmp_path,
            is_wandb=True,
            wandb_entity="team",
            is_mlflow=True,
            mlflow_tracking_uri="sqlite:///unused.db",
        )

    assert tracker.get_backend(WandbBackend).entity == "team"
    assert (
        tracker.get_backend(MLflowBackend).tracking_uri
        == "sqlite:///unused.db"
    )


@pytest.mark.parametrize(
    ("flags", "replacement"),
    [
        (
            {"is_tensorboard": True, "is_wandb": True},
            "tensorboard=True, wandb=True",
        ),
        (
            {"is_mlflow": True, "mlflow_tracking_uri": "sqlite:///unused.db"},
            "mlflow={'tracking_uri': 'sqlite:///unused.db'}",
        ),
    ],
)
def test_the_warning_names_the_replacement(
    tmp_path: Path, flags: dict[str, Any], replacement: str
):
    with pytest.deprecated_call(match=re.escape(f"`{replacement}`")):
        LuxonisTracker(
            project_name="project", save_directory=tmp_path, **flags
        )


def test_a_backend_keyword_overrides_a_deprecated_flag(tmp_path: Path):
    with pytest.deprecated_call():
        tracker = LuxonisTracker(
            project_name="project",
            save_directory=tmp_path,
            is_wandb=True,
            wandb_entity="old",
            is_tensorboard=True,
            wandb={"entity": "new"},
            tensorboard=False,
        )

    assert list(tracker.backends) == ["wandb"]
    assert tracker.get_backend(WandbBackend).entity == "new"


def test_a_generated_run_name_takes_the_next_number(tmp_path: Path):
    for name in ["3-old", "10", "½-fraction", "notes"]:
        (tmp_path / name).mkdir()
    (tmp_path / "20-file").touch()

    tracker = make_tracker(tmp_path, run_name=None)

    assert tracker.run_name.startswith("11-")
    assert tracker.name == tracker.run_name
    assert tracker.version == 11
    assert tracker.run_directory.is_dir()


def test_an_empty_run_name_gets_a_generated_one(tmp_path: Path):
    tracker = make_tracker(tmp_path, run_name="")

    assert tracker.run_name.startswith("0-")
    assert tracker.run_directory != tmp_path


def test_the_first_run_is_number_zero(tmp_path: Path):
    tracker = make_tracker(tmp_path, run_name=None)

    assert tracker.version == 0
    assert tracker.run_name.startswith("0-")


def test_a_run_name_without_a_number_has_version_zero(tmp_path: Path):
    assert make_tracker(tmp_path, run_name="baseline").version == 0


def test_rank_zero_exports_a_generated_run_name(tmp_path: Path):
    tracker = make_tracker(tmp_path, run_name=None)

    assert os.environ[RUN_NAME_ENV] == tracker.run_name


def test_an_explicit_run_name_is_not_exported(tmp_path: Path):
    make_tracker(tmp_path, run_name="baseline")

    assert os.environ[RUN_NAME_ENV] == ""


def test_other_ranks_join_the_exported_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(RUN_NAME_ENV, "5-exported")

    tracker = make_tracker(tmp_path, run_name=None, rank=1)

    assert tracker.run_name == "5-exported"
    # a later tracker of the same worker must not join this run again
    assert RUN_NAME_ENV not in os.environ


def test_other_ranks_wait_for_a_new_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Without the exported name, as with ``torchrun``, a worker must
    not take a run of an earlier training for the one that rank 0 is
    about to create.
    """
    (tmp_path / "7-earlier").mkdir()
    clock = FakeClock(on_sleep=(tmp_path / "2-new").mkdir)
    monkeypatch.setattr(tracker_module, "time", clock)

    tracker = make_tracker(tmp_path, run_name=None, rank=1)

    assert tracker.run_name == "2-new"


def test_other_ranks_fall_back_to_the_newest_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warnings_log: list[str]
):
    for name in ["2-older", "10-newest", "9-newer"]:
        (tmp_path / name).mkdir()
    monkeypatch.setattr(tracker_module, "time", FakeClock())

    tracker = make_tracker(tmp_path, run_name=None, rank=1)

    assert tracker.run_name == "10-newest"
    assert any("Joining the newest run" in m for m in warnings_log)


def test_other_ranks_give_up_without_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(tracker_module, "time", FakeClock())

    with pytest.raises(RuntimeError, match="Pass `run_name`"):
        make_tracker(tmp_path, run_name=None, rank=1)


def test_other_ranks_never_start_or_log(
    tmp_path: Path, image: npt.NDArray[np.uint8]
):
    tracker = make_tracker(tmp_path, rank=1)

    tracker.log_hyperparams({"lr": 0.1})
    tracker.log_metric("loss", 0.5, 1)
    tracker.log_image("image", image, 1)
    tracker.log_matrix(np.eye(2), "matrix", 1)
    tracker.upload_artifact(tmp_path / "missing.txt")
    tracker.close()

    assert tracker.experiment == {}
    assert fake(tracker).starts == 0
    assert fake(tracker).calls == []
    assert fake(tracker).status is None


def test_backends_start_on_first_use_and_only_once(tmp_path: Path):
    tracker = make_tracker(tmp_path)
    assert fake(tracker).starts == 0

    tracker.log_metric("loss", 0.5, 1)
    tracker.log_metric("loss", 0.4, 2)

    assert fake(tracker).starts == 1


def test_a_failed_start_is_retried_on_the_next_call(tmp_path: Path):
    tracker = make_tracker(tmp_path)
    fake(tracker).error = RuntimeError("down")

    with pytest.raises(RuntimeError, match="down"):
        tracker.log_metric("loss", 0.5, 1)

    fake(tracker).error = None
    tracker.log_metric("loss", 0.4, 2)

    assert fake(tracker).starts == 1
    assert fake(tracker).calls == [("log_metrics", {"loss": 0.4}, 2)]


def test_each_call_reaches_each_backend(
    tmp_path: Path, image: npt.NDArray[np.uint8]
):
    artifact = tmp_path / "model.txt"
    artifact.write_text("weights")
    tracker = make_tracker(tmp_path, other_fake=True)

    tracker.log_hyperparams({"lr": 0.1})
    tracker.log_metric("loss", 0.5, 1)
    tracker.log_metrics({"acc": 0.9}, 2)
    tracker.log_images({"a": image, "b": image}, 3)
    tracker.log_matrix(np.eye(2), "matrix", 4)
    tracker.log_matrix(np.eye(2), "labelled", 5, {"labels": ["x", "y"]})
    tracker.upload_artifact(str(artifact), name="final.txt", typ="weights")

    for name in ["fake", "other_fake"]:
        calls = fake(tracker, name).calls
        assert [call[0] for call in calls] == [
            "log_hyperparams",
            "log_metrics",
            "log_metrics",
            "log_image",
            "log_image",
            "log_matrix",
            "log_matrix",
            "upload_artifact",
        ]
        assert calls[1] == ("log_metrics", {"loss": 0.5}, 1)
        assert calls[3][1:] == ("a", image, 3)
        assert calls[5][2:] == ("matrix", 4, {})
        assert calls[6][4] == {"labels": ["x", "y"]}
        assert calls[7] == (
            "upload_artifact",
            "weights",
            "final.txt",
            "weights",
        )


def test_experiment_starts_the_backends_and_maps_their_handles(
    tmp_path: Path,
):
    tracker = make_tracker(tmp_path, other_fake=True)

    assert tracker.experiment == {
        "fake": fake(tracker),
        "other_fake": fake(tracker, "other_fake"),
    }
    assert fake(tracker).starts == 1


def test_get_backend_unwraps_a_buffered_backend(tmp_path: Path):
    tracker = make_tracker(tmp_path, fake=True, buffered_fake=True)

    assert isinstance(tracker.backends["buffered_fake"], BufferedBackend)
    assert isinstance(
        tracker.get_backend(BufferedFakeBackend), BufferedFakeBackend
    )
    assert tracker.get_backend(FakeBackend) is tracker.backends["fake"]
    with pytest.raises(KeyError, match="TensorBoardBackend"):
        tracker.get_backend(TensorBoardBackend)


def test_the_backends_cannot_be_replaced(tmp_path: Path):
    backends: Any = make_tracker(tmp_path).backends

    with pytest.raises(TypeError):
        backends["fake"] = None


@pytest.mark.parametrize(
    ("backend", "flag"),
    [
        ("tensorboard", "is_tensorboard"),
        ("wandb", "is_wandb"),
        ("mlflow", "is_mlflow"),
    ],
)
def test_the_deprecated_flags_read_the_backends(
    tmp_path: Path, backend: str, flag: str
):
    options = (
        {"tracking_uri": "sqlite:///unused.db"} if backend == "mlflow" else {}
    )
    tracker = make_tracker(
        tmp_path, project_name="project", **{backend: options}
    )
    other = make_tracker(tmp_path)

    with pytest.deprecated_call():
        assert getattr(tracker, flag)
    with pytest.deprecated_call():
        assert not getattr(other, flag)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("success", "success"),
        ("finished", "success"),
        ("failed", "failed"),
        ("interrupted", "failed"),
    ],
)
def test_close_ends_each_started_run(
    tmp_path: Path, status: str, expected: str
):
    tracker = make_tracker(tmp_path, other_fake=True)
    tracker.log_metric("loss", 0.5, 1)

    tracker.close(status)

    assert fake(tracker).status == expected
    assert fake(tracker, "other_fake").status == expected


def test_close_leaves_a_backend_that_never_started_alone(tmp_path: Path):
    tracker = make_tracker(tmp_path)

    tracker.close()

    assert fake(tracker).status is None


def test_a_backend_that_fails_to_close_does_not_stop_the_others(
    tmp_path: Path, warnings_log: list[str]
):
    tracker = make_tracker(tmp_path, other_fake=True)
    tracker.log_metric("loss", 0.5, 1)
    fake(tracker).error = RuntimeError("broken")

    tracker.close()

    assert fake(tracker, "other_fake").status == "success"
    assert any("Could not close the fake run" in m for m in warnings_log)


def test_close_runs_once(tmp_path: Path):
    tracker = make_tracker(tmp_path)
    tracker.log_metric("loss", 0.5, 1)
    tracker.close("success")

    tracker.close("failed")

    assert fake(tracker).status == "success"


def test_calls_after_close_are_ignored(
    tmp_path: Path, warnings_log: list[str]
):
    tracker = make_tracker(tmp_path, other_fake=True)
    tracker.log_metric("loss", 0.5, 1)
    tracker.close()

    tracker.log_metric("loss", 0.4, 2)

    assert fake(tracker).calls == [("log_metrics", {"loss": 0.5}, 1)]
    assert any("tracker is closed" in m for m in warnings_log)


def test_other_ranks_stay_silent_after_close(
    tmp_path: Path, warnings_log: list[str]
):
    tracker = make_tracker(tmp_path, rank=1)
    tracker.close()

    tracker.log_metric("loss", 0.5, 1)

    assert warnings_log == []


def test_experiment_after_close_starts_nothing(tmp_path: Path):
    tracker = make_tracker(tmp_path)
    tracker.close()

    assert tracker.experiment == {}
    assert fake(tracker).starts == 0


def test_the_context_manager_closes_a_successful_run(tmp_path: Path):
    with make_tracker(tmp_path) as tracker:
        tracker.log_metric("loss", 0.5, 1)

    assert fake(tracker).status == "success"


def test_the_context_manager_marks_a_failed_run(tmp_path: Path):
    tracker = make_tracker(tmp_path)
    tracker.log_metric("loss", 0.5, 1)

    with pytest.raises(KeyError), tracker:
        raise KeyError

    assert fake(tracker).status == "failed"


class FakeEntryPoint:
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value

    def load(self) -> Any:
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class PluginBackend(FakeBackend):
    pass


@pytest.fixture
def plugins(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[FakeEntryPoint]]:
    entry_points: list[FakeEntryPoint] = []

    def select(*, group: str) -> list[FakeEntryPoint]:
        assert group == "tracker_plugins"
        return entry_points

    monkeypatch.setattr(tracker_package, "entry_points", select)
    yield entry_points
    TRACKER_BACKENDS._module_dict.pop("plugin", None)


def test_a_plugin_is_registered_under_its_entry_point_name(
    plugins: list[FakeEntryPoint],
):
    plugins.append(FakeEntryPoint("plugin", PluginBackend))

    tracker_package._load_backend_plugins()

    assert TRACKER_BACKENDS.get("plugin") is PluginBackend


def test_a_plugin_that_registered_itself_is_kept(
    plugins: list[FakeEntryPoint],
):
    plugins.append(FakeEntryPoint("fake", PluginBackend))

    tracker_package._load_backend_plugins()

    assert TRACKER_BACKENDS.get("fake") is FakeBackend


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (ImportError("no module"), "no module"),
        (object, "not a `TrackerBackend` subclass"),
        (len, "not a `TrackerBackend` subclass"),
    ],
)
def test_a_broken_plugin_is_skipped(
    plugins: list[FakeEntryPoint],
    warnings_log: list[str],
    value: Any,
    reason: str,
):
    plugins.append(FakeEntryPoint("plugin", value))

    tracker_package._load_backend_plugins()

    assert "plugin" not in TRACKER_BACKENDS
    assert any(reason in m for m in warnings_log)


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (ConnectionError("reset"), True),
        (TimeoutError("slow"), True),
        (FileNotFoundError("model.txt"), False),
        (ValueError("bad value"), False),
    ],
)
def test_the_default_retries_only_io_errors(
    tmp_path: Path, error: Exception, transient: bool
):
    backend = TensorBoardBackend(RunContext("0-test", tmp_path))

    assert backend.is_transient(error) is transient
