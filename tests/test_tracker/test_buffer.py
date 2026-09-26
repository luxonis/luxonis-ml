import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import luxonis_ml.tracker.buffer as buffer_module
from luxonis_ml.tracker import BufferedBackend, RunContext
from luxonis_ml.tracker.buffer import _to_json

from .conftest import FakeBackend, Rejected

RETRY_INTERVAL = 60.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(buffer_module, "time", clock)
    return clock


@pytest.fixture
def inner(run: RunContext) -> FakeBackend:
    return FakeBackend(run)


@pytest.fixture
def buffered(inner: FakeBackend, clock: FakeClock) -> BufferedBackend:
    return BufferedBackend(inner, "fake", retry_interval=RETRY_INTERVAL)


def metrics(step: int) -> tuple[str, dict[str, float], int]:
    return ("log_metrics", {"loss": 1 / (step + 1)}, step)


def log(buffered: BufferedBackend, step: int) -> None:
    buffered.log_metrics({"loss": 1 / (step + 1)}, step)


def test_a_healthy_backend_gets_each_call_at_once(
    buffered: BufferedBackend,
    inner: FakeBackend,
    image: npt.NDArray[np.uint8],
    tmp_path: Path,
):
    artifact = tmp_path / "model.txt"
    artifact.write_text("weights")
    buffered.start()

    buffered.log_hyperparams({"lr": 0.1})
    log(buffered, 1)
    buffered.log_image("image", image, 2)
    buffered.log_matrix(np.eye(2), "matrix", 3, {"labels": ["a", "b"]})
    buffered.upload_artifact(artifact, "final.txt", "weights")

    assert [call[0] for call in inner.calls] == [
        "log_hyperparams",
        "log_metrics",
        "log_image",
        "log_matrix",
        "upload_artifact",
    ]
    assert inner.calls[4] == (
        "upload_artifact",
        "weights",
        "final.txt",
        "weights",
    )
    assert buffered.experiment is inner
    assert buffered.is_transient(RuntimeError())
    assert not buffered.is_transient(Rejected())
    assert not (buffered.unsent_directory / "artifacts").exists()


def test_calls_wait_for_a_backend_that_failed_to_start(
    buffered: BufferedBackend,
    inner: FakeBackend,
    clock: FakeClock,
    warnings_log: list[str],
):
    inner.error = ConnectionError("down")
    buffered.start()
    log(buffered, 0)
    inner.error = None

    clock.now = RETRY_INTERVAL - 1
    log(buffered, 1)
    assert inner.starts == 0

    clock.now = RETRY_INTERVAL
    log(buffered, 2)
    assert inner.starts == 1
    assert inner.calls == [metrics(0), metrics(1), metrics(2)]
    assert any("fake is unavailable: down" in m for m in warnings_log)


def test_a_rejected_start_raises(
    buffered: BufferedBackend, inner: FakeBackend
):
    inner.error = Rejected("wrong project")

    with pytest.raises(Rejected):
        buffered.start()


def test_a_call_that_fails_waits_for_the_next_attempt(
    buffered: BufferedBackend, inner: FakeBackend, clock: FakeClock
):
    buffered.start()
    inner.error = ConnectionError("down")
    log(buffered, 0)
    inner.error = None

    log(buffered, 1)
    assert inner.calls == []

    clock.now = RETRY_INTERVAL
    log(buffered, 2)
    assert inner.calls == [metrics(0), metrics(1), metrics(2)]


def test_a_rejected_call_is_dropped(
    buffered: BufferedBackend, inner: FakeBackend, warnings_log: list[str]
):
    buffered.start()
    inner.error = Rejected("bad value")
    log(buffered, 0)
    inner.error = None

    log(buffered, 1)

    assert inner.calls == [metrics(1)]
    assert any("fake rejected a call: bad value" in m for m in warnings_log)


class FlakyBackend(FakeBackend):
    """Raise the queued errors, one for each call, then accept."""

    def __init__(self, run: RunContext, errors: list[Exception]) -> None:
        super().__init__(run)
        self.errors: list[Exception] = errors

    def _raise(self) -> None:
        if self.errors:
            raise self.errors.pop(0)


def test_the_replay_stops_at_an_outage_and_skips_a_rejected_call(
    run: RunContext, clock: FakeClock
):
    inner = FlakyBackend(run, [])
    buffered = BufferedBackend(inner, "fake", retry_interval=RETRY_INTERVAL)
    buffered.start()
    inner.errors = [ConnectionError(), ConnectionError()]
    log(buffered, 0)
    log(buffered, 1)

    clock.now = RETRY_INTERVAL
    inner.errors = [ConnectionError()]
    log(buffered, 2)
    assert inner.calls == []

    clock.now = 2 * RETRY_INTERVAL
    inner.errors = [Rejected()]
    log(buffered, 3)
    assert inner.calls == [metrics(1), metrics(2), metrics(3)]


def test_a_full_buffer_drops_the_oldest_call_of_the_kind(
    buffered: BufferedBackend,
    inner: FakeBackend,
    image: npt.NDArray[np.uint8],
    clock: FakeClock,
    warnings_log: list[str],
):
    buffered.start()
    inner.error = ConnectionError()
    buffered.log_hyperparams({"lr": 0.1})
    for step in range(52):
        buffered.log_image(f"image_{step}", image, step)
    log(buffered, 0)
    inner.error = None

    clock.now = RETRY_INTERVAL
    buffered.close("success")

    names = [call[1] for call in inner.calls if call[0] == "log_image"]
    assert names == [f"image_{step}" for step in range(2, 52)]
    assert inner.calls[0] == ("log_hyperparams", {"lr": 0.1})
    assert inner.calls[-1] == metrics(0)
    drops = [m for m in warnings_log if "dropping the oldest" in m]
    assert len(drops) == 1


def test_each_outage_reports_its_drops(
    buffered: BufferedBackend,
    inner: FakeBackend,
    image: npt.NDArray[np.uint8],
    clock: FakeClock,
    warnings_log: list[str],
):
    buffered.start()
    for outage in range(2):
        inner.error = ConnectionError()
        for step in range(51):
            buffered.log_image("image", image, step)
        inner.error = None
        clock.now += RETRY_INTERVAL
        log(buffered, outage)

    assert len([m for m in warnings_log if "dropping the oldest" in m]) == 2


def test_a_buffered_artifact_outlives_the_original(
    buffered: BufferedBackend,
    inner: FakeBackend,
    clock: FakeClock,
    tmp_path: Path,
):
    buffered.start()
    inner.error = ConnectionError()
    for name in ["first", "second"]:
        artifact = tmp_path / "model.txt"
        artifact.write_text(name)
        buffered.upload_artifact(artifact, None, "weights")
        artifact.unlink()
    kept = buffered.unsent_directory / "artifacts"
    assert len(list(kept.iterdir())) == 2
    inner.error = None

    clock.now = RETRY_INTERVAL
    log(buffered, 0)

    assert inner.calls[:2] == [
        ("upload_artifact", "first", None, "weights"),
        ("upload_artifact", "second", None, "weights"),
    ]
    assert list(kept.iterdir()) == []


def test_an_artifact_is_copied_where_a_link_fails(
    buffered: BufferedBackend,
    inner: FakeBackend,
    clock: FakeClock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def refuse(self: Path, target: Path) -> None:
        raise OSError("cross-device link")

    monkeypatch.setattr(Path, "hardlink_to", refuse)
    artifact = tmp_path / "model.txt"
    artifact.write_text("weights")
    buffered.start()
    inner.error = ConnectionError()
    buffered.upload_artifact(artifact, None, "weights")
    artifact.unlink()
    inner.error = None

    clock.now = RETRY_INTERVAL
    log(buffered, 0)

    assert inner.calls[0] == ("upload_artifact", "weights", None, "weights")


def test_an_artifact_that_cannot_be_kept_keeps_its_path(
    buffered: BufferedBackend,
    inner: FakeBackend,
    clock: FakeClock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    warnings_log: list[str],
):
    def refuse(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "hardlink_to", refuse)
    monkeypatch.setattr(buffer_module.shutil, "copy2", refuse)
    artifact = tmp_path / "model.txt"
    artifact.write_text("weights")
    buffered.start()
    inner.error = ConnectionError()
    buffered.upload_artifact(artifact, None, "weights")
    inner.error = None

    clock.now = RETRY_INTERVAL
    log(buffered, 0)

    assert inner.calls[0] == ("upload_artifact", "weights", None, "weights")
    assert artifact.exists()
    assert list((buffered.unsent_directory / "artifacts").iterdir()) == []
    assert any("Could not keep a copy" in m for m in warnings_log)


def test_an_artifact_that_is_gone_is_saved_with_its_path(
    buffered: BufferedBackend, inner: FakeBackend, tmp_path: Path
):
    buffered.start()
    inner.error = ConnectionError()
    buffered.upload_artifact(tmp_path / "missing.txt", None, "weights")
    inner.error = None

    buffered.close("failed")

    record = json.loads(
        (buffered.unsent_directory / "calls.jsonl").read_text()
    )
    assert record["path"] == str(tmp_path / "missing.txt")


def test_close_sends_the_buffer_to_a_recovered_backend(
    buffered: BufferedBackend, inner: FakeBackend
):
    buffered.start()
    inner.error = ConnectionError()
    log(buffered, 0)
    inner.error = None

    buffered.close("success")

    assert inner.calls == [metrics(0)]
    assert inner.status == "success"
    assert not buffered.unsent_directory.exists()


def test_close_saves_what_never_got_through(
    buffered: BufferedBackend,
    inner: FakeBackend,
    image: npt.NDArray[np.uint8],
    tmp_path: Path,
):
    artifact = tmp_path / "model.txt"
    artifact.write_text("weights")
    inner.error = ConnectionError()
    buffered.start()
    buffered.log_hyperparams({"lr": 0.1, "layers": [1, 2]})
    buffered.log_metrics({"loss": 0.5}, 1)
    buffered.log_image("val/image", image, 2)
    buffered.log_matrix(np.eye(2), "matrix", 3, {"labels": ["a", "b"]})
    buffered.upload_artifact(artifact, "final.txt", "weights")

    buffered.close("failed")

    lines = (
        (buffered.unsent_directory / "calls.jsonl").read_text().splitlines()
    )
    records = [json.loads(line) for line in lines]
    assert records[0] == {
        "call": "log_hyperparams",
        "params": {"lr": 0.1, "layers": [1, 2]},
    }
    assert records[1] == {
        "call": "log_metrics",
        "metrics": {"loss": 0.5},
        "step": 1,
    }
    assert records[2]["name"] == "val/image"
    np.testing.assert_array_equal(np.load(records[2]["image"]), image)
    assert records[3] == {
        "call": "log_matrix",
        "name": "matrix",
        "step": 3,
        "matrix": [[1.0, 0.0], [0.0, 1.0]],
        "extra_data": {"labels": ["a", "b"]},
    }
    assert records[4]["name"] == "final.txt"
    assert Path(records[4]["path"]).read_text() == "weights"
    assert inner.status is None


def test_a_second_save_appends(run: RunContext, clock: FakeClock):
    for step in range(2):
        inner = FakeBackend(run)
        inner.error = ConnectionError()
        buffered = BufferedBackend(inner, "fake")
        log(buffered, step)
        buffered.close("failed")

    path = run.run_directory / "unsent_logs" / "fake" / "calls.jsonl"
    steps = [
        json.loads(line)["step"] for line in path.read_text().splitlines()
    ]
    assert steps == [0, 1]


def test_close_ends_the_run_when_the_save_fails(
    run: RunContext, clock: FakeClock
):
    inner = FlakyBackend(run, [])
    buffered = BufferedBackend(inner, "fake")
    buffered.start()
    inner.errors = [ConnectionError(), ConnectionError()]
    log(buffered, 0)
    buffered.unsent_directory.parent.mkdir(parents=True)
    buffered.unsent_directory.touch()

    with pytest.raises(FileExistsError):
        buffered.close("failed")

    assert inner.status == "failed"


def test_close_saves_the_calls_of_a_run_that_was_rejected(
    buffered: BufferedBackend, inner: FakeBackend, warnings_log: list[str]
):
    inner.error = ConnectionError()
    log(buffered, 0)
    inner.error = Rejected("no such experiment")

    buffered.close("failed")

    assert (buffered.unsent_directory / "calls.jsonl").exists()
    assert inner.status is None
    assert any("rejected the run" in m for m in warnings_log)


def test_a_start_rejected_after_an_outage_keeps_the_calls(
    buffered: BufferedBackend,
    inner: FakeBackend,
    clock: FakeClock,
    warnings_log: list[str],
):
    """Only the first start may raise. Later, the training must go on,
    and the calls must reach the disk at close.
    """
    inner.error = ConnectionError()
    buffered.start()
    log(buffered, 0)
    inner.error = Rejected("no such experiment")

    clock.now = RETRY_INTERVAL
    log(buffered, 1)
    clock.now = 100 * RETRY_INTERVAL
    log(buffered, 2)
    inner.error = None
    buffered.close("failed")

    assert inner.starts == 1
    assert inner.calls == [metrics(0), metrics(1), metrics(2)]
    assert inner.status == "failed"
    assert any("fake rejected the run" in m for m in warnings_log)


class ReentrantBackend(FakeBackend):
    """Log another call while a call is being sent, as a signal handler
    can.
    """

    def __init__(self, run: RunContext) -> None:
        super().__init__(run)
        self.buffered: BufferedBackend | None = None

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        if step == 0 and self.buffered is not None:
            log(self.buffered, 99)
        super().log_metrics(metrics, step)


def test_a_call_made_during_a_replay_waits_its_turn(
    run: RunContext, clock: FakeClock
):
    inner = ReentrantBackend(run)
    buffered = BufferedBackend(inner, "fake", retry_interval=RETRY_INTERVAL)
    inner.buffered = buffered
    inner.error = ConnectionError()
    buffered.start()
    log(buffered, 0)
    log(buffered, 1)
    inner.error = None

    clock.now = RETRY_INTERVAL
    log(buffered, 2)

    assert inner.calls == [metrics(0), metrics(1), metrics(99), metrics(2)]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float32(0.5), 0.5),
        (np.int64(3), 3),
        (np.array([[1, 2]]), [[1, 2]]),
        (Path("a/b"), str(Path("a/b"))),
    ],
)
def test_numpy_values_become_json(value: object, expected: object):
    assert _to_json(value) == expected
