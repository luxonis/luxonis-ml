import struct
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

# the generated protobuf modules have no stubs
from tensorboardX.proto.event_pb2 import (
    Event,  # pyright: ignore[reportAttributeAccessIssue]
)
from tensorboardX.proto.summary_pb2 import (
    Summary,  # pyright: ignore[reportAttributeAccessIssue]
)

from luxonis_ml.tracker import RunContext, TensorBoardBackend


def read_summaries(log_dir: Path) -> list[tuple[int, Summary.Value]]:
    """Read the summary values of the event files in ``log_dir``.

    An event file is a sequence of records: a length of 8 bytes, a CRC
    of 4 bytes, the event, and a CRC of 4 bytes.
    """
    values: list[tuple[int, Summary.Value]] = []
    for path in sorted(log_dir.glob("events.out.tfevents.*")):
        data = path.read_bytes()
        offset = 0
        while offset < len(data):
            (length,) = struct.unpack_from("<Q", data, offset)
            event = Event.FromString(data[offset + 12 : offset + 12 + length])
            values.extend((event.step, value) for value in event.summary.value)
            offset += 12 + length + 4
    return values


@pytest.fixture
def backend(run: RunContext) -> TensorBoardBackend:
    backend = TensorBoardBackend(run)
    backend.start()
    return backend


def test_the_events_go_to_the_run_directory(
    backend: TensorBoardBackend, run: RunContext
):
    log_dir = run.save_directory / "tensorboard_logs" / run.run_name

    assert Path(backend.experiment.logdir) == log_dir
    assert list(log_dir.glob("events.out.tfevents.*"))


def test_each_sweep_trial_gets_the_next_directory(tmp_path: Path):
    run = RunContext("0-sweep", tmp_path, is_sweep=True)
    log_dir = tmp_path / "tensorboard_logs" / "0-sweep"
    first = TensorBoardBackend(run)
    first.start()
    for name in ["trial_3", "trial_x", "notes"]:
        (log_dir / name).mkdir()
    second = TensorBoardBackend(run)
    second.start()

    assert Path(first.experiment.logdir) == log_dir / "trial_0"
    assert Path(second.experiment.logdir) == log_dir / "trial_4"


def test_the_logged_values_reach_the_event_file(
    backend: TensorBoardBackend, image: npt.NDArray[np.uint8]
):
    matrix = np.arange(1200).reshape(40, 30)
    backend.log_metrics({"loss": 0.5, "acc": 0.9}, 3)
    backend.log_image("val/image", image, 4)
    backend.log_matrix(matrix, "matrix", 5, {"ignored": 1})
    backend.close("success")

    values = read_summaries(Path(backend.experiment.logdir))
    by_tag = {value.tag: (step, value) for step, value in values}
    assert by_tag["loss"][0] == 3
    assert by_tag["loss"][1].simple_value == 0.5
    assert by_tag["acc"][1].simple_value == pytest.approx(0.9)
    assert by_tag["val/image"][0] == 4
    assert by_tag["val/image"][1].image.height == 4
    assert by_tag["val/image"][1].image.width == 5
    step, text = by_tag["matrix/text_summary"]
    assert step == 5
    # the whole matrix, not the abbreviated form of `array2string`
    assert "1199" in text.tensor.string_val[0].decode()
    assert "..." not in text.tensor.string_val[0].decode()


def test_the_hyperparameters_stay_in_the_run(
    backend: TensorBoardBackend, run: RunContext
):
    """`add_hparams` would write them into a new run directory, which
    TensorBoard shows as a second run. TensorBoard reads only the first
    set in a run, so the backend writes all sets at once on close.
    """
    backend.log_hyperparams({"lr": 0.1, "layers": [1, 2]})
    backend.log_hyperparams({"note": None})
    log_dir = run.save_directory / "tensorboard_logs" / run.run_name
    backend.experiment.flush()
    assert read_summaries(log_dir) == []

    backend.close("success")

    assert [path for path in log_dir.iterdir() if path.is_dir()] == []
    values = read_summaries(log_dir)
    starts = [
        value
        for _, value in values
        if value.tag == "_hparams_/session_start_info"
    ]
    assert len(starts) == 1
    content = starts[0].metadata.plugin_data.content
    assert b"lr" in content
    assert b"[1, 2]" in content
    assert b"None" in content
    assert "_hparams_/experiment" in {value.tag for _, value in values}


def test_a_run_without_hyperparameters_writes_none(
    backend: TensorBoardBackend, run: RunContext
):
    backend.close("success")

    log_dir = run.save_directory / "tensorboard_logs" / run.run_name
    assert read_summaries(log_dir) == []


def test_an_artifact_is_not_stored(
    backend: TensorBoardBackend, run: RunContext, tmp_path: Path
):
    artifact = tmp_path / "model.txt"
    artifact.write_text("weights")

    backend.upload_artifact(artifact, None, "weights")
    backend.close("success")

    log_dir = run.save_directory / "tensorboard_logs" / run.run_name
    assert read_summaries(log_dir) == []


def test_the_writer_exists_only_after_start(run: RunContext):
    with pytest.raises(RuntimeError, match="not started"):
        _ = TensorBoardBackend(run).experiment


def test_a_missing_sdk_names_the_extra(
    run: RunContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "tensorboardX", None)

    with pytest.raises(ImportError, match=r"luxonis-ml\[tensorboard\]"):
        TensorBoardBackend(run).start()
