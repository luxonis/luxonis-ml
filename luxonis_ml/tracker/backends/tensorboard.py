# pyright: strict
# tensorboardX carries no type annotations. Strict mode would report each
# use of it, so these rules are off in this file.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The TensorBoard backend of the tracker.

`TensorBoardBackend` writes TensorBoard event files through
`tensorboardX`_. It turns on with ``LuxonisTracker(tensorboard=True)``
and needs the ``tensorboard`` extra.

.. _tensorboardX:
    https://github.com/lanpa/tensorboardX

See:
    `luxonis_ml.tracker` for what each backend does with each logging
    call.

"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from luxonis_ml.guard_extras import guard_missing_extra
from luxonis_ml.typing import ParamValue

from .base import RunContext, RunStatus, TrackerBackend

if TYPE_CHECKING:
    from tensorboardX import SummaryWriter


class TensorBoardBackend(TrackerBackend, register_name="tensorboard"):
    """Write TensorBoard event files through ``tensorboardX``.

    The events go to ``<save_directory>/tensorboard_logs/<run_name>``. A
    sweep gives each trial its own ``trial_<n>`` directory below that.

    The backend has no options. TensorBoard stores no files, so
    ``upload_artifact`` does nothing, and the hyperparameters reach the
    event file only when the run closes.

    """

    def __init__(self, run: RunContext) -> None:
        """Store the run.

        Args:
            run: The run to log to.

        """
        super().__init__(run)
        self._writer: SummaryWriter | None = None
        self._hparams: dict[str, ParamValue] = {}

    @property
    def experiment(self) -> "SummaryWriter":
        """The ``SummaryWriter`` of the run.

        Raises:
            RuntimeError: If the backend is not started.

        """
        if self._writer is None:
            raise RuntimeError("The TensorBoard backend is not started.")
        return self._writer

    def start(self) -> None:
        """Open the event file of the run.

        A sweep trial takes the next free ``trial_<n>`` directory.

        Raises:
            ImportError: If ``tensorboardX`` is not installed.

        """
        with guard_missing_extra("tensorboard"):
            from tensorboardX import SummaryWriter

        log_dir = (
            self.run.save_directory / "tensorboard_logs" / self.run.run_name
        )
        if self.run.is_sweep:
            log_dir /= f"trial_{_next_trial(log_dir)}"
        self._writer = SummaryWriter(logdir=str(log_dir))

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        """Collect the hyperparameters for the HParams dashboard.

        The dashboard reads only the first set of hyperparameters in a
        run, so the backend adds each call to one set, and `close` writes
        that set.

        Args:
            params: The hyperparameters, keyed by name.

        """
        self._hparams.update(params)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Write each metric as a scalar.

        Args:
            metrics: The metric values, keyed by metric name, which is
                the tag of the scalar.
            step: The training step of the values.

        """
        for name, value in metrics.items():
            self.experiment.add_scalar(name, value, step)

    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        r"""Write the image.

        Args:
            name: Name of the image, which is its tag.
            image: The image, of shape :math:`\left(H, W, C\right)`.
            step: The training step of the image.

        """
        self.experiment.add_image(name, image, step, dataformats="HWC")

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        """Write the matrix as text.

        The text holds every element, also of a large matrix.

        Args:
            matrix: The matrix.
            name: Name of the matrix, which is the tag of the text.
            step: The training step of the matrix.
            extra_data: Ignored, because the text has no place for it.

        """
        text = np.array2string(matrix, separator=", ", threshold=matrix.size)
        self.experiment.add_text(name, text, step)

    def close(self, status: RunStatus) -> None:
        """Write the hyperparameters, and close the event file.

        Args:
            status: Ignored, because TensorBoard has no run status.

        """
        if self._hparams:
            self._write_hparams()
        self.experiment.close()

    def _write_hparams(self) -> None:
        """Write the hyperparameters into the event file of the run.

        TensorBoard accepts only scalar values, so any other value
        becomes a string. The dashboard shows only a run that declares a
        metric, so the run declares ``placeholder_metric``.
        """
        from tensorboardX.summary import hparams

        scalars = {
            key: value
            if isinstance(value, bool | int | float | str)
            else str(value)
            for key, value in self._hparams.items()
        }
        # `add_hparams` would write into a new run directory of its own
        file_writer = self.experiment.file_writer
        # the constructor of the writer creates the file writer
        assert file_writer is not None
        for summary in hparams(scalars, {"placeholder_metric": 0}):
            file_writer.add_summary(summary)


def _next_trial(log_dir: Path) -> int:
    """Return the number of the next sweep trial in ``log_dir``."""
    if not log_dir.exists():
        return 0
    trials = [
        int(match[1])
        for path in log_dir.iterdir()
        if (match := re.fullmatch(r"trial_(\d+)", path.name))
    ]
    return max(trials, default=-1) + 1
