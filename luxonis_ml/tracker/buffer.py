# pyright: strict
"""The buffer that keeps a backend running while its service is down.

`LuxonisTracker` wraps each backend that sets `TrackerBackend.buffered`
in a `BufferedBackend`. Of the built-in backends, only `MLflowBackend`
sets it. The wrapper keeps the calls that the service does not take,
sends them again later, and saves the calls that never get through.

See:
    `luxonis_ml.tracker` for where the saved calls go, and
    `TrackerBackend.is_transient` for how a backend tells an outage from
    a rejected call.

"""

import json
import math
import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from loguru import logger

from luxonis_ml.typing import ParamValue

from .backends.base import RunStatus, TrackerBackend


class BufferedBackend(TrackerBackend, register=False):
    """Keep the calls that a remote backend cannot send, and send them
    later.

    `LuxonisTracker` wraps each backend that sets
    `TrackerBackend.buffered`. A failed call does not reach the training
    loop:

        - A transient failure, as `TrackerBackend.is_transient` tells it,
          puts the call into a buffer. The wrapper then leaves the
          service alone for ``retry_interval`` seconds, so that an outage
          does not stall every logging call. The next call after that
          time sends the buffer first, in the original order.
        - Any other failure drops the call with a warning. The service
          rejected it, and it would fail again.
        - A rejected first `start` raises, so that a wrong configuration
          fails at once. When a start that follows an outage is
          rejected, the wrapper keeps the calls until `close`.

    The buffer holds at most 100 hyperparameter calls, 500 metric calls,
    50 images, 500 matrices and 10 artifacts. A full buffer drops the
    oldest call of that kind, and warns once for each outage. The wrapper
    keeps the default `TrackerBackend.log_metric` and
    `TrackerBackend.log_images`, so it sends them as `log_metrics` and
    `log_image` calls.

    An artifact is buffered as a hard link, or a copy, under
    ``<run_directory>/unsent_logs/<name>/artifacts/``, because callers
    often delete the file right after they hand it over. The other calls
    keep a copy of their arrays and mappings, because a caller can
    change them before a buffered call is sent.

    `close` tries the buffer one last time. What is still left goes to
    ``<run_directory>/unsent_logs/<name>/``:

        - ``calls.jsonl`` holds one line for each call, in order;
        - ``images/`` holds the images as ``.npy`` files, which keep the
          exact data.

    A signal handler can log, and close the run, while a call is sent.
    Its calls join the buffer, and its close waits until the call that
    it interrupted ends. A failure of that close gives a warning.

    Attributes:
        backend: The backend that the calls go to.
        name: Name of the backend, used in the messages and in the path
            of the unsent calls.
        retry_interval: Seconds to wait after a transient failure before
            the next attempt.
        unsent_directory: Where `close` saves the calls that never got
            through, ``<run_directory>/unsent_logs/<name>``.

    """

    def __init__(
        self, backend: TrackerBackend, name: str, retry_interval: float = 60
    ) -> None:
        """Wrap a backend.

        Args:
            backend: The backend to send the calls to.
            name: Name of the backend, used in the messages and in the
                path of the unsent calls.
            retry_interval: Seconds to wait after a transient failure
                before the next attempt.

        """
        super().__init__(backend.run)
        self.backend = backend
        self.name = name
        self.retry_interval = retry_interval
        self.unsent_directory = (
            backend.run.run_directory / "unsent_logs" / name
        )
        self._calls: list[_Call] = []
        self._started = False
        self._busy = False
        self._pending_close: RunStatus | None = None
        self._retry_at = 0.0
        self._reported_drop = False

    def start(self) -> None:
        """Start the wrapped backend, and send the buffered calls.

        A transient failure only schedules the next attempt, which the
        next call makes after ``retry_interval`` seconds.

        Raises:
            Exception: The error of the wrapped backend, if it rejects
                the start. A wrong configuration then fails at once.

        """
        if self._busy:
            return
        with self._hold():
            self._flush(raise_rejected=True)

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        """Send the hyperparameters, or buffer them.

        Args:
            params: The hyperparameters, keyed by name.

        """
        self._submit(_Hyperparams(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Send the metrics, or buffer them.

        Args:
            metrics: The metric values, keyed by metric name.
            step: The training step of the values.

        """
        self._submit(_Metrics(metrics, step))

    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        r"""Send the image, or buffer it.

        Args:
            name: Name of the image.
            image: The image, of shape :math:`\left(H, W, C\right)`.
            step: The training step of the image.

        """
        self._submit(_Image(name, image, step))

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        """Send the matrix, or buffer it.

        Args:
            matrix: The matrix.
            name: Name of the matrix.
            step: The training step of the matrix.
            extra_data: More data to store with the matrix.

        """
        self._submit(_Matrix(matrix, name, step, extra_data))

    def upload_artifact(self, path: Path, name: str | None, typ: str) -> None:
        """Upload the file, or buffer a link or a copy of it.

        Args:
            path: Path to the file.
            name: Name to store the file under. ``None`` keeps the name
                of the file.
            typ: Kind of the artifact, such as ``"weights"``.

        """
        self._submit(_Artifact(path, name, typ))

    def close(self, status: RunStatus) -> None:
        """Send the buffer one last time, save the rest, and end the run.

        The last attempt ignores the retry interval. The calls that still
        fail go to `unsent_directory`. The wrapped backend closes only if
        it started.

        Args:
            status: The final state of the run.

        Raises:
            OSError: If the calls that never got through cannot be
                saved. The run ends all the same.
            Exception: The error of the close of the wrapped backend.

        """
        if self._busy:
            self._pending_close = status
            return
        # the last chance, whatever the backoff says
        self._retry_at = 0
        try:
            self._flush()
        finally:
            try:
                self._spill()
            finally:
                if self._started:
                    self.backend.close(status)

    def flush(self) -> None:
        """Send the buffered calls, and flush the wrapped backend.

        The buffer waits for the retry interval, as at a logging call.

        Raises:
            Exception: The error of the flush of the wrapped backend.

        """
        if self._busy:
            return
        with self._hold():
            self._flush()
            if self._started:
                self.backend.flush()

    def is_transient(self, error: Exception) -> bool:
        """Ask the wrapped backend whether the error is transient.

        Args:
            error: The error of a call to the service.

        Returns:
            The answer of the wrapped backend.

        """
        return self.backend.is_transient(error)

    @contextmanager
    def _hold(self) -> Generator[None, None, None]:
        """Keep the calls and the close of a signal handler until the
        block ends.
        """
        self._busy = True
        try:
            yield
        finally:
            self._busy = False
            if self._pending_close is not None:
                status, self._pending_close = self._pending_close, None
                try:
                    self.close(status)
                except Exception as error:
                    logger.warning(
                        f"Could not close the {self.name} run: {error}"
                    )

    def _submit(self, call: "_Call") -> None:
        """Send the call, or buffer it when that is not possible now."""
        if self._busy:
            # a signal handler logs while a call is sent
            self._buffer(call)
            return
        with self._hold():
            try:
                self._flush()
            except BaseException:
                # an interrupt of the replay keeps the new call too
                self._buffer(call)
                raise
            if self._calls or not self._started:
                self._buffer(call)
                return
            try:
                call.send(self.backend)
            except Exception as error:
                if not self.backend.is_transient(error):
                    logger.warning(f"{self.name} rejected a call: {error}")
                    return
                self._back_off(error)
                self._buffer(call)
            except BaseException:
                self._buffer(call)
                raise

    def _flush(self, *, raise_rejected: bool = False) -> None:
        """Start the backend if needed, and send the buffered calls."""
        if time.monotonic() < self._retry_at:
            return
        if not self._started and not self._start(raise_rejected):
            return
        while self._calls:
            call = self._calls.pop(0)
            try:
                call.send(self.backend)
            except Exception as error:
                if self.backend.is_transient(error):
                    self._calls.insert(0, call)
                    self._back_off(error)
                    return
                logger.warning(
                    f"{self.name} rejected a buffered call: {error}"
                )
            except BaseException:
                self._calls.insert(0, call)
                raise
            call.discard()
        self._reported_drop = False

    def _start(self, raise_rejected: bool) -> bool:
        """Start the backend, and return whether it started."""
        try:
            self.backend.start()
        except Exception as error:
            if self.backend.is_transient(error):
                self._back_off(error)
            elif raise_rejected:
                raise
            else:
                # the configuration is wrong, and a retry cannot help
                self._retry_at = math.inf
                logger.error(
                    f"{self.name} rejected the run: {error}. The calls are "
                    "kept until the tracker closes."
                )
            return False
        self._started = True
        return True

    def _back_off(self, error: Exception) -> None:
        """Leave the service alone for ``retry_interval`` seconds."""
        self._retry_at = time.monotonic() + self.retry_interval
        logger.warning(
            f"{self.name} is unavailable: {error}. The calls are buffered, "
            f"the next attempt is in {self.retry_interval:.0f} seconds."
        )

    def _buffer(self, call: "_Call") -> None:
        """Add the call to the buffer, and drop the oldest call of its
        kind when the kind is over its limit.
        """
        call = call.copy()
        if isinstance(call, _Artifact):
            call = self._keep_file(call)
        self._calls.append(call)
        same_kind = [c for c in self._calls if type(c) is type(call)]
        if len(same_kind) <= call.limit:
            return
        oldest = same_kind[0]
        self._calls.remove(oldest)
        oldest.discard()
        if not self._reported_drop:
            self._reported_drop = True
            logger.warning(
                f"The {self.name} buffer holds {call.limit} calls of this "
                "kind already, dropping the oldest. Further drops in this "
                "outage are not reported."
            )

    def _keep_file(self, call: "_Artifact") -> "_Artifact":
        """Link or copy the artifact, so that it outlives the original.

        A hard link shares the data with the original, so a rewrite of
        the original in place changes what is sent later.
        """
        directory = self.unsent_directory / "artifacts" / uuid4().hex
        target = directory / call.path.name
        try:
            directory.mkdir(parents=True)
            try:
                target.hardlink_to(call.path)
            except OSError:
                shutil.copy2(call.path, target)
        except OSError as error:
            shutil.rmtree(directory, ignore_errors=True)
            logger.warning(
                f"Could not keep a copy of '{call.path}'. The artifact is "
                f"lost if the file disappears: {error}"
            )
            return call
        return _Artifact(target, call.name, call.typ, owned=True)

    def _spill(self) -> None:
        """Save the buffered calls to `unsent_directory`."""
        if not self._calls:
            return
        self.unsent_directory.mkdir(parents=True, exist_ok=True)
        path = self.unsent_directory / "calls.jsonl"
        with path.open("a") as file:
            for call in self._calls:
                record = call.record(self.unsent_directory)
                file.write(json.dumps(record, default=_to_json) + "\n")
        logger.warning(
            f"{len(self._calls)} calls never reached {self.name}. "
            f"They are saved in '{path}'."
        )
        self._calls.clear()


class _Call(ABC):
    """A logging call that `BufferedBackend` keeps until the service
    takes it.
    """

    limit: ClassVar[int]
    """How many calls of this kind the buffer holds."""

    @abstractmethod
    def send(self, backend: TrackerBackend) -> None:
        """Make the call on ``backend``."""

    @abstractmethod
    def record(self, directory: Path) -> dict[str, ParamValue]:
        """Describe the call for ``calls.jsonl``.

        Large data goes into a file under ``directory``.
        """

    def discard(self) -> None:
        """Delete the files that the call owns."""
        return

    def copy(self) -> "_Call":
        """Return the call with its own copy of the arrays and mappings
        of the caller.
        """
        return self


@dataclass(frozen=True, eq=False)
class _Hyperparams(_Call):
    """A buffered `TrackerBackend.log_hyperparams` call."""

    limit: ClassVar[int] = 100
    params: Mapping[str, ParamValue]

    def send(self, backend: TrackerBackend) -> None:
        backend.log_hyperparams(self.params)

    def record(self, directory: Path) -> dict[str, ParamValue]:
        return {"call": "log_hyperparams", "params": dict(self.params)}

    def copy(self) -> "_Hyperparams":
        return _Hyperparams(dict(self.params))


@dataclass(frozen=True, eq=False)
class _Metrics(_Call):
    """A buffered `TrackerBackend.log_metrics` call."""

    limit: ClassVar[int] = 500
    metrics: Mapping[str, float]
    step: int

    def send(self, backend: TrackerBackend) -> None:
        backend.log_metrics(self.metrics, self.step)

    def record(self, directory: Path) -> dict[str, ParamValue]:
        return {
            "call": "log_metrics",
            "metrics": dict(self.metrics),
            "step": self.step,
        }

    def copy(self) -> "_Metrics":
        return _Metrics(dict(self.metrics), self.step)


@dataclass(frozen=True, eq=False)
class _Image(_Call):
    """A buffered `TrackerBackend.log_image` call."""

    limit: ClassVar[int] = 50
    name: str
    image: npt.NDArray[Any]
    step: int

    def send(self, backend: TrackerBackend) -> None:
        backend.log_image(self.name, self.image, self.step)

    def record(self, directory: Path) -> dict[str, ParamValue]:
        path = directory / "images" / f"{uuid4().hex}.npy"
        path.parent.mkdir(exist_ok=True)
        np.save(path, self.image)
        return {
            "call": "log_image",
            "name": self.name,
            "step": self.step,
            "image": str(path),
        }

    def copy(self) -> "_Image":
        return _Image(self.name, self.image.copy(), self.step)


@dataclass(frozen=True, eq=False)
class _Matrix(_Call):
    """A buffered `TrackerBackend.log_matrix` call."""

    limit: ClassVar[int] = 500
    matrix: npt.NDArray[Any]
    name: str
    step: int
    extra_data: Mapping[str, ParamValue]

    def send(self, backend: TrackerBackend) -> None:
        backend.log_matrix(self.matrix, self.name, self.step, self.extra_data)

    def record(self, directory: Path) -> dict[str, ParamValue]:
        return {
            "call": "log_matrix",
            "name": self.name,
            "step": self.step,
            "matrix": self.matrix.tolist(),
            "extra_data": dict(self.extra_data),
        }

    def copy(self) -> "_Matrix":
        return _Matrix(
            self.matrix.copy(), self.name, self.step, dict(self.extra_data)
        )


@dataclass(frozen=True, eq=False)
class _Artifact(_Call):
    """A buffered `TrackerBackend.upload_artifact` call.

    Attributes:
        path: Path to the file, or to the copy that the buffer keeps.
        name: Name to store the file under.
        typ: Kind of the artifact.
        owned: Whether ``path`` is a copy that the buffer made.

    """

    limit: ClassVar[int] = 10
    path: Path
    name: str | None
    typ: str
    owned: bool = False

    def send(self, backend: TrackerBackend) -> None:
        backend.upload_artifact(self.path, self.name, self.typ)

    def record(self, directory: Path) -> dict[str, ParamValue]:
        return {
            "call": "upload_artifact",
            "path": str(self.path),
            "name": self.name,
            "typ": self.typ,
        }

    def discard(self) -> None:
        if self.owned:
            shutil.rmtree(self.path.parent, ignore_errors=True)


def _to_json(value: object) -> ParamValue:
    """Convert the values that ``json`` does not know.

    A mapping becomes a ``dict``, and an array, such as a NumPy scalar,
    becomes a number or a list. Any other value becomes its string.
    """
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, ParamValue], value))
    if hasattr(value, "__array__"):
        return np.asarray(value).tolist()
    return str(value)
