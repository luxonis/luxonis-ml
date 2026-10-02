# pyright: strict
"""The base class, the registry and the run of the tracker backends.

A backend connects `LuxonisTracker` to one tracking service. Each
backend is a subclass of `TrackerBackend`. The subclass registers itself
in `TRACKER_BACKENDS` when Python creates the class, and the name that
it registers under is the keyword argument of `LuxonisTracker` that
turns the backend on. Each backend receives a `RunContext`, which
describes the run that it logs to.

See:
    `luxonis_ml.tracker` for the built-in backends and for a complete
    example of a custom backend.

"""

from abc import ABC, abstractmethod
from collections.abc import Container, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeAlias

import numpy.typing as npt

from luxonis_ml.typing import ParamValue
from luxonis_ml.utils.registry import AutoRegisterMeta, Registry

RunStatus: TypeAlias = Literal["success", "failed"]
"""The final state of a run, as `TrackerBackend.close` receives it.

`LuxonisTracker.close` turns ``"success"`` and ``"finished"`` into
``"success"``, and every other status into ``"failed"``.
"""

TRACKER_BACKENDS: Registry[type["TrackerBackend"]] = Registry(
    name="tracker_backends"
)
"""The backends that `LuxonisTracker` can create, keyed by name.

Each subclass of `TrackerBackend` registers itself here when Python
creates the class. The name is the ``register_name`` class argument, or
else the name of the class. A plugin of the ``tracker_plugins`` entry
points replaces the backend of the same name. A subclass with
``register=False`` stays out of the registry.
"""


@dataclass(frozen=True)
class RunContext:
    """The run that a backend logs to.

    `LuxonisTracker` creates one `RunContext` and gives it to each of its
    backends.

    Attributes:
        run_name: Name of the run.
        save_directory: Root directory of the local run outputs.
        project_name: Project name, if the caller gave one.
        project_id: Project identifier, if the caller gave one.
        run_id: Identifier of an earlier run to continue, if the caller
            gave one.
        is_sweep: Whether the run is one trial of a sweep.

    """

    run_name: str
    save_directory: Path
    project_name: str | None = None
    project_id: str | None = None
    run_id: str | None = None
    is_sweep: bool = False

    @property
    def run_directory(self) -> Path:
        """The local directory of the run, ``<save_directory>/<run_name>``."""
        return self.save_directory / self.run_name


class TrackerBackend(
    ABC, metaclass=AutoRegisterMeta, registry=TRACKER_BACKENDS, register=False
):
    """One tracking service behind `LuxonisTracker`.

    `LuxonisTracker` uses a backend in this order:

        1. It creates the backend on every rank, with the options of its
           keyword argument. The constructor only checks and stores the
           options, so it does no I/O.
        2. It calls `start` once, on rank :math:`0` only, before the
           first logging call.
        3. It calls the logging methods, on rank :math:`0` only.
        4. It calls `close` once, when the run ends, if `start`
           succeeded.

    A subclass registers itself in `TRACKER_BACKENDS`. Give it a short
    name with the ``register_name`` class argument, because the name is
    the keyword argument that turns the backend on:

    .. code-block:: python

        class MyServiceBackend(TrackerBackend, register_name="my_service"): ...


        tracker = LuxonisTracker(my_service={"api_key": key})

    A subclass with ``register=False`` stays out of the registry, for
    example a helper base class. `luxonis_ml.tracker` shows a complete
    backend.

    Attributes:
        run: The run that the backend logs to.
        buffered: Whether `LuxonisTracker` wraps the backend in a
            `BufferedBackend`. Set it for a remote service that can be
            unreachable for a while. The default is ``False``.

    """

    buffered: ClassVar[bool] = False

    def __init__(self, run: RunContext) -> None:
        """Store the run.

        A subclass takes its options as keyword arguments, checks them,
        and calls this constructor. It does no I/O, because the tracker
        creates the backend on every rank.

        Args:
            run: The run that the backend logs to.

        """
        self.run = run

    @abstractmethod
    def start(self) -> None:
        """Connect to the service and open the run.

        `BufferedBackend` calls it again after a transient failure, so a
        failed start must leave the backend ready for the next attempt.

        Raises:
            Exception: Any error of the service. `is_transient` tells
                whether a later attempt can succeed.

        """

    @abstractmethod
    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        """Log the hyperparameters of the run.

        The tracker can call it more than once in a run, and each call
        adds to the earlier ones.

        Args:
            params: The hyperparameters, keyed by name. A value can be
                any value of a YAML configuration, such as a list.

        Raises:
            Exception: Any error of the service. `is_transient` tells
                whether a later attempt can succeed.

        """

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log scalar metrics.

        Args:
            metrics: The metric values, keyed by metric name.
            step: The training step of the values.

        Raises:
            Exception: Any error of the service. `is_transient` tells
                whether a later attempt can succeed.

        """

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log one scalar metric.

        The default calls `log_metrics`.

        Args:
            name: Name of the metric.
            value: Value of the metric.
            step: The training step of the value.

        Raises:
            Exception: Any error of the service. `is_transient` tells
                whether a later attempt can succeed.

        """
        self.log_metrics({name: value}, step)

    @abstractmethod
    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        r"""Log an image.

        Args:
            name: Name of the image. It can hold ``/`` to group images.
            image: The image, of shape :math:`\left(H, W, C\right)`.
            step: The training step of the image.

        Raises:
            Exception: Any error of the service. `is_transient` tells
                whether a later attempt can succeed.

        """

    def log_images(
        self, images: Mapping[str, npt.NDArray[Any]], step: int
    ) -> None:
        r"""Log several images.

        The default calls `log_image` for each image. A service that
        takes several images in one request can override it.

        Args:
            images: The images, keyed by name. Each image has the shape
                :math:`\left(H, W, C\right)`.
            step: The training step of the images.

        Raises:
            Exception: Any error of the service. `is_transient` tells
                whether a later attempt can succeed.

        """
        for name, image in images.items():
            self.log_image(name, image, step)

    @abstractmethod
    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        r"""Log a matrix, such as a confusion matrix.

        Args:
            matrix: The matrix, usually of shape :math:`\left(M, N\right)`.
            name: Name of the matrix.
            step: The training step of the matrix.
            extra_data: More data to store with the matrix, such as the
                class names. A service that has no place for it can
                ignore it.

        Raises:
            Exception: Any error of the service. `is_transient` tells
                whether a later attempt can succeed.

        """

    def upload_artifact(self, path: Path, name: str | None, typ: str) -> None:
        """Upload a file.

        The default does nothing, for a service that stores no files.

        Args:
            path: Path to the file.
            name: Name to store the file under. ``None`` keeps the name
                of the file.
            typ: Kind of the artifact, such as ``"weights"``.

        """
        return

    def flush(self) -> None:
        """Write the pending data now, and keep the run open.

        The default does nothing, for a service that receives each call
        at once.
        """
        return

    @abstractmethod
    def close(self, status: RunStatus) -> None:
        """Send the pending data and end the run.

        Args:
            status: The final state of the run.

        Raises:
            Exception: Any error of the service. `is_transient` tells
                whether a later attempt can succeed.

        """

    def is_transient(self, error: Exception) -> bool:
        """Tell whether a failed call can succeed when it is sent again.

        `BufferedBackend` keeps a call that failed with a transient error
        and drops the others. The default treats an ``OSError`` as
        transient, which covers the network errors of ``socket`` and
        ``requests``. An ``HTTPError`` of ``requests`` is transient only
        for a 5xx status and for 429, because with any other status the
        server rejected the call. The errors of a local file are not
        transient: a missing file, a denied access, or a directory in
        place of a file.

        Args:
            error: The error of a call to the service.

        Returns:
            ``True`` if a later attempt of the same call can succeed.

        """
        status = _http_status(error)
        if status is not None:
            return self._is_outage_status(status)
        return isinstance(error, OSError) and not isinstance(
            error,
            FileNotFoundError
            | PermissionError
            | IsADirectoryError
            | NotADirectoryError,
        )

    @staticmethod
    def _is_outage_status(status: int) -> bool:
        """Tell whether an HTTP status means that the server cannot take
        the call now: a 5xx status, or 429 for too many requests.
        """
        return status >= 500 or status == 429


def _http_status(error: Exception) -> int | None:
    """Return the HTTP status of an ``HTTPError`` of ``requests``."""
    try:
        from requests import HTTPError
    except ImportError:
        return None
    if isinstance(error, HTTPError) and error.response is not None:
        return error.response.status_code
    return None


def check_options(
    options: Mapping[str, object], known: Container[str]
) -> None:
    """Reject an option that a backend does not take.

    A backend that takes its options as ``**options: Unpack[...]`` calls
    it in its constructor. ``Unpack`` of a ``TypedDict`` checks the
    options only for pyright, and a mapping from a configuration file
    reaches the backend unchecked.

    Args:
        options: The options that the backend received.
        known: The names of the options that the backend takes.

    Raises:
        TypeError: If ``options`` has a key that is not in ``known``.

    """
    for key in options:
        if key not in known:
            raise TypeError(f"The backend got an unknown option '{key}'.")
