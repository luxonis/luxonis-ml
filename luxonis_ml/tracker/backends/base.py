# pyright: strict
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeAlias

import numpy.typing as npt

from luxonis_ml.typing import ParamValue
from luxonis_ml.utils.registry import AutoRegisterMeta, Registry

RunStatus: TypeAlias = Literal["success", "failed"]
"""The final state of a run, as `TrackerBackend.close` receives it."""

TRACKER_BACKENDS: Registry[type["TrackerBackend"]] = Registry(
    name="tracker_backends"
)
"""The backends that `LuxonisTracker` can create, keyed by name.

Each subclass of `TrackerBackend` registers itself here when Python
creates the class, under its ``register_name`` class argument.
"""


@dataclass(frozen=True)
class RunContext:
    """The run that a backend logs to.

    Attributes:
        run_name: Name of the run.
        save_directory: Root directory of the local run outputs.
        project_name: Project name, if the caller gave one.
        project_id: Project identifier, if the caller gave one.
        run_id: Identifier of an earlier run to continue.
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
    """One logging service behind `LuxonisTracker`.

    The constructor only checks and stores the options, so a tracker can
    be created on every rank at no cost. `start` connects to the service.
    `LuxonisTracker` calls it once, before the first logging call, and
    only on rank :math:`0`.

    A subclass registers itself in `TRACKER_BACKENDS` under its
    ``register_name``, which is the keyword argument of `LuxonisTracker`
    that turns it on. A subclass with ``register=False`` stays out of
    the registry:

    .. code-block:: python

        class MyServiceBackend(TrackerBackend, register_name="my_service"):
            def __init__(self, run: RunContext, *, api_key: str) -> None:
                super().__init__(run)
                self.api_key = api_key

            ...


        tracker = LuxonisTracker(my_service={"api_key": key})

    Attributes:
        run: The run that the backend logs to.
        buffered: Whether `LuxonisTracker` wraps the backend in a
            `BufferedBackend`. Set it for a remote service that can be
            unreachable for a while.

    """

    buffered: ClassVar[bool] = False

    def __init__(self, run: RunContext) -> None:
        self.run = run

    @property
    @abstractmethod
    def experiment(self) -> object:
        """The native handle of the service, such as its client."""

    @abstractmethod
    def start(self) -> None:
        """Connect to the service and open the run."""

    @abstractmethod
    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        """Log the hyperparameters of the run."""

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log scalar metrics at ``step``."""

    @abstractmethod
    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        r"""Log an image of shape :math:`\left(H, W, C\right)`."""

    @abstractmethod
    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        """Log a matrix, such as a confusion matrix."""

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

    @abstractmethod
    def close(self, status: RunStatus) -> None:
        """Flush the pending data and end the run with ``status``."""

    def is_transient(self, error: Exception) -> bool:
        """Tell whether a failed call can succeed when it is sent again.

        `BufferedBackend` keeps a call that failed with a transient error
        and drops the others. The default treats an ``OSError`` as
        transient, which covers the network errors of ``socket`` and
        ``requests``, but not the errors of a local file: a missing
        file, a denied access, or a directory in place of a file.
        """
        return isinstance(error, OSError) and not isinstance(
            error,
            FileNotFoundError
            | PermissionError
            | IsADirectoryError
            | NotADirectoryError,
        )


def check_options(options: Mapping[str, object], known: Iterable[str]) -> None:
    """Reject an option that a backend does not take.

    ``Unpack`` of a ``TypedDict`` checks the options only for pyright. A
    mapping from a configuration file reaches the backend unchecked.

    Raises:
        TypeError: If ``options`` has a key that is not in ``known``.

    """
    for key in options:
        if key not in known:
            raise TypeError(f"The backend got an unknown option '{key}'.")
