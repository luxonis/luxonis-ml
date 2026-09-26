# pyright: strict
"""The Weights & Biases backend of the tracker.

`WandbBackend` logs a run to `Weights & Biases`_. It turns on with
``LuxonisTracker(wandb=True)``, or with a mapping of `WandbOptions`, and
needs the ``wandb`` extra.

.. _Weights & Biases:
    https://wandb.ai/site

See:
    `luxonis_ml.tracker` for what each backend does with each logging
    call.

"""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt
from typing_extensions import Unpack

from luxonis_ml.guard_extras import guard_missing_extra
from luxonis_ml.typing import ParamValue

from .base import RunContext, RunStatus, TrackerBackend, check_options

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class WandbOptions(TypedDict, total=False):
    """The options of `WandbBackend`.

    Attributes:
        entity: The WandB user or team. ``None`` uses the default entity
            of the logged-in user.

    """

    entity: str | None


class WandbBackend(TrackerBackend, register_name="wandb"):
    """Log to Weights & Biases.

    The ``project_name`` of the run, or else its ``project_id``, names
    the WandB project, and the run name names the WandB run. The local
    files of WandB go to ``<save_directory>/wandb_logs``.

    The backend never passes ``step`` to WandB. WandB drops a call whose
    step is lower than the last one, and the callers do not keep one
    step counter for all calls. WandB counts the steps itself instead.

    Attributes:
        project: The WandB project.
        entity: The WandB user or team, or ``None`` for the default.

    """

    def __init__(
        self, run: RunContext, **options: Unpack[WandbOptions]
    ) -> None:
        """Check the options.

        Args:
            run: The run to log to.
            **options: See `WandbOptions`.

        Raises:
            TypeError: If an option is unknown.
            ValueError: If the run has no project.

        """
        super().__init__(run)
        check_options(options, WandbOptions.__optional_keys__)
        project = run.project_name or run.project_id
        if project is None:
            raise ValueError("WandB needs `project_name` or `project_id`.")
        self.project = project
        self.entity = options.get("entity")
        self._run: Run | None = None

    @property
    def wandb_run(self) -> "Run":
        """The WandB ``Run``, for the calls that the tracker does not
        make, such as ``watch``.

        Raises:
            RuntimeError: If the backend is not started.

        """
        if self._run is None:
            raise RuntimeError("The WandB backend is not started.")
        return self._run

    def start(self) -> None:
        """Start the WandB run.

        Raises:
            ImportError: If ``wandb`` is not installed.
            Exception: Any error of ``wandb.init``, such as a failed
                login.

        """
        with guard_missing_extra("wandb"):
            import wandb

        log_dir = self.run.save_directory / "wandb_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            dir=log_dir,
            name=self.run.run_name,
        )

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        """Add the hyperparameters to the configuration of the run.

        Args:
            params: The hyperparameters, keyed by name.

        """
        # WandB leaves the argument of `update` unannotated
        self.wandb_run.config.update(  # pyright: ignore[reportUnknownMemberType]
            dict(params)
        )

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log the metrics at the next WandB step.

        Args:
            metrics: The metric values, keyed by metric name.
            step: Ignored. See the class description.

        """
        self.wandb_run.log(dict(metrics))

    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        r"""Log the image at the next WandB step.

        Args:
            name: Name of the image, which is its key and its caption.
            image: The image, of shape :math:`\left(H, W, C\right)`.
            step: Ignored. See the class description.

        """
        import wandb

        self.wandb_run.log({name: wandb.Image(image, caption=name)})

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        """Log the matrix as a table under ``<name>_table``.

        The table has a ``Row Index`` column and a ``Col <n>`` column for
        each column of the matrix. A 1-dimensional array is one row.

        Args:
            matrix: The matrix.
            name: Name of the matrix.
            step: Ignored. See the class description.
            extra_data: Ignored, because the table has no place for it.

        """
        import wandb

        rows = np.atleast_2d(matrix)
        table = wandb.Table(
            columns=["Row Index", *(f"Col {i}" for i in range(rows.shape[1]))]
        )
        for i, row in enumerate(rows):
            table.add_data(i, *row)
        self.wandb_run.log({f"{name}_table": table})

    def upload_artifact(self, path: Path, name: str | None, typ: str) -> None:
        """Log the file as a WandB artifact of the run.

        Args:
            path: Path to the file.
            name: Name of the artifact. WandB rejects a ``/``, so only
                the last component counts. ``None`` takes the stem of
                the file.
            typ: The type of the WandB artifact.

        """
        import wandb

        artifact = wandb.Artifact(
            name=Path(name).name if name else path.stem, type=typ
        )
        artifact.add_file(local_path=str(path))
        self.wandb_run.log_artifact(artifact)

    def close(self, status: RunStatus) -> None:
        """Finish the WandB run.

        Args:
            status: ``"success"`` finishes the run with the exit code
                :math:`0`, ``"failed"`` with :math:`1`.

        """
        self.wandb_run.finish(exit_code=0 if status == "success" else 1)
