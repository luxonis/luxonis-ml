# pyright: strict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt
from typing_extensions import Unpack

from luxonis_ml.guard_extras import guard_missing_extra
from luxonis_ml.typing import ParamValue

from .base import RunContext, RunStatus, TrackerBackend

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class WandbOptions(TypedDict, total=False):
    """The options of `WandbBackend`.

    Attributes:
        entity: The WandB user or team. ``None`` uses the default entity
            of the logged-in user.

    """

    entity: str | None


class WandbBackend(TrackerBackend):
    """Log to `Weights & Biases`_.

    The local files of WandB go to ``<save_directory>/wandb_logs``.

    The backend never passes ``step`` to WandB. WandB drops a call whose
    step is lower than the last one, and the callers do not keep one
    step counter for all calls. WandB counts the steps itself instead.

    .. _Weights & Biases:
        https://wandb.ai/site

    """

    def __init__(
        self, run: RunContext, **options: Unpack[WandbOptions]
    ) -> None:
        """Check the options.

        The ``project_name`` of the run, or else its ``project_id``,
        names the WandB project.

        Args:
            run: The run to log to.
            **options: See `WandbOptions`.

        Raises:
            ValueError: If the run has no project.

        """
        super().__init__(run)
        project = run.project_name or run.project_id
        if project is None:
            raise ValueError("WandB needs `project_name` or `project_id`.")
        self.project = project
        self.entity = options.get("entity")
        self._run: Run | None = None

    @property
    def experiment(self) -> "Run":
        """The WandB ``Run``."""
        if self._run is None:
            raise RuntimeError("The WandB backend is not started.")
        return self._run

    def start(self) -> None:
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
        # WandB leaves the argument of `update` unannotated
        self.experiment.config.update(  # pyright: ignore[reportUnknownMemberType]
            dict(params)
        )

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self.experiment.log(dict(metrics))

    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        import wandb

        self.experiment.log({name: wandb.Image(image, caption=name)})

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        """Log the matrix as a table. ``extra_data`` is not logged."""
        import wandb

        rows = np.atleast_2d(matrix)
        table = wandb.Table(
            columns=["Row Index", *(f"Col {i}" for i in range(rows.shape[1]))]
        )
        for i, row in enumerate(rows):
            table.add_data(i, *row)
        self.experiment.log({f"{name}_table": table})

    def upload_artifact(self, path: Path, name: str | None, typ: str) -> None:
        """Log the file as a WandB artifact.

        The artifact takes the last component of ``name``, or else the
        stem of the file, because WandB rejects a ``/`` in the name.
        """
        import wandb

        artifact = wandb.Artifact(
            name=Path(name).name if name else path.stem, type=typ
        )
        artifact.add_file(local_path=str(path))
        self.experiment.log_artifact(artifact)

    def close(self, status: RunStatus) -> None:
        self.experiment.finish(exit_code=0 if status == "success" else 1)
