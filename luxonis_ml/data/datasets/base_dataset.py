from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TypeAlias

from semver.version import Version

from luxonis_ml.data.datasets.source import LuxonisSource
from luxonis_ml.ldf import DatasetRecord, Skeleton
from luxonis_ml.typing import PathType
from luxonis_ml.utils import AutoRegisterMeta, Registry

DATASETS_REGISTRY: Registry[type["BaseDataset"]] = Registry(name="datasets")


DatasetIterator: TypeAlias = Iterator[dict | DatasetRecord]

SkeletonEdge: TypeAlias = tuple[int, int] | tuple[str, str]
"""A pair of keypoints, given either by index or by name."""


class BaseDataset(
    ABC, metaclass=AutoRegisterMeta, registry=DATASETS_REGISTRY, register=False
):
    """Base class for datasets in the Luxonis MLOps ecosystem."""

    @property
    @abstractmethod
    def identifier(self) -> str:
        """The unique identifier for the dataset."""
        ...

    @property
    @abstractmethod
    def version(self) -> Version:
        """The version of the underlying LDF."""
        ...

    @abstractmethod
    def set_tasks(self, tasks: dict[str, list[str]]) -> None:
        """Set dataset tasks.

        Args:
            tasks: Mapping from task names to task types.

        """
        ...

    @abstractmethod
    def get_tasks(self) -> dict[str, list[str]]:
        """Return task names and task types.

        Returns:
            Task types keyed by task name.

        """
        ...

    @abstractmethod
    def set_classes(
        self,
        classes: list[str] | dict[str, int],
        task: str | None = None,
    ) -> None:
        """Set classes for one or more tasks.

        Args:
            classes: Class names, or class IDs keyed by class name. If
                class names are provided, IDs are assigned
                alphabetically starting from :math:`0`. A class named
                ``"background"`` is always assigned ID :math:`0`.
            task: Optional task to update. If omitted, all tasks are
                updated.

        """
        ...

    @abstractmethod
    def get_classes(self) -> dict[str, dict[str, int]]:
        """Get class names and IDs per task.

        Returns:
            Mapping from class names to class IDs grouped by task name:

            .. python::

                {
                    "color": {"red": 0, "green": 1, "blue": 2},
                    "brand": {"audi": 0, "bmw": 1, "mercedes": 2},
                }

        """
        ...

    @abstractmethod
    def get_source_names(self) -> list[str]:
        """Return input source names for the dataset.

        Returns:
            Source names used to identify input data.

        """
        ...

    @abstractmethod
    def update_source(self, source: LuxonisSource) -> None:
        """Update the dataset source definition.

        Args:
            source: Source definition to store.

        """
        ...

    @abstractmethod
    def set_skeletons(
        self,
        labels: list[str] | None = None,
        edges: list[SkeletonEdge] | None = None,
        task: str | None = None,
        *,
        flip_pairs: list[SkeletonEdge] | None = None,
        sigmas: list[float] | None = None,
        infer_flip_pairs: bool = True,
    ) -> None:
        """Set keypoint skeleton semantics for tasks that use keypoints.

        Only the fields that are provided are replaced, so a skeleton can
        be built up over several calls.

        For example:

        .. python::

            dataset.set_skeletons(
                labels=["right hand", "right shoulder", ...],
                edges=[[0, 1], [4, 5], ...]
            )

        Edges and flip pairs may also refer to keypoints by name:

        .. python::

            dataset.set_skeletons(
                labels=["nose", "left_eye", "right_eye"],
                edges=[("nose", "left_eye"), ("nose", "right_eye")],
            )

        Args:
            labels: Optional keypoint names.
            edges: Optional edges between keypoints.
            task: Optional task to update. If omitted, all keypoint tasks
                are updated.
            flip_pairs: Optional pairs of keypoints swapped by a horizontal
                flip. Inferred from ``left``/``right`` names when omitted.
            sigmas: Optional per-keypoint OKS standard deviations.
            infer_flip_pairs: Whether to infer flip pairs from the keypoint
                names when none are known.

        Raises:
            ValueError: If none of the skeleton fields are provided.

        """
        ...

    @abstractmethod
    def get_skeletons(self) -> dict[str, Skeleton]:
        """Return keypoint skeletons for each task.

        Returns:
            Keypoint skeletons keyed by task name.

        """
        ...

    @abstractmethod
    def add(
        self, generator: DatasetIterator, batch_size: int = 1_000_000
    ) -> None:
        """Write annotations to parquet files.

        Args:
            generator: Iterator yielding ``DatasetRecord`` instances or
                dictionaries that can be converted to ``DatasetRecord``.
            batch_size: Number of records to buffer before processing.
                Lower values reduce peak memory usage.

        """
        ...

    @abstractmethod
    def make_splits(
        self,
        splits: dict[str, Sequence[PathType]]
        | dict[str, float]
        | tuple[float, float, float]
        | None = None,
        *,
        ratios: dict[str, float] | tuple[float, float, float] | None = None,
        definitions: dict[str, list[PathType]] | None = None,
        replace_old_splits: bool = False,
    ) -> None:
        """Generate dataset splits.

        Args:
            splits: Split definitions or ratios. Accepts explicit
                filepath lists, split ratios keyed by split name, or a
                ``(train, val, test)`` ratio tuple.
            ratios: Optional deprecated split ratios. Use ``splits``
                instead.
            definitions: Optional deprecated filepath split definitions.
                Use ``splits`` instead.
            replace_old_splits: Whether to replace existing split
                assignments instead of adding only new files.

        """
        ...

    @abstractmethod
    def delete_dataset(self) -> None:
        """Delete local files belonging to the dataset."""
        ...

    @staticmethod
    @abstractmethod
    def exists(dataset_name: str) -> bool:
        """Check whether a dataset exists.

        Args:
            dataset_name: Dataset name to check.

        Returns:
            ``True`` if the dataset exists, ``False`` otherwise.

        """
        ...

    def get_n_classes(self) -> dict[str, int]:
        """Return number of classes per task.

        Returns:
            Mapping from task names to class counts.

        """
        return {
            task_name: len(classes)
            for task_name, classes in self.get_classes().items()
        }

    def get_class_names(self) -> dict[str, list[str]]:
        """Return class names per task.

        Returns:
            Class names keyed by task name:

            .. python::

                {
                    "vehicles": ["red", "green", "blue"],
                    "brands": ["audi", "bmw", "mercedes"],
                }

        """
        return {
            task_name: list(classes.keys())
            for task_name, classes in self.get_classes().items()
        }

    def get_task_names(self) -> list[str]:
        """Return task names for the dataset.

        This is equivalent to `get_tasks` but returns only the task names.

        Returns:
            Task names.

        """
        return list(self.get_tasks().keys())

    def get_n_keypoints(self) -> dict[str, int]:
        """Return the number of keypoints for each task.

        Returns:
            Number of keypoints keyed by task name.

        """
        return {
            task: len(skeleton.labels)
            # A skeleton set from edges alone has no labels to count.
            or max((max(edge) for edge in skeleton.edges), default=-1) + 1
            for task, skeleton in self.get_skeletons().items()
        }
