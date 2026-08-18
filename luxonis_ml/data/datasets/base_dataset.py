from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import TypeAlias

from semver.version import Version
from typing_extensions import deprecated

from luxonis_ml.data.datasets.source import LuxonisSource
from luxonis_ml.ldf import DatasetRecord, KeypointMetadata
from luxonis_ml.typing import PathType
from luxonis_ml.utils import AutoRegisterMeta, Registry

DATASETS_REGISTRY: Registry[type["BaseDataset"]] = Registry(name="datasets")


DatasetIterator: TypeAlias = Iterator[dict | DatasetRecord]

KeypointPair: TypeAlias = tuple[int, int] | tuple[str, str]
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
    def set_keypoint_metadata(
        self,
        labels: list[str] | None = None,
        edges: list[KeypointPair] | None = None,
        task: str | None = None,
        *,
        flip_pairs: list[KeypointPair] | None = None,
        sigmas: list[float] | None = None,
        infer_flip_pairs: bool = True,
    ) -> None:
        """Set the keypoint definitions of the tasks that use keypoints.

        Only the fields that you provide are replaced, so a definition can
        be built up over several calls.

        Prefer the records. A record can carry ``edges``, ``flip_pairs``
        and ``sigmas`` beside its keypoints, and `add` moves them here.

        For example:

        .. python::

            dataset.set_keypoint_metadata(
                labels=["right hand", "right shoulder", ...],
                edges=[[0, 1], [4, 5], ...]
            )

        Edges and flip pairs may also refer to keypoints by name:

        .. python::

            dataset.set_keypoint_metadata(
                labels=["nose", "left_eye", "right_eye"],
                edges=[("nose", "left_eye"), ("nose", "right_eye")],
            )

        Args:
            labels: Optional keypoint names.
            edges: Optional edges between keypoints.
            task: Optional task to update. If omitted, all tasks are
                updated.
            flip_pairs: Optional pairs of keypoints swapped by a horizontal
                flip. Inferred from ``left``/``right`` names when omitted.
            sigmas: Optional per-keypoint OKS standard deviations.
            infer_flip_pairs: Whether to infer flip pairs from the keypoint
                names when none are known.

        Raises:
            ValueError: If you provide none of the fields.

        """
        ...

    @abstractmethod
    def get_keypoint_metadata(self) -> dict[str, KeypointMetadata]:
        """Return the keypoint definition of each task.

        Returns:
            Keypoint metadata keyed by task name.

        """
        ...

    @deprecated("Use `set_keypoint_metadata` instead.")
    def set_skeletons(
        self,
        labels: list[str] | None = None,
        edges: list[KeypointPair] | None = None,
        task: str | None = None,
        *,
        flip_pairs: list[KeypointPair] | None = None,
        sigmas: list[float] | None = None,
        infer_flip_pairs: bool = True,
    ) -> None:
        """Set the keypoint definitions of the tasks that use keypoints.

        .. deprecated:: 0.10.0
            Use `set_keypoint_metadata`, or declare the keypoints on the
            records.
        """
        self.set_keypoint_metadata(
            labels,
            edges,
            task,
            flip_pairs=flip_pairs,
            sigmas=sigmas,
            infer_flip_pairs=infer_flip_pairs,
        )

    @deprecated("Use `get_keypoint_metadata` instead.")
    def get_skeletons(self) -> dict[str, KeypointMetadata]:
        """Return the keypoint definition of each task.

        .. deprecated:: 0.10.0
            Use `get_keypoint_metadata`.
        """
        return self.get_keypoint_metadata()

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
        splits: Mapping[str, Sequence[PathType]]
        | Mapping[str, float]
        | tuple[float, float, float]
        | None = None,
        *,
        replace_old_splits: bool = False,
    ) -> None:
        """Create dataset splits for training, validation, and testing.

        Note:
            Although ``"train"``, ``"val"``, and ``"test"``
            are the conventional split names, you can use any split names
            you want by providing a mapping to the ``splits`` argument.
            This can be useful for combining records from multiple
            sources (``"train_real"``, ``"train_synth"``) or for
            creating fully custom splits.

        Args:
            splits: A mapping defining the splits. Can be one of the following:

                - A mapping of split names to lists of file paths.
                - A mapping of split names to ratios.
                - A tuple of three ratios for train, val, and test splits.

                A ratio is a number from 0 to 1, and the ratios sum to
                1. ``1`` and ``1.0`` both work.

            replace_old_splits: Whether to replace old splits with new ones.
                If ``False`` (default), new splits are added to the existing
                splits. If ``True``, the existing splits are discarded first.

        Raises:
            TypeError: If the mapping values are neither ratios nor
                filepath lists. The method warns and skips each element
                of a filepath list that is not a filepath.
            ValueError: If ``splits`` is provided but is empty.
            ValueError: If the ratios are outside the range from 0 to 1 or
                do not sum to 1.
            ValueError: If split ratios are used but all the data already
                belongs to a split while ``replace_old_splits`` is ``False``.
            FileNotFoundError: If the dataset is empty.

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
        n_keypoints: dict[str, int] = {}
        for task, task_keypoints in self.get_keypoint_metadata().items():
            if task_keypoints.labels:
                n_keypoints[task] = len(task_keypoints.labels)
            else:
                # A definition set from edges alone has no labels to count.
                last = max((max(e) for e in task_keypoints.edges), default=-1)
                n_keypoints[task] = last + 1
        return n_keypoints
