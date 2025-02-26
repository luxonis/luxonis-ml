from abc import ABC, abstractmethod
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from semver.version import Version
from typing_extensions import TypeAlias

from luxonis_ml.data.datasets.annotation import DatasetRecord
from luxonis_ml.data.datasets.source import LuxonisSource
from luxonis_ml.typing import PathType
from luxonis_ml.utils import AutoRegisterMeta, Registry

DATASETS_REGISTRY: Registry[Type["BaseDataset"]] = Registry(name="datasets")


DatasetIterator: TypeAlias = Iterator[Union[dict, DatasetRecord]]


class BaseDataset(
    ABC, metaclass=AutoRegisterMeta, registry=DATASETS_REGISTRY, register=False
):
    """Base abstract dataset class for managing datasets in the Luxonis
    MLOps ecosystem."""

    @property
    @abstractmethod
    def identifier(self) -> str:
        """The unique identifier for the dataset.

        @type: str
        """
        ...

    @property
    @abstractmethod
    def version(self) -> Version:
        """The version of the underlying LDF.

        @type: L{Version}
        """
        ...

    @abstractmethod
    def set_tasks(self, tasks: Dict[str, List[str]]) -> None:
        """Sets the tasks for the dataset.

        @type tasks: Dict[str, List[str]]
        @param tasks: A dictionary mapping task names to task types.
        """
        ...

    @abstractmethod
    def get_tasks(self) -> Dict[str, str]:
        """Returns a dictionary mapping task names to task types.

        @rtype: Dict[str, str]
        @return: A dictionary mapping task names to task types.
        """
        ...

    @abstractmethod
    def update_source(self, source: LuxonisSource) -> None:
        """Updates underlying source of the dataset with a new
        LuxonisSource.

        @type source: L{LuxonisSource}
        @param source: The new C{LuxonisSource} to replace the old one.
        """
        ...

    @abstractmethod
    def set_classes(
        self,
        classes: Union[List[str], Dict[str, int]],
        task: Optional[str] = None,
    ) -> None:
        """Sets the classes for the dataset. This can be across all CV
        tasks or certain tasks.

        @type classes: Union[List[str], Dict[str, int]]
        @param classes: Either a list of class names or a dictionary
            mapping class names to class IDs. If list is provided, the
            class IDs will be assigned I{alphabetically} starting from
            C{0}. If the class names contain the class C{"background"},
            it will be assigned the class ID C{0}.
        @type task: Optional[str]
        @param task: Optionally specify the task where these classes
            apply.
        """
        ...

    @abstractmethod
    def get_classes(self) -> Dict[str, List[str]]:
        """Get classes according to computer vision tasks.

        @rtype: Dict[str, List[str]]
        @return: A dictionary mapping tasks to the classes used in each
            task.
        """
        ...

    @abstractmethod
    def set_skeletons(
        self,
        labels: Optional[List[str]] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
        task: Optional[str] = None,
    ) -> None:
        """Sets the semantic structure of keypoint skeletons for the
        classes that use keypoints.

        Example::

            dataset.set_skeletons(
                labels=["right hand", "right shoulder", ...],
                edges=[[0, 1], [4, 5], ...]
            )

        @type labels: Optional[List[str]]
        @param labels: List of keypoint names.
        @type edges: Optional[List[Tuple[int, int]]]
        @param edges: List of edges between keypoints.
        @type task: Optional[str]
        @param task: Optionally specify the task where these skeletons apply.
            If not specified, the skeletons are set for all tasks that use keypoints.
        """
        ...

    @abstractmethod
    def get_skeletons(
        self,
    ) -> Dict[str, Tuple[List[str], List[Tuple[int, int]]]]:
        """Returns the dictionary defining the semantic skeleton for
        each class using keypoints.

        @rtype: Dict[str, Tuple[List[str], List[Tuple[int, int]]]]
        @return: For each task, a tuple containing a list of keypoint
            names and a list of edges between the keypoints.
        """
        ...

    @abstractmethod
    def add(
        self, generator: DatasetIterator, batch_size: int = 1_000_000
    ) -> None:
        """Write annotations to parquet files.

        @type generator: L{DatasetIterator}
        @param generator: A Python iterator that yields either instances
            of C{DatasetRecord} or a dictionary that can be converted to
            C{DatasetRecord}.
        @type batch_size: int
        @param batch_size: The number of annotations generated before
            processing. This can be set to a lower value to reduce
            memory usage.
        """
        ...

    @abstractmethod
    def make_splits(
        self,
        splits: Optional[
            Union[
                Dict[str, Sequence[PathType]],
                Dict[str, float],
                Tuple[float, float, float],
            ]
        ] = None,
        *,
        ratios: Optional[
            Union[Dict[str, float], Tuple[float, float, float]]
        ] = None,
        definitions: Optional[Dict[str, List[PathType]]] = None,
        replace_old_splits: bool = False,
    ) -> None:
        """Generates splits for the dataset.

        @type splits: Optional[Union[Dict[str, Sequence[PathType]], Dict[str, float], Tuple[float, float, float]]]
        @param splits: A dictionary of splits or a tuple of ratios for train, val, and test splits. Can be one of:
            - A dictionary of splits with keys as split names and values as lists of filepaths
            - A dictionary of splits with keys as split names and values as ratios
            - A 3-tuple of ratios for train, val, and test splits
        @type ratios: Optional[Union[Dict[str, float], Tuple[float, float, float]]]
        @param ratios: Deprecated! A dictionary of splits with keys as split names and values as ratios.
        @type definitions: Optional[Dict[str, List[PathType]]]
        @param definitions: Deprecated! A dictionary of splits with keys as split names and values as lists of filepaths.
        @type replace_old_splits: bool
        @param replace_old_splits: Whether to remove old splits and generate new ones. If set to False, only new files will be added to the splits. Default is False.
        """
        ...

    @abstractmethod
    def delete_dataset(self) -> None:
        """Deletes all local files belonging to the dataset."""
        ...

    @staticmethod
    @abstractmethod
    def exists(dataset_name: str) -> bool:
        """Checks whether a dataset exists.

        @warning: For offline mode only.
        @type dataset_name: str
        @param dataset_name: Name of the dataset
        @rtype: bool
        @return: Whether the dataset exists
        """
        ...

    def get_task_names(self) -> List[str]:
        """Get the task names for the dataset.

        Like `get_tasks`, but returns only the task names
        instead of the entire names.

        @rtype: List[str]
        @return: List of task names.
        """
        return list(self.get_tasks().keys())

    def get_n_keypoints(self) -> Dict[str, int]:
        skeletons = self.get_skeletons()
        return {task: len(skeletons[task][0]) for task in skeletons}
