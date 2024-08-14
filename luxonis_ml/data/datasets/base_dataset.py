from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from typing_extensions import TypeAlias

from luxonis_ml.utils import AutoRegisterMeta, Registry
from luxonis_ml.utils.filesystem import PathType

from .annotation import DatasetRecord
from .source import LuxonisSource

DATASETS_REGISTRY = Registry(name="datasets")


DatasetIterator: TypeAlias = Iterator[Union[dict, DatasetRecord]]


class BaseDataset(
    ABC, metaclass=AutoRegisterMeta, registry=DATASETS_REGISTRY, register=False
):
    """Base abstract dataset class for managing datasets in the Luxonis MLOps
    ecosystem."""

    @property
    @abstractmethod
    def identifier(self) -> str:
        """The unique identifier for the dataset."""
        pass

    @abstractmethod
    def get_tasks(self) -> List[str]:
        """Returns the list of tasks in the dataset.

        @rtype: List[str]
        @return: List of task names.
        """
        pass

    @abstractmethod
    def update_source(self, source: LuxonisSource) -> None:
        """Updates underlying source of the dataset with a new LuxonisSource.

        @type source: L{LuxonisSource}
        @param source: The new L{LuxonisSource} to replace the old one.
        """
        pass

    @abstractmethod
    def set_classes(self, classes: List[str], task: Optional[str] = None) -> None:
        """Sets the names of classes for the dataset. This can be across all CV tasks or
        certain tasks.

        @type classes: List[str]
        @param classes: List of class names to set.
        @type task: Optional[str]
        @param task: Optionally specify the task where these classes apply.
        """
        pass

    @abstractmethod
    def get_classes(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """Gets overall classes in the dataset and classes according to computer vision
        task.

        @type sync_mode: bool
        @param sync_mode: If C{True}, reads classes from remote storage. If C{False},
            classes are read locally.
        @rtype: Tuple[List[str], Dict]
        @return: A combined list of classes for all tasks and a dictionary mapping tasks
            to the classes used in each task.
        """
        pass

    @abstractmethod
    def set_skeletons(
        self,
        labels: Optional[List[str]] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
        task: Optional[str] = None,
    ) -> None:
        """Sets the semantic structure of keypoint skeletons for the classes that use
        keypoints.

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
        pass

    @abstractmethod
    def get_skeletons(self) -> Dict[str, Tuple[List[str], List[Tuple[int, int]]]]:
        """Returns the dictionary defining the semantic skeleton for each class using
        keypoints.

        @rtype: Dict[str, Tuple[List[str], List[Tuple[int, int]]]]
        @return: For each task, a tuple containing a list of keypoint names and a list
            of edges between the keypoints.
        """
        pass

    @abstractmethod
    def add(self, generator: DatasetIterator, batch_size: int = 1_000_000) -> None:
        """Write annotations to parquet files.

        @type generator: L{DatasetGenerator}
        @param generator: A Python iterator that yields either instances of
            L{DatasetRecord} or a dictionary that can be converted to L{DatasetRecord}.
        @type batch_size: int
        @param batch_size: The number of annotations generated before processing. This
            can be set to a lower value to reduce memory usage.
        """
        pass

    @abstractmethod
    def make_splits(
        self,
        ratios: Optional[Union[Dict[str, float], Tuple[float, float, float]]] = None,
        definitions: Optional[Mapping[str, Sequence[PathType]]] = None,
    ) -> None:
        """Saves a splits json file that specified the train/val/test splis.

        @type ratios: Optional[Union[Dict[str, float], Tuple[float, float, float]]]
        @param ratios: Dictionary specifying the ratios of the splits. Can be a tuple
            of floats for "train", "val", and "test" respectively, or a dictionary
            specifying the ratios for each split.
            Defaults to C{{"train": 0.8, "val": 0.1, "test": 0.1}}.

        @type definitions: Optional[Mapping[str, Sequence[PathType]]
        @param definitions: Dictionary specifying split keys to lists
            of filepath values. Note that this assumes unique filenames.
            Example::

                {
                    "train": ["/path/to/cat.jpg", "/path/to/dog.jpg"],
                    "val": [...],
                    "test": [...]
                }

            Only overrides splits that are present in the dictionary.
        """
        pass

    @abstractmethod
    def delete_dataset(self) -> None:
        """Deletes all local files belonging to the dataset."""
        pass

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
        pass
