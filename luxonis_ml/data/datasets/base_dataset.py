from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

from typing_extensions import TypeAlias

from luxonis_ml.utils import AutoRegisterMeta, Registry

from .annotation import DatasetRecord
from .source import LuxonisSource

DATASETS_REGISTRY = Registry(name="datasets")


DatasetIterator: TypeAlias = Iterator[Union[dict, DatasetRecord]]
KeypointVisibility: TypeAlias = Literal[0, 1, 2]
ParquetDict: TypeAlias = Dict[str, Any]


# class Metadata(BaseModel):
#     """Metadata for a dataset."""
#
#     classes: Dict[str, List[str]]
#     n_keypoints: Dict[str, int]
#     # keypoint_names: Dict[str, List[str]]
#     # keypoint_skeletons: Dict[str, List[Tuple[int, int]]]
#
#     @property
#     def all_classes(self) -> List[str]:
#         return list(set(c for classes in self.classes.values() for c in classes))


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
    def add(self, generator: DatasetIterator) -> None:
        """Write annotations to parquet files.

        @type generator: L{DatasetGenerator}
        @param generator: A Python iterator that yields dictionaries of data
            with the key described by the C{ANNOTATIONS_SCHEMA} but also listed below:
                - file (C{str}) : path to file on local disk or object storage
                - class (C{str}): string specifying the class name or label name
                - type (C{str}) : the type of label or annotation
                - value (C{Union[str, list, int, float, bool]}): the actual annotation value.
                The function will check to ensure `value` matches this for each annotation type

                    - value (classification) [bool] : Marks whether the class is present or not
                        (e.g. True/False)
                    - value (box) [List[float]] : the normalized (0-1) x, y, w, and h of a bounding box
                        (e.g. [0.5, 0.4, 0.1, 0.2])
                    - value (polyline) [List[List[float]]] : an ordered list of [x, y] polyline points
                        (e.g. [[0.2, 0.3], [0.4, 0.5], ...])
                    - value (segmentation) [Tuple[int, int, List[int]]]: an RLE representation of (height, width, counts) based on the COCO convention
                    - value (keypoints) [List[List[float]]] : an ordered list of [x, y, visibility] keypoints for a keypoint skeleton instance
                        (e.g. [[0.2, 0.3, 2], [0.4, 0.5, 2], ...])
                    - value (array) [str]: path to a numpy .npy file

        @type batch_size: int
        @param batch_size: The number of annotations generated before processing.
            This can be set to a lower value to reduce memory usage.
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
