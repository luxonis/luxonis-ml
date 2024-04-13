import json
import os.path as osp
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

from pydantic import BaseModel, ConfigDict, model_validator
from typeguard import TypeCheckError, check_type
from typing_extensions import TypeAlias

from luxonis_ml.data.utils.data_utils import (
    ArrayType,
    BoxType,
    ClassificationType,
    KeypointsType,
    LabelType,
    PolylineType,
    SegmentationType,
    transform_segmentation_value,
)
from luxonis_ml.enums import AnnotationType
from luxonis_ml.utils.registry import AutoRegisterMeta, Registry

from .source import LuxonisSource

DATASETS_REGISTRY = Registry(name="datasets")


class BaseAnnotation(ABC, BaseModel):
    model_config: ConfigDict = ConfigDict(extra="forbid")

    file: str
    task_group: str = "default"

    @abstractmethod
    def to_parquet(self, instance_id: str) -> Dict[str, Union[str, datetime, None]]:
        pass


class EmptyAnnotation(BaseAnnotation):
    """Empty annotation class for creating empty annotations."""

    def to_parquet(self, instance_id: str) -> Dict[str, Union[str, datetime, None]]:
        return {
            "instance_id": instance_id,
            "file": osp.basename(self.file),
            "task_group": self.task_group,
            "created_at": datetime.utcnow(),
        }


class Annotation(BaseAnnotation):
    """Base class for annotations in a dataset."""

    class_: str
    type_: AnnotationType
    value: Union[
        ClassificationType,
        LabelType,
        BoxType,
        SegmentationType,
        PolylineType,
        KeypointsType,
        ArrayType,
    ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """Creates an annotation from a dictionary.

        @type data: Dict[str, Any]
        @param data: A dictionary of annotation data.
        @rtype: L{Annotation}
        @return: An annotation object.
        """
        return cls(**{k.rstrip("_"): v for k, v in data.items()})

    def to_parquet(self, instance_id: str) -> Dict[str, Union[str, datetime, None]]:
        """Converts an annotation to a dictionary for writing to a parquet file.

        @type instance_id: str
        @param instance_id: Unique identifier for the instance
        @rtype: Dict[str, str]
        @return: A dictionary of annotation data.
        """
        value = self.value

        if self.type_ == AnnotationType.SEGMENTATION:
            value = transform_segmentation_value(cast(SegmentationType, self.value))

        if isinstance(value, (list, tuple)):
            value = json.dumps(value)
        else:
            value = str(value)

        return {
            "instance_id": instance_id,
            "file": osp.basename(self.file),
            "value_type": type(self.value).__name__,
            "type": self.type_.value,
            "class": self.class_,
            "value": value,
            "task_group": self.task_group,
            "created_at": datetime.utcnow(),
        }

    @model_validator(mode="after")
    def _value(self):
        self.check_value_type(self.type_, self.value)
        return self

    @staticmethod
    def check_value_type(typ: AnnotationType, value: Any) -> None:
        def _check_value_type(expected_type: Any, value: Any) -> None:
            """Checks if a value is of a given type, and raises a TypeError if not."""
            try:
                check_type(value, expected_type)
            except TypeCheckError as e:
                raise TypeError(
                    f"Value {value} for '{typ.value}' is not of type {expected_type}"
                ) from e

        for expected_type, annotation_type in zip(
            [
                ClassificationType,
                LabelType,
                BoxType,
                SegmentationType,
                PolylineType,
                KeypointsType,
                ArrayType,
            ],
            [
                AnnotationType.CLASSIFICATION,
                AnnotationType.LABEL,
                AnnotationType.BOX,
                AnnotationType.SEGMENTATION,
                AnnotationType.POLYLINE,
                AnnotationType.KEYPOINTS,
                AnnotationType.ARRAY,
            ],
        ):
            if typ == annotation_type:
                _check_value_type(expected_type, value)
                break


DatasetGenerator: TypeAlias = Generator[Annotation, None, None]


class BaseDataset(
    ABC, metaclass=AutoRegisterMeta, registry=DATASETS_REGISTRY, register=False
):
    """Base abstract dataset class for managing datasets in the Luxonis MLOps
    ecosystem."""

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
        @param task: Optionally specify the LabelType where these classes apply.
        """
        pass

    @abstractmethod
    def set_skeletons(self, skeletons: Dict[str, Dict]) -> None:
        """Sets the semantic structure of keypoint skeletons for the classes that use
        keypoints.

        @type skeletons: Dict[str, Dict]
        @param skeletons: A dict mapping class name to keypoint "labels" and "edges"
            between keypoints.
            The length of the "labels" determines the official number of keypoints.
            The inclusion of "edges" is optional.

            Example::

                {
                    "person": {
                        "labels": ["right hand", "right shoulder", ...]
                        "edges" [[0, 1], [4, 5], ...]
                    }
                }
        """
        pass

    @abstractmethod
    def get_classes(self) -> Tuple[List[str], Dict]:
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
    def get_skeletons(self) -> Dict[str, Dict]:
        """Returns the dictionary defining the semantic skeleton for each class using
        keypoints.

        @rtype: Dict[str, Dict]
        @return: A dictionary mapping classes to their skeleton definitions.
        """
        pass

    @abstractmethod
    def add(self, generator: DatasetGenerator) -> None:
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
