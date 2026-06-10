"""Annotation schemas as `pydantic models`_ used by Luxonis Data Format
datasets.

.. _pydantic models: https://pydantic.dev/docs/validation/latest/concepts/models/
"""

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypeAlias

import cv2
import numpy as np
import pycocotools.mask
from loguru import logger
from PIL import Image, ImageDraw
from pydantic import (
    AliasChoices,
    Field,
    GetCoreSchemaHandler,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.types import FilePath, PositiveInt
from pydantic_core import core_schema
from typing_extensions import Self, deprecated, override

from luxonis_ml.data.utils.parquet import ParquetRecord
from luxonis_ml.typing import BaseModelExtraForbid, PathType, check_type
from luxonis_ml.utils.logging import log_once

KeypointVisibility: TypeAlias = Literal[0, 1, 2]
"""Keypoint visibility following the COCO convention.

The values indicate the visibility of a keypoint in an image:

    - :math:`0`: Not visible or not labeled.
    - :math:`1`: Occluded.
    - :math:`2`: Visible.
"""
NormalizedFloat: TypeAlias = Annotated[float, Field(ge=0, le=1)]
"""A float value normalized to the range [0, 1]."""


class Category(str):
    """Category label for metadata values.

    This class is used to distinguish categorical metadata values from
    free-form string values.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(cls)


class Detection(BaseModelExtraForbid):
    """Detection record containing annotations and metadata for one
    object.

    It describes a single detected object in an image and can contain various
    types of annotations and metadata as well as nested sub-detections for
    hierarchical annotations.

    When ``scale_to_boxes`` is enabled, keypoints and segmentation data are
    interpreted relative to the bounding box and rescaled to image-normalized
    coordinates.

    Example:
        >>> detection = Detection(
        ...     class_name="person",
        ...     instance_id=1,
        ...     metadata={
        ...         "category": Category("adult"),
        ...     },
        ...     boundingbox={
        ...         "x": 0.1,
        ...         "y": 0.2,
        ...         "w": 0.3,
        ...         "h": 0.4,
        ...     },
        ...     instance_segmentation={
        ...         "mask": np.array([[0, 1], [1, 0]]),
        ...     },
        ...     sub_detections={
        ...         "face": {
        ...             "class_name": "face",
        ...             "boundingbox": {
        ...                 "x": 0.2,
        ...                 "y": 0.3,
        ...                 "w": 0.1,
        ...                 "h": 0.1,
        ...             },
        ...             "keypoints": {
        ...                 "keypoints": [
        ...                     (0.25, 0.35, 2),  # left eye
        ...                     (0.3, 0.35, 2),  # right eye
        ...                 ],
        ...             },
        ...             "metadata": {
        ...                 "expression": Category("happy"),
        ...                 "eye_color": Category("blue"),
        ...             },
        ...         },
        ...     },
        ... )

    Attributes:
        class_name: optional class name for the detection. Input data may use
            the ``"class"`` alias.
        instance_id: Instance identifier. If not provided, the
            instance IDs will correspond to the order in which
            the detections were added to the dataset.
            Note that this might lead to incorrect pairing of instance
            annotations if individual detection types are added separately
            and in an inconsistent order across records:

            .. python::

                # Without specifying `instance_id`, the
                # bounding box and keypoint annotation will
                # not be correctly paired as they are added in separate
                # detections and in a different order.
                def generator():
                    yield {
                        "file": ...,
                        "annotation": {"boundingbox": bbox1},
                    }
                    yield {
                        "file": ...,
                        "annotation": {"keypoints": kpts2},
                    }
                    yield {
                        "file": ...,
                        "annotation": {"boundingbox": bbox2},
                    }
                    yield {
                        "file": ...,
                        "annotation": {"keypoints": kpts1},
                    }


            It is recommended to provide instance IDs if possible
            and to avoid generating annotations individually in separate
            detections:

            .. python::

                # This is the correct way
                def generator():
                    yield {
                        "file": ...,
                        "annotation": {
                            "instance_id": 1,
                            "boundingbox": bbox1
                            "keypoints": kpts1,
                        },
                    }
                    yield {
                        "file": ...,
                        "annotation": {
                            "instance_id": 2,
                            "boundingbox": bbox2
                            "keypoints": kpts2,
                        },
                    }

        metadata: Metadata values keyed by metadata name.
        boundingbox: Optional bounding box annotation.
        keypoints: Optional keypoint annotation.
        instance_segmentation: Optional instance segmentation annotation.
        segmentation: Optional semantic segmentation annotation.
        array: Optional array annotation.
        scale_to_boxes: Whether annotation coordinates should be rescaled from
            bounding-box-relative coordinates.
        sub_detections: Nested detections keyed by sub-detection name.

    """

    class_name: str | None = Field(
        None, validation_alias=AliasChoices("class", "class_name")
    )
    instance_id: int = -1

    metadata: dict[str, int | float | str | Category] = {}

    boundingbox: Optional["BBoxAnnotation"] = None
    keypoints: Optional["KeypointAnnotation"] = None
    instance_segmentation: Optional["InstanceSegmentationAnnotation"] = None
    segmentation: Optional["SegmentationAnnotation"] = None
    array: Optional["ArrayAnnotation"] = None

    scale_to_boxes: bool = False

    sub_detections: dict[str, "Detection"] = {}

    def get_task_types(self) -> set[str]:
        """Get all the task type associated with this detection.

        Example:
            >>> detection = Detection(
            ...     class_name="cat",
            ...     boundingbox=BBoxAnnotation(x=0.1, y=0.2, w=0.3, h=0.4),
            ...     metadata={"color": "black"},
            ... )
            >>> sorted(detection.get_task_types())
            ['boundingbox', 'classification', 'metadata/color']

        Returns:
            Annotation task types and metadata keys.

        """
        task_types = {
            task_type
            for task_type in [
                "boundingbox",
                "keypoints",
                "segmentation",
                "instance_segmentation",
                "array",
            ]
            if getattr(self, task_type) is not None
        }
        if self.class_name is not None:
            task_types.add("classification")
        for metadata_key in self.metadata:
            task_types.add(f"metadata/{metadata_key}")

        return task_types

    @model_validator(mode="after")
    def _validate_names(self) -> Self:
        for name in self.sub_detections:
            self._check_valid_identifier(name, label="Sub-detection name")
        for key in self.metadata:
            self._check_valid_identifier(key, label="Metadata key")
        return self

    @model_validator(mode="after")
    def _rescale_values(self) -> Self:
        if not self.scale_to_boxes:
            return self
        if self.boundingbox is None:
            raise ValueError(
                "`scaled_to_boxes` is set to True, "
                "but no bounding box is provided."
            )
        x, y, w, h = (
            self.boundingbox.x,
            self.boundingbox.y,
            self.boundingbox.w,
            self.boundingbox.h,
        )

        if self.keypoints is not None:
            self.keypoints = KeypointAnnotation(
                keypoints=[
                    (x + w * kp[0], y + h * kp[1], kp[2])
                    for kp in self.keypoints.keypoints
                ]
            )
        return self

    @staticmethod
    def _check_valid_identifier(name: str, *, label: str) -> None:
        name = name.replace("-", "_")
        if name and not name.isidentifier():
            raise ValueError(
                f"{label} can only contain alphanumeric characters, "
                "underscores, and dashes. Additionally, the first character "
                f"must be a letter or underscore. Got {name}"
            )


class Annotation(ABC, BaseModelExtraForbid):
    """Base class for an annotation."""

    @staticmethod
    @abstractmethod
    def combine_to_numpy(
        annotations: list["Annotation"], classes: list[int], n_classes: int
    ) -> np.ndarray:
        """Combine multiple annotations into a single numpy array.

        Args:
            annotations: Annotations to combine.
            classes: Class IDs corresponding to each annotation.
            n_classes: Total number of classes.

        Returns:
            Combined annotation representation.

        """
        ...


class ClassificationAnnotation(Annotation):
    """Dummy wrapper annotation for classification tasks.

    There is no explicit annotation field for classification tasks,
    instead the class name of a detection is interpreted as the class
    label and interpreted as belonging to the entire image.

    Multiple classification annotations are multi-hot encoded into a
    single vector with length equal to the total number of classes.
    """

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: list["ClassificationAnnotation"],
        classes: list[int],
        n_classes: int,
    ) -> np.ndarray:
        r"""Combine classification annotations into a multi-hot label
        vector.

        Args:
            annotations: Classification annotations to combine.
            classes: Class IDs associated with the annotations.
            n_classes: Total number of known classes.

        Returns:
            Multi-hot class label vector of shape :math:`\left(N\right,)`
            where :math:`N` is the total number of classes.

        """
        classify_vector = np.zeros(n_classes)
        for i in range(len(annotations)):
            classify_vector[classes[i]] = 1
        return classify_vector


class BBoxAnnotation(Annotation):
    """Bounding box annotation.

    Values are normalized based on the image size.

    Attributes:
        x: Normalized top-left x coordinate.
        y: Normalized top-left y coordinate.
        w: Normalized bounding box width.
        h: Normalized bounding box height.

    """

    x: NormalizedFloat
    y: NormalizedFloat
    w: NormalizedFloat
    h: NormalizedFloat

    def to_numpy(self, class_id: int) -> np.ndarray:
        r"""Convert the bounding box annotation to row format.

        Args:
            class_id: The numeric class ID of the annotation.

        Returns:
            An array of shape :math:`\left(5\right,)`
            in the format ``[class_id, x, y, w, h]``.

        """
        return np.array([class_id, self.x, self.y, self.w, self.h])

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: list["BBoxAnnotation"],
        classes: list[int],
        n_classes: int | None = None,
    ) -> np.ndarray:
        r"""Combine bounding box annotations into rows with class IDs.

        Args:
            annotations: Bounding box annotations to combine.
            classes: Class IDs associated with the annotations.
            n_classes: Unused class count kept for API compatibility.

        Returns:
            An array of shape :math:`\left(N, 5\right)`
            where :math:`N` is the number of bounding box annotations
            and each row is in the format ``[class_id, x, y, w, h]``.

        """
        boxes = np.empty((len(annotations), 5))
        for i, ann in enumerate(annotations):
            boxes[i] = ann.to_numpy(classes[i])
        return boxes

    @model_validator(mode="before")
    @classmethod
    def _validate_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        warn = False
        for key in ["x", "y", "w", "h"]:
            if values[key] < -2 or values[key] > 2:
                raise ValueError(
                    "BBox annotation has value outside of automatic clipping range ([-2, 2]). "
                    "Values should be normalized based on image size to range [0, 1]."
                )
            if not (0 <= values[key] <= 1):
                warn = True
                values[key] = max(0, min(1, values[key]))
        if warn:
            logger.warning(
                "BBox annotation has values outside of [0, 1] range. Clipping them to [0, 1]."
            )

        return cls._clip_sum(values)

    @staticmethod
    def _clip_sum(values: dict[str, Any]) -> dict[str, Any]:
        if values["x"] + values["w"] > 1:
            values["w"] = 1 - values["x"]
            logger.warning(
                "BBox annotation has x + width > 1. Clipping width so the sum is 1."
            )
        if values["y"] + values["h"] > 1:
            values["h"] = 1 - values["y"]
            logger.warning(
                "BBox annotation has y + height > 1. Clipping height so the sum is 1."
            )
        return values


class KeypointAnnotation(Annotation):
    r"""Keypoint annotation.

    The coordinates are normalized to :math:`\left[0, 1\right]`
    based on the image size.

    Attributes:
        keypoints: Keypoints in ``(x, y, visibility)`` format.
            Visibility follows the COCO convention:

                - :math:`0`: Not visible or not labeled.
                - :math:`1`: Occluded.
                - :math:`2`: Visible.

    """

    keypoints: list[
        tuple[NormalizedFloat, NormalizedFloat, KeypointVisibility]
    ]

    def to_numpy(self) -> np.ndarray:
        r"""Convert the keypoint annotation to flattened row format.

        Returns:
            An array of shape :math:`\left(3K\right,)` where :math:`K`
            is the number of keypoints. The format of the array is
            :math:`\left[x_1, y_1, v_1, x_2, y_2, v_2, \ldots \right]`
            where :math:`\left(x_i, y_i, v_i\right)` are the coordinates and visibility
            of the :math:`i`-th keypoint.

        """
        return np.array(self.keypoints).reshape((-1, 3)).flatten()

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: list["KeypointAnnotation"],
        classes: list[int] | None = None,  # pyright: ignore[reportUnusedParameter]
        n_classes: int | None = None,
    ) -> np.ndarray:
        r"""Combine keypoint annotations into flattened keypoint rows.

        Args:
            annotations: Keypoint annotations to combine.
            classes: Unused class IDs kept for API compatibility.
            n_classes: Unused class count kept for API compatibility.

        Returns:
            An array of shape :math:`\left(N, 3K\right)` where :math:`N`
            is the number of keypoint annotations and :math:`K` is the number
            of keypoints per annotation.
            Flattened keypoint rows. Each row contains keypoint coordinates
            and visibility in the format
            :math:`\left[x_1, y_1, v_1, x_2, y_2, v_2, \ldots \right]`
            where :math:`\left(x_i, y_i, v_i\right)` are the coordinates and visibility
            of the :math:`i`-th keypoint.

        """
        keypoints = np.empty(
            (len(annotations), len(annotations[0].keypoints) * 3)
        )
        for i, ann in enumerate(annotations):
            keypoints[i] = ann.to_numpy()
        return keypoints

    @model_validator(mode="before")
    @classmethod
    def _validate_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "keypoints" not in values:
            return values

        warn = False
        for i, keypoint in enumerate(values["keypoints"]):
            if (keypoint[0] < -2 or keypoint[0] > 2) or (
                keypoint[1] < -2 or keypoint[1] > 2
            ):
                raise ValueError(
                    "Keypoint annotation has value outside of automatic clipping range ([-2, 2]). "
                    "Values should be normalized based on image size to range [0, 1]."
                )
            new_keypoint = list(keypoint)
            if not (0 <= keypoint[0] <= 1):
                new_keypoint[0] = max(0, min(1, keypoint[0]))
                warn = True
            if not (0 <= keypoint[1] <= 1):
                new_keypoint[1] = max(0, min(1, keypoint[1]))
                warn = True
            values["keypoints"][i] = tuple(new_keypoint)

        if warn:
            logger.warning(
                "Keypoint annotation has values outside of [0, 1] range. Clipping them to [0, 1]."
            )
        return values


class SegmentationAnnotation(Annotation):
    """Run-length encoded segmentation mask.

    The encoded mask uses COCO-style `run-length encoding`_.

    This class support parsing segmentation masks from multiple input formats:

        - Run-length encoding (RLE) directly as a list of counts or as a byte string.
        - Binary mask arrays as numpy arrays or saved as ``.npy`` or ``.png`` files.
        - Polygons as lists of normalized points together with the image width and height.

    Example:
        >>> rle = SegmentationAnnotation(
        ...     height=4,
        ...     width=4,
        ...     counts=b'11213ON0'
        ... )
        >>> mask = SegmentationAnnotation(
        ...     mask=np.array(
        ...         [
        ...            [0, 1, 0, 0],
        ...            [1, 1, 0, 0],
        ...            [0, 0, 0, 0],
        ...            [0, 0, 1, 1],
        ...         ]
        ...     )
        ... )
        >>> np.array_equal(rle.to_numpy(), mask.to_numpy())
        True

    Note:
        When providing the RLE as a list of counts instead of encoded bytes
        make sure the counts follow FORTRAN (column-major)
        order as expected by the COCO RLE format.

    Attributes:
        height: The height of the segmentation mask.
        width: The width of the segmentation mask.
        counts: Run-length encoded mask data.

    .. _run-length encoding:
        https://en.wikipedia.org/wiki/Run-length_encoding

    """

    height: PositiveInt
    width: PositiveInt
    counts: bytes

    def to_numpy(self) -> np.ndarray:
        r"""Convert the segmentation annotation to a binary mask.

        Returns:
            Binary mask of shape :math:`\left(H, W\right)`.

        """
        with warnings.catch_warnings(record=True):
            return pycocotools.mask.decode(
                {"counts": self.counts, "size": [self.height, self.width]}
            ).astype(np.uint8)

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: list["SegmentationAnnotation"],
        classes: list[int],
        n_classes: int,
    ) -> np.ndarray:
        r"""Combine segmentation annotations into class masks.

        Args:
            annotations: Segmentation annotations to combine.
            classes: Class IDs associated with the annotations.
            n_classes: Total number of known classes.

        Returns:
            Combined semantic segmentation masks of shape
            :math:`\left(C, H, W\right)`.

        Note:
            In case of overlapping annotations,
            the **first** mask in the list takes precedence.

        """
        ref = annotations[0]
        width, height = ref.width, ref.height
        masks = np.stack([ann.to_numpy() for ann in annotations])

        segmentation = np.zeros((n_classes, height, width), dtype=np.uint8)

        assigned_pixels = np.zeros((height, width), dtype=bool)
        for i, class_id in enumerate(classes):
            mask = masks[i] & (assigned_pixels == 0)
            segmentation[class_id, ...] = np.maximum(
                segmentation[class_id, ...], mask
            )
            assigned_pixels |= mask.astype(bool)

        return segmentation

    @field_serializer("counts", when_used="json")
    def _serialize_counts(self, counts: bytes) -> str:
        return counts.decode("utf-8")

    @model_validator(mode="before")
    @classmethod
    def _validate_rle(cls, values: dict[str, Any]) -> dict[str, Any]:
        if {"counts", "width", "height"} - set(values.keys()):
            return values

        height = values["height"]
        width = values["width"]

        if not check_type(height, int) or not check_type(width, int):
            raise ValueError("Height and width must be integers")

        counts = values["counts"]
        if isinstance(counts, str):
            values["counts"] = counts.encode("utf-8")

        elif isinstance(counts, list):
            for c in counts:
                if not isinstance(c, int) or c < 0:
                    raise ValueError(
                        "RLE counts must be a list of positive integers"
                    )

            with warnings.catch_warnings(record=True):
                rle: Any = pycocotools.mask.frPyObjects(
                    {"counts": counts, "size": [height, width]},  # type: ignore
                    height,
                    width,
                )
            values["counts"] = rle["counts"]
            values["height"] = rle["size"][0]
            values["width"] = rle["size"][1]

        return values

    @staticmethod
    def _numpy_to_rle(mask: np.ndarray) -> dict[str, Any]:
        mask = np.asfortranarray(mask.astype(np.uint8))
        with warnings.catch_warnings(record=True):
            rle = pycocotools.mask.encode(mask)
        return {
            "height": rle["size"][0],
            "width": rle["size"][1],
            "counts": rle["counts"].decode("utf-8"),  # type: ignore
        }

    @model_validator(mode="before")
    @classmethod
    def _validate_mask(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "mask" not in values:
            return values
        values = deepcopy(values)

        mask = values.pop("mask")
        if isinstance(mask, PathType):
            mask_path = Path(mask)
            if mask_path.suffix == ".npy":
                try:
                    mask = np.load(mask_path)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load mask from array at '{mask_path}'"
                    ) from e
            elif mask_path.suffix == ".png":
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise ValueError(
                        f"Failed to load mask from image at '{mask_path}'"
                    )

                mask = mask.astype(bool).astype(np.uint8)
            else:
                raise ValueError(
                    f"Unsupported mask format: {mask_path.suffix}. "
                    "Supported formats are .npy and .png"
                )
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                "Mask must be either a numpy array, "
                "or a path to a saved numpy array"
            )

        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D binary array")

        return {**cls._numpy_to_rle(mask), **values}

    @model_validator(mode="before")
    @classmethod
    def _validate_polyline(cls, values: dict[str, Any]) -> dict[str, Any]:
        if {"points", "width", "height"} - set(values.keys()):
            return values

        values = deepcopy(values)

        width = values.pop("width")
        height = values.pop("height")
        if not check_type(height, int) or not check_type(width, int):
            raise ValueError("Height and width must be integers")

        points = values.pop("points")
        if not check_type(points, list[tuple[float, float]]):
            raise ValueError("Polyline must be a list of float 2D points")

        if len(points) < 3:
            raise ValueError("Polyline must contain at least 3 points")

        cls._clip_points(points)

        polyline = [(round(x * width), round(y * height)) for x, y in points]
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polyline, fill=1, outline=1)
        return {"mask": np.array(mask).astype(np.uint8), **values}

    @staticmethod
    def _clip_points(points: list[tuple[float, float]]) -> None:
        warn = False
        for i in range(len(points)):
            x, y = points[i]
            if (x < -2 or x > 2) or (y < -2 or y > 2):
                raise ValueError(
                    "Polyline annotation has value outside of automatic clipping range ([-2, 2]). "
                    "Values should be normalized based on image size to range [0, 1]."
                )
            new_x, new_y = x, y
            if not (0 <= x <= 1):
                new_x = max(0, min(1, x))
                warn = True
            if not (0 <= y <= 1):
                new_y = max(0, min(1, y))
                warn = True

            points[i] = (new_x, new_y)

        if warn:
            logger.warning(
                "Polyline annotation has values outside of [0, 1] range. Clipping them to [0, 1]."
            )


class InstanceSegmentationAnnotation(SegmentationAnnotation):
    r"""Instance segmentation annotation.

    Subclass of `SegmentationAnnotation` used to distinguish
    instance segmentation annotations from semantic segmentation annotations.

    The array representation of a single instance segmentation annotation
    is the same as that of a semantic segmentation annotation,
    but multiple instance segmentation annotations are combined into
    an array of shape :math:`\left(N, H, W\right)` where the leading
    dimension :math:`N` corresponds to the number of instance annotations
    instead of the number of classes as in semantic segmentation.
    """

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: list["InstanceSegmentationAnnotation"],
        classes: list[int] | None = None,
        n_classes: int | None = None,
    ) -> np.ndarray:
        r"""Combine instance segmentation annotations into instance
        masks.

        Args:
            annotations: Instance segmentation annotations to combine.
            classes: Unused class IDs kept for API compatibility.
            n_classes: Unused class count kept for API compatibility.

        Returns:
            Combined instance segmentation masks of shape
            :math:`\left(N, H, W\right)` where :math:`N`
            is the number of instances.

        Note:
            As opposed to semantic segmentation, overlapping annotations
            are allowed and are not resolved in any way. One pixel
            can belong to multiple instances and will be marked as
            :math:`1` in each instance mask it belongs to.

        """
        return np.stack([ann.to_numpy() for ann in annotations])


class ArrayAnnotation(Annotation):
    """Custom annotation backed by an array file.

    All instances of this annotation must have the same shape.

    Attributes:
        path: Path to the array saved as a ``.npy`` file.

    """

    path: FilePath

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: list["ArrayAnnotation"],
        classes: list[int],
        n_classes: int,
    ) -> np.ndarray:
        r"""Combine array annotations into instance-class-indexed arrays.

        Args:
            annotations: Array annotations to combine.
            classes: Class IDs associated with the annotations.
            n_classes: Total number of known classes.

        Returns:
            Combined arrays of shape :math:`\left(N, C, \ldots\right)`
            where :math:`C` is the number of classes and
            :math:`N` is the number of instances.

        """
        out_arr = np.zeros(
            (len(annotations), n_classes, *np.load(annotations[0].path).shape)
        )
        for i, ann in enumerate(annotations):
            out_arr[i, classes[i]] = np.load(ann.path)
        return out_arr

    @field_serializer("path", when_used="json")
    def _serialize_path(self, value: FilePath) -> str:
        return str(value)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, path: FilePath) -> FilePath:
        if path.suffix != ".npy":
            raise ValueError(
                f"Array annotation file must be a .npy file. Got {path}"
            )
        try:
            np.load(path)
        except Exception as e:
            raise ValueError(
                f"Failed to load array annotation from {path}."
            ) from e
        return path


class DatasetRecord(BaseModelExtraForbid):
    """Dataset record containing file paths and an optional annotation.

    Attributes:
        files: File paths keyed by source name.
        annotation: Optional detection associated with the dataset record.
        task_name: The name of the task to which the record belongs.

    """

    files: dict[str, FilePath]
    annotation: Detection | None = None
    task_name: str = ""

    @property
    def file(self) -> FilePath:
        """The file path of the dataset record.

        This property is provided for convenience when the dataset record has
        exactly one file.

        Raises:
            ValueError: If the dataset record has zero or multiple files.

        """
        if len(self.files) != 1:
            raise ValueError("DatasetRecord must have exactly one file")
        return next(iter(self.files.values()))

    @property
    @deprecated("Use `list(record.files.values())` instead.")
    def all_file_paths(self) -> list[FilePath]:
        """All file paths associated with the dataset record.

        .. deprecated:: 0.9.0
            Use ``list(record.files.values())`` instead.
        """
        return list(self.files.values())

    def to_parquet_rows(self) -> Iterable[ParquetRecord]:
        """Recursively convert the dataset record and all its
        annotations and sub-annotations to parquet rows.

        Yields:
            Annotation data rows.

        """
        yield from self._to_parquet_rows(self.annotation, self.task_name)

    def _to_parquet_rows(
        self, annotation: Detection | None, task_name: str
    ) -> Iterable[ParquetRecord]:
        file_items = sorted(self.files.items(), key=lambda x: str(x[1]))
        for i, (source, file_path) in enumerate(file_items):
            is_main = i == 0

            if annotation is None or not is_main:
                yield {
                    "file": str(file_path),
                    "source_name": source,
                    "task_name": task_name,
                    "class_name": None,
                    "instance_id": None,
                    "task_type": None,
                    "annotation": None,
                }
            else:
                for task_type in [
                    "boundingbox",
                    "keypoints",
                    "segmentation",
                    "instance_segmentation",
                    "array",
                ]:
                    label: Annotation | None = getattr(annotation, task_type)

                    if label is not None:
                        yield {
                            "file": str(file_path),
                            "source_name": source,
                            "task_name": task_name,
                            "class_name": annotation.class_name,
                            "instance_id": annotation.instance_id,
                            "task_type": task_type,
                            "annotation": label.model_dump_json(),
                        }
                for key, data in annotation.metadata.items():
                    yield {
                        "file": str(file_path),
                        "source_name": source,
                        "task_name": task_name,
                        "class_name": annotation.class_name,
                        "instance_id": annotation.instance_id,
                        "task_type": f"metadata/{key}",
                        "annotation": json.dumps(data),
                    }
                if annotation.class_name is not None:
                    yield {
                        "file": str(file_path),
                        "source_name": source,
                        "task_name": task_name,
                        "class_name": annotation.class_name,
                        "instance_id": annotation.instance_id,
                        "task_type": "classification",
                        "annotation": "{}",
                    }
                for name, detection in annotation.sub_detections.items():
                    yield from self._to_parquet_rows(
                        detection, f"{task_name}/{name}"
                    )

    @model_validator(mode="before")
    @classmethod
    def _validate_task_name(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "task" in values:
            log_once(
                logger.warning,
                "The 'task' field is deprecated. Use 'task_name' instead.",
            )
            values["task_name"] = values.pop("task")
        return values

    @model_validator(mode="before")
    @classmethod
    def _validate_files(cls, values: dict[str, Any]) -> dict[str, Any]:
        values = deepcopy(values)
        if "file" in values:
            values["files"] = {"image": values.pop("file")}
        if "files" in values:
            files_dict = values["files"]
            values["files"] = {
                k: Path(v).absolute() for k, v in files_dict.items()
            }
        return values


def load_annotation(
    task_type: Literal[
        "classification",
        "boundingbox",
        "keypoints",
        "segmentation",
        "instance_segmentation",
        "array",
    ],
    data: dict[str, Any],
) -> "Annotation":
    """Load an annotation from serialized data.

    Args:
        task_type: The type of the annotation task.
        data: Serialized annotation data.

    Returns:
        An instance of the appropriate `Annotation` subclass based on the task type.

    Raises:
        ValueError: If the task type is unknown.

    """
    classes = {
        "classification": ClassificationAnnotation,
        "boundingbox": BBoxAnnotation,
        "keypoints": KeypointAnnotation,
        "segmentation": SegmentationAnnotation,
        "instance_segmentation": InstanceSegmentationAnnotation,
        "array": ArrayAnnotation,
    }
    if task_type not in classes:
        raise ValueError(f"Unknown label type: {task_type}")
    return classes[task_type](**data)
