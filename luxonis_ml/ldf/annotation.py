r"""Annotation schemas used by Luxonis Data Format datasets.

This module owns the record and annotation payload contracts accepted by
`LuxonisDataset.add` and produced by format-specific parsers. The schemas are
implemented as `pydantic models`_, so input dictionaries are validated and
normalized before they are written to LDF parquet shards.

.. _pydantic models: https://pydantic.dev/docs/validation/latest/concepts/models/

.. contents:: Table of Contents
   :depth: 2


Record Model
============

Dataset ingestion starts with `DatasetRecord`. A record points to media,
optionally assigns a task name, and optionally carries an annotation payload
validated by `Detection`.

Single-source records use ``"file"``:

.. code-block:: json

    {
      "file": "path/to/image.jpg",
      "task_name": "detection",

      "sample_metadata": {
        "record_id": 123,
        "camera": "left",
        "tags": ["night", "warehouse"]
      },

      "annotation": {
        "class": "car",
        "boundingbox": {
          "x": 0.1,
          "y": 0.2,
          "w": 0.3,
          "h": 0.4
        }
      }
    }

Multi-source records use ``"files"``:

.. code-block:: json

    {
      "files": {
        "rgb": "path/to/rgb.png",
        "depth": "path/to/depth.png"
      },
      "task_name": "detection",

      "sample_metadata": {
        "sequence": "loading_dock_07",
        "frame": 42
      },

      "annotation": {
        "class": "person",
        "boundingbox": {
          "x": 0.1,
          "y": 0.1,
          "w": 0.3,
          "h": 0.4
        }
      }
    }

**Record-level metadata** lives in ``sample_metadata``. It describes the whole
sample: capture IDs, UI tags, source-system identifiers, camera names, frame
numbers, timestamps, or other values that should travel with the sample.
`LuxonisLoader` exposes it through `LoaderOutput.metadata`.

**Annotation metadata** lives in `Detection.metadata`. It is converted into
label tasks such as ``"detection/metadata/weather"`` and is meant to be
consumed by training code as annotation labels.

**Frontend note:** ``sample_metadata`` is sample data, not an annotation
target.

Task names group annotations that should be consumed together. If no
``task_name`` is provided, the empty string ``""`` is used. Loader label keys
therefore follow ``"task_name/task_type"`` and default-task keys start with
``"/"``.


Coordinates and Instances
=========================

Spatial annotations use image-normalized coordinates. For an image with width
:math:`W` and height :math:`H`, an absolute point :math:`(x, y)` is stored as
:math:`\left(x / W, y / H\right)`.

When multiple annotation types describe the same physical object, use the same
``instance_id`` so bounding boxes, keypoints, and instance masks can be
associated even when yielded as separate records.

Warning:
    If ``instance_id`` is omitted and related annotation types are yielded in
    separate records, association falls back to insertion order. That is only
    reliable when every record is emitted consistently.

`Detection.scale_to_boxes` supports box-relative annotations. When enabled,
keypoints are interpreted relative to the bounding box and rescaled to
image-normalized coordinates before storage.


Classification
==============

Classification assigns a class to the whole sample or instance:

.. python::

    {"class": "vehicle"}

Classification is represented internally as the ``classification`` task type.
Any detection that provides a class name contributes a classification target,
even when the same detection also contains boxes, masks, keypoints, arrays, or
metadata.
When loaded, classes are usually returned as one-hot vectors with shape
:math:`\left(C\right)`.


Bounding Boxes
==============

`BBoxAnnotation` stores normalized ``xywh`` boxes, where ``x`` and ``y`` are
the top-left corner and ``w`` and ``h`` are width and height:

.. python::

    {
        "class": "car",
        "instance_id": 17,
        "boundingbox": {
            "x": 0.20,
            "y": 0.10,
            "w": 0.35,
            "h": 0.25,
        },
    }

Loader output combines boxes into :math:`\left(N, 5\right)` arrays with rows
:math:`\left[c, x, y, w, h\right]`, where :math:`c` is the class index.


Keypoints
=========

`KeypointAnnotation` stores keypoints as ``(x, y, visibility)`` triplets keyed
by name. Coordinates are normalized and visibility follows the COCO
convention:

    - :math:`0`: not visible or not labeled.
    - :math:`1`: occluded.
    - :math:`2`: visible.

.. python::

    {
        "class": "car",
        "instance_id": 17,
        "keypoints": {
            "keypoints": {
                "front_left_wheel": (0.10, 0.20, 2),
                "front_right_wheel": (0.30, 0.40, 1),
            },
        },
    }

An annotation can name only the keypoints it has. The other keypoints get
:math:`\left(0, 0, 0\right)`. A plain list of triplets is also accepted. The
keypoints are then keyed by position as ``"0"``, ``"1"``, ....

Each keypoint is a `Keypoint`. It is a named tuple, so ``keypoint[2]`` and
``keypoint.visibility`` give the same value. Visibility defaults to
:math:`2`.

An annotation can also carry three task-level fields: the edges between the
keypoints, the pairs that a horizontal flip swaps, and the OKS sigmas. Edges
and flip pairs can refer to keypoints by name:

.. python::

    {
        "class": "person",
        "keypoints": {
            "keypoints": {
                "nose": (0.50, 0.30, 2),
                "left_eye": (0.40, 0.20, 2),
                "right_eye": (0.60, 0.20, 1),
            },
            "edges": [("nose", "left_eye"), ("nose", "right_eye")],
            "flip_pairs": [("left_eye", "right_eye")],
            "sigmas": [0.026, 0.025, 0.025],
        },
    }

These three fields describe the task, not the instance.
`LuxonisDataset.add` thus moves them into a `KeypointMetadata` and keeps one
entry for each task. If you give no flip pairs, `LuxonisDataset.add` infers
them from ``left`` and ``right`` names. The dataset stores the keypoints of
a task in the order that the keypoint metadata defines.

For :math:`K` keypoints and :math:`N` instances, loader output uses shape
:math:`\left(N, 3 \cdot K\right)`.


Segmentation
============

`SegmentationAnnotation` supports polygon, binary-mask, and run-length encoded
inputs.

Polyline segmentation stores normalized polygon points. The final point is
implicitly connected to the first one:

.. python::

    {
        "class": "road",
        "segmentation": {
            "height": 720,
            "width": 1280,
            "points": [
                (0.10, 0.10),
                (0.90, 0.10),
                (0.80, 0.80),
                (0.20, 0.80),
            ],
        },
    }

Binary masks are two-dimensional arrays where foreground pixels are
:math:`1` and background pixels are :math:`0`:

.. python::

    {
        "class": "road",
        "segmentation": {
            "mask": binary_mask,
        },
    }

Run-length encoded masks use COCO RLE. The ``counts`` value may be an
uncompressed list of integers or a compressed byte string:

.. python::

    {
        "class": "road",
        "segmentation": {
            "height": 720,
            "width": 1280,
            "counts": [120, 8, 200, 12],
        },
    }

Note:
    Numpy masks are converted to RLE internally. RLE input is primarily for
    compatibility with datasets that already store masks in that format.

Semantic segmentation loader output uses channel-first masks with shape
:math:`\left(C, H, W\right)`.


Instance Segmentation
=====================

`InstanceSegmentationAnnotation` uses the same mask encodings as semantic
segmentation, but stores one mask per instance. A detection may include both a
bounding box and an instance mask:

.. python::

    {
        "class": "car",
        "instance_id": 17,
        "boundingbox": {"x": 0.20, "y": 0.10, "w": 0.35, "h": 0.25},
        "instance_segmentation": {
            "height": 720,
            "width": 1280,
            "points": [
                (0.20, 0.10),
                (0.55, 0.10),
                (0.55, 0.35),
                (0.20, 0.35),
            ],
        },
    }

Instance-mask loader output uses shape :math:`\left(N, H, W\right)`.


Arrays
======

`ArrayAnnotation` references arbitrary ``.npy`` data synchronized with a
sample:

.. python::

    {
        "class": "embedding",
        "array": {
            "path": "path/to/embedding.npy",
        },
    }

Arrays are useful for modality-specific targets or auxiliary data that should
be stored with the dataset but does not fit standard spatial schemas.


Metadata and Categories
=======================

Metadata stores flexible key-value values. Use `Category` to mark a string as
categorical metadata rather than free-form text:

.. python::

    from luxonis_ml.ldf import Category

    {
        "metadata": {
            "text": "ABC-123",
            "text_color": Category("white"),
            "track_id": 42,
        },
    }

Categorical metadata can be encoded as integers by `LuxonisLoader`, or kept as
strings when loader configuration requests it.

OCR annotations commonly store recognized text and categorical visual
properties:

.. python::

    {
        "metadata": {
            "text": "ABC-123",
            "color": Category("red"),
        },
    }

Embedding and re-identification datasets commonly store identifiers or other
lookup keys:

.. python::

    {
        "metadata": {
            "id": 42,
            "color": Category("red"),
        },
    }

Important:
    Metadata and arrays have no universal geometric semantics. Built-in
    augmentations can discard values associated with boxes that leave the
    image, but arbitrary values are otherwise preserved unless a custom
    augmentation explicitly handles them.

"""

import json
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    NamedTuple,
    Optional,
    TypeAlias,
    TypedDict,
)

import numpy as np
import pycocotools.mask
from loguru import logger
from PIL import Image, ImageDraw
from pydantic import (
    AliasChoices,
    Field,
    GetCoreSchemaHandler,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.types import FilePath, NonNegativeInt, PositiveFloat, PositiveInt
from pydantic_core import core_schema
from typing_extensions import Self, deprecated, override

from luxonis_ml.ldf.parquet import ParquetRecord
from luxonis_ml.typing import (
    BaseModelExtraForbid,
    Params,
    PathType,
    check_type,
)
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


class Keypoint(NamedTuple):
    r"""A single keypoint.

    It is a named tuple, not a model. It compares, unpacks and converts to
    NumPy as a plain :math:`\left(x, y, \text{visibility}\right)` triplet
    does.

    Example:
        >>> keypoint = Keypoint(0.1, 0.2)
        >>> keypoint.visibility
        2
        >>> keypoint == (0.1, 0.2, 2)
        True

    Attributes:
        x: Normalized x coordinate.
        y: Normalized y coordinate.
        visibility: Visibility following the COCO convention.

    """

    x: NormalizedFloat
    y: NormalizedFloat
    visibility: KeypointVisibility = 2


#: A keypoint the task defines but an annotation leaves out.
_UNLABELED_KEYPOINT: Final[Keypoint] = Keypoint(0.0, 0.0, 0)

#: Side markers recognized when inferring flip pairs from keypoint names.
_SIDE_MARKERS: Final = {
    "left": "left",
    "l": "left",
    "right": "right",
    "r": "right",
}


class KeypointMetadata(BaseModelExtraForbid):
    r"""Task-level description of a set of keypoints.

    It describes the keypoints of a whole task, not those of one instance.
    A `KeypointAnnotation` carries the same values as flat fields, and
    `LuxonisDataset.add` moves them here. A dataset keeps one entry for
    each task.

    Edges and flip pairs accept keypoint names. They resolve against
    `labels` and store indices, which index into a keypoint array. The
    fields declare indices, so pass the names through `model_validate`.
    The constructor resolves them too, but a type checker rejects a name
    there:

    Example:
        >>> KeypointMetadata.model_validate(
        ...     {
        ...         "labels": ["nose", "left_eye", "right_eye"],
        ...         "edges": [("nose", "left_eye"), ("nose", "right_eye")],
        ...         "flip_pairs": [("left_eye", "right_eye")],
        ...     }
        ... )
        KeypointMetadata(labels=['nose', 'left_eye', 'right_eye'], edges=[(0, 1), (0, 2)], flip_pairs=[(1, 2)], sigmas=[])

    Attributes:
        labels: Keypoint names in index order.
        edges: Keypoint graph edges as :math:`0`-based index pairs.
        flip_pairs: Index pairs swapped by a horizontal flip, used to keep
            symmetric keypoints such as left and right eyes consistent.
        sigmas: Per-keypoint OKS standard deviations.

    """

    labels: list[str] = []
    edges: list[tuple[int, int]] = []
    flip_pairs: list[tuple[NonNegativeInt, NonNegativeInt]] = []
    sigmas: list[PositiveFloat] = []

    @property
    def has_names(self) -> bool:
        """Whether the labels are chosen names.

        An annotation that carries a plain list of triplets is keyed
        ``"0"``, ``"1"``, ..., and `LuxonisDataset.add` stores those keys
        as the labels. They record only how many keypoints there are.
        """
        return bool(self.labels) and not _is_positional(self.labels)

    def merge_with(
        self, other: "KeypointMetadata", context: str = ""
    ) -> "KeypointMetadata":
        """Merge two keypoint declarations into one.

        A field that one declaration leaves empty comes from the other one.
        Two values for the same field must agree.

        Args:
            other: Keypoint metadata to merge into this one.
            context: Description of the merge, used in the error message.

        Returns:
            The merged keypoint metadata.

        Raises:
            ValueError: If the two declarations disagree on any field.

        """
        conflicts = []
        for field in KeypointMetadata.model_fields:
            mine, theirs = getattr(self, field), getattr(other, field)
            if not mine or not theirs or mine == theirs:
                continue
            # A name identifies a keypoint. Two records that give the same
            # names thus agree, even in a different order. The first
            # declaration sets the order.
            if field == "labels" and set(mine) == set(theirs):
                continue
            conflicts.append(field)

        if conflicts:
            differences = "\n".join(
                f"    {field}: {getattr(self, field)} != {getattr(other, field)}"
                for field in conflicts
            )
            hint = (
                "\nA record that annotates only some of the keypoints must "
                "still name the full set. It can give the missing ones a "
                "visibility of 0."
                if "labels" in conflicts
                else ""
            )
            raise ValueError(
                f"Conflicting keypoint metadata declared{_where(context)}. "
                f"The following fields disagree:\n{differences}\n"
                "All records of a task must describe the same keypoints. "
                "Declare them on a single record, or use "
                f"`LuxonisDataset.set_keypoint_metadata`.{hint}"
            )
        return KeypointMetadata(
            **{
                field: getattr(self, field) or getattr(other, field)
                for field in KeypointMetadata.model_fields
            }
        )

    def validate_for(self, n_keypoints: int, context: str = "") -> None:
        """Check the keypoint metadata against a number of keypoints.

        Args:
            n_keypoints: Number of annotated keypoints.
            context: Description of what is being checked, used in the error
                messages.

        Raises:
            ValueError: If the keypoint metadata does not describe
                ``n_keypoints`` keypoints.

        """
        for field in ("labels", "sigmas"):
            value = getattr(self, field)
            if value and len(value) != n_keypoints:
                raise ValueError(
                    f"The keypoint metadata{_where(context)} defines "
                    f"{len(value)} {field}, but the annotations contain "
                    f"{n_keypoints} keypoints."
                )
        for field in ("edges", "flip_pairs"):
            for pair in getattr(self, field):
                for index in pair:
                    if not 0 <= index < n_keypoints:
                        raise ValueError(
                            f"The keypoint metadata{_where(context)} refers "
                            f"to keypoint {index} in `{field}`, but only "
                            f"{n_keypoints} keypoints are annotated."
                        )

    def align(self, keypoints: Mapping[str, Keypoint]) -> dict[str, Keypoint]:
        r"""Order keypoints to match the task, padding missing ones.

        A keypoint that the task defines but the annotation omits gets
        :math:`\left(0, 0, 0\right)`. This is the COCO value for a keypoint
        that is not labeled. An annotation can thus name only the keypoints
        it has.

        Args:
            keypoints: Keypoints keyed by name.

        Returns:
            The keypoints in `labels` order.

        Raises:
            ValueError: If a keypoint is not part of the task.

        """
        if not self.labels:
            return dict(keypoints)
        # Positional keys carry only their order, so they count as already
        # in task order. A record without names can thus sit next to
        # records that have them.
        if len(keypoints) == len(self.labels) and _is_positional(keypoints):
            return dict(zip(self.labels, keypoints.values(), strict=True))
        unknown = sorted(set(keypoints) - set(self.labels))
        if unknown:
            raise ValueError(
                f"Keypoints {', '.join(unknown)} are not part of the task. "
                f"Known keypoints: {', '.join(self.labels)}."
            )
        return {
            label: keypoints.get(label, _UNLABELED_KEYPOINT)
            for label in self.labels
        }

    @staticmethod
    def infer_flip_pairs(labels: Iterable[str]) -> list[tuple[int, int]]:
        """Infer horizontal flip pairs from ``left``/``right`` names.

        A name must carry a ``left``/``right`` or ``l``/``r`` marker at the
        start or at the end, and a separator must delimit it. The rest of
        the two names must match exactly. A keypoint on the midline, such
        as ``nose``, stays unpaired. So does a keypoint with no partner.
        The match is narrow on purpose. A wrong flip pair mirrors the wrong
        keypoints and never fails.

        Args:
            labels: Keypoint names in index order.

        Returns:
            Flip pairs as :math:`0`-based index pairs.

        Example:
            >>> KeypointMetadata.infer_flip_pairs(
            ...     ["nose", "left_eye", "right_eye", "l_ear", "r_ear"]
            ... )
            [(1, 2), (3, 4)]

        """
        sides: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {"left": [], "right": []}
        )
        for index, label in enumerate(labels):
            if (marker := _split_side(label)) is not None:
                side, name = marker
                sides[name][side].append(index)

        flip_pairs = []
        for name, by_side in sides.items():
            left, right = by_side["left"], by_side["right"]
            if not left or not right:
                continue
            if len(left) > 1 or len(right) > 1:
                logger.warning(
                    f"Cannot infer a flip pair for keypoint '{name}': it "
                    f"matches {len(left)} left and {len(right)} right names."
                )
                continue
            flip_pairs.append((min(left[0], right[0]), max(left[0], right[0])))
        return sorted(flip_pairs)

    @model_validator(mode="before")
    @classmethod
    def _resolve_names(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        labels = values.get("labels")
        return _resolve_pairs(
            values,
            labels if check_type(labels, list[str]) else None,
            "Provide `labels`, or refer to the keypoints by index.",
        )

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        # A name is the key of the keypoint, on disk and in the loader.
        # A duplicate name thus drops a keypoint instead of failing.
        duplicates = sorted(
            label for label, count in Counter(self.labels).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                f"Duplicate keypoint names: {', '.join(duplicates)}."
            )

        self.edges = sorted(self.edges)

        seen: dict[int, tuple[int, int]] = {}
        flip_pairs = []
        for a, b in self.flip_pairs:
            if a == b:
                raise ValueError(
                    f"Flip pair ({a}, {b}) flips keypoint {a} onto itself."
                )
            for index in (a, b):
                if index in seen:
                    raise ValueError(
                        f"Keypoint {index} appears in both flip pairs "
                        f"{seen[index]} and {(a, b)}. "
                        "Flip pairs must be disjoint."
                    )
            seen[a] = seen[b] = (a, b)
            flip_pairs.append((min(a, b), max(a, b)))
        self.flip_pairs = sorted(flip_pairs)
        return self


class _SerializedRLE(TypedDict):
    """JSON-compatible RLE fields stored by segmentation annotations."""

    height: int
    width: int
    counts: str


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


#: The `Detection` fields holding a single `Annotation`, in parquet row order.
_LABEL_TASK_TYPES = (
    "boundingbox",
    "keypoints",
    "segmentation",
    "instance_segmentation",
    "array",
)


class Detection(BaseModelExtraForbid):
    """Detection record containing annotations and metadata for one
    object.

    It describes a single detected object in an image and can contain various
    types of annotations and metadata as well as nested sub-detections for
    hierarchical annotations.

    When ``scale_to_boxes`` is enabled, keypoints are interpreted relative to
    the bounding box and rescaled to image-normalized coordinates.

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
        scale_to_boxes: Whether keypoint coordinates should be rescaled from
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
            for task_type in _LABEL_TASK_TYPES
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
                "`scale_to_boxes` is set to True, "
                "but no bounding box is provided."
            )
        x, y, w, h = (
            self.boundingbox.x,
            self.boundingbox.y,
            self.boundingbox.w,
            self.boundingbox.h,
        )

        if self.keypoints is not None:
            # The constructor clips the coordinates that the rescale pushes
            # out of the image. It does not reject them. `Keypoint` does
            # not validate, so a value out of range is safe here.
            self.keypoints = KeypointAnnotation(
                keypoints={
                    label: Keypoint(
                        x + w * keypoint.x,
                        y + h * keypoint.y,
                        keypoint.visibility,
                    )
                    for label, keypoint in self.keypoints.keypoints.items()
                },
                edges=self.keypoints.edges,
                flip_pairs=self.keypoints.flip_pairs,
                sigmas=self.keypoints.sigmas,
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

    def to_parquet_json(
        self, keypoint_metadata: KeypointMetadata | None = None
    ) -> str:
        """Serialize the annotation into its stored parquet payload.

        Args:
            keypoint_metadata: Keypoint metadata of the task, when known.
                A keypoint payload is positional, so the keypoint metadata
                sets the order. Every other annotation ignores it.

        Returns:
            The serialized annotation.

        """
        return self.model_dump_json()

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
            Multi-hot class label vector of shape :math:`\left(N,\right)`
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
            An array of shape :math:`\left(5,\right)`
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
    def _validate_values(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values

        # Coerce up front so everything pydantic accepts -- numpy scalars,
        # `Decimal`s, numeric strings -- gets clipped. Anything that is not
        # a number is left for pydantic to report instead of failing below.
        try:
            coordinates = {
                key: float(values[key]) for key in ["x", "y", "w", "h"]
            }
        except (LookupError, TypeError, ValueError):
            return values

        values = {**values, **coordinates}
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

    Keypoints are keyed by name, so an annotation can say which keypoints
    it holds instead of relying on their position. The coordinates are
    normalized to :math:`\left[0, 1\right]` based on the image size.

    A plain list of triplets is also accepted. The keypoints are then keyed
    by position as ``"0"``, ``"1"``, ....

    `edges`, `flip_pairs` and `sigmas` describe the task, not the instance.
    `LuxonisDataset.add` moves them into a `KeypointMetadata`, so a stored
    payload holds only the coordinates. Edges and flip pairs accept
    keypoint names, which resolve against the names of `keypoints`.

    Example:
        >>> KeypointAnnotation(
        ...     keypoints={"nose": (0.5, 0.3, 2), "left_eye": (0.4, 0.2, 1)}
        ... ).to_numpy()
        array([0.5, 0.3, 2. , 0.4, 0.2, 1. ])

    Attributes:
        keypoints: Keypoints in ``(x, y, visibility)`` format, keyed by
            name. Visibility follows the COCO convention:

                - :math:`0`: Not visible or not labeled.
                - :math:`1`: Occluded.
                - :math:`2`: Visible.

        edges: Keypoint graph edges, as index pairs or as name pairs.
        flip_pairs: Pairs that a horizontal flip swaps, as index pairs or
            as name pairs.
        sigmas: Per-keypoint OKS standard deviations.

    """

    keypoints: dict[str, Keypoint]
    edges: list[tuple[int, int]] = []
    flip_pairs: list[tuple[NonNegativeInt, NonNegativeInt]] = []
    sigmas: list[PositiveFloat] = []

    def to_numpy(self) -> np.ndarray:
        r"""Convert the keypoint annotation to flattened row format.

        Returns:
            An array of shape :math:`\left(3K,\right)` where :math:`K`
            is the number of keypoints. The format of the array is
            :math:`\left[x_1, y_1, v_1, x_2, y_2, v_2, \ldots \right]`
            where :math:`\left(x_i, y_i, v_i\right)` are the coordinates and visibility
            of the :math:`i`-th keypoint.

        """
        return np.array(list(self.keypoints.values()), dtype=float).reshape(-1)

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

        Raises:
            ValueError: If the annotations do not all have the same number
                of keypoints.

        """
        n_keypoints = len(annotations[0].keypoints)
        keypoints = np.empty((len(annotations), n_keypoints * 3))
        for i, ann in enumerate(annotations):
            if len(ann.keypoints) != n_keypoints:
                raise ValueError(
                    "Cannot combine keypoint annotations with different "
                    f"numbers of keypoints ({n_keypoints} and "
                    f"{len(ann.keypoints)}). All annotations of a task must "
                    "describe the same keypoints."
                )
            keypoints[i] = ann.to_numpy()
        return keypoints

    def declared_metadata(self) -> KeypointMetadata | None:
        """Return the task-level metadata this annotation describes.

        Keypoint names count as a declaration only if they are more than
        the positional fallback. An annotation built from a plain list is
        keyed ``"0"``, ``"1"``, .... Those keys give only the number of
        keypoints, so they must not conflict with real names.

        Returns:
            The declared metadata, or ``None`` if the annotation declares
            nothing.

        """
        labels = list(self.keypoints)
        if _is_positional(labels):
            labels = []
        if not (labels or self.edges or self.flip_pairs or self.sigmas):
            return None
        return KeypointMetadata(
            labels=labels,
            edges=self.edges,
            flip_pairs=self.flip_pairs,
            sigmas=self.sigmas,
        )

    @override
    def to_parquet_json(
        self, keypoint_metadata: KeypointMetadata | None = None
    ) -> str:
        annotation = self
        if keypoint_metadata is not None:
            aligned = keypoint_metadata.align(self.keypoints)
            # Compared as items, not as mappings. Dict equality ignores
            # order, and the stored payload encodes only the order.
            if list(aligned.items()) != list(self.keypoints.items()):
                annotation = self.model_copy(update={"keypoints": aligned})
        return annotation.model_dump_json()

    @model_serializer(mode="plain", when_used="json")
    def _serialize(self) -> dict[str, Any]:
        # The payload is positional. The names, the edges, the flip pairs
        # and the sigmas describe the task, not the instance.
        # `LuxonisDataset.add` thus keeps them in the dataset metadata and
        # not on every row.
        return {
            "keypoints": [
                list(keypoint) for keypoint in self.keypoints.values()
            ]
        }

    @model_validator(mode="before")
    @classmethod
    def _validate_values(cls, values: Any, info: ValidationInfo) -> Any:
        if not isinstance(values, Mapping) or "keypoints" not in values:
            return values

        # A stored payload is positional and carries no names. The loader
        # supplies the names of the task when it reads one back.
        labels = (info.context or {}).get("keypoint_labels")

        # Coerced up front for the same reason as in `BBoxAnnotation`.
        try:
            keypoints = cls._as_mapping(values["keypoints"], labels)
        except (LookupError, TypeError, ValueError):
            return values

        warn = False
        for keypoint in keypoints.values():
            x, y = keypoint[0], keypoint[1]
            if (x < -2 or x > 2) or (y < -2 or y > 2):
                raise ValueError(
                    "Keypoint annotation has value outside of automatic clipping range ([-2, 2]). "
                    "Values should be normalized based on image size to range [0, 1]."
                )
            if not (0 <= x <= 1):
                keypoint[0] = max(0.0, min(1.0, x))
                warn = True
            if not (0 <= y <= 1):
                keypoint[1] = max(0.0, min(1.0, y))
                warn = True

        if warn:
            logger.warning(
                "Keypoint annotation has values outside of [0, 1] range. Clipping them to [0, 1]."
            )

        values = {
            **values,
            "keypoints": {
                label: tuple(keypoint) for label, keypoint in keypoints.items()
            },
        }
        names = list(keypoints)
        return _resolve_pairs(
            values,
            None if _is_positional(names) else names,
            "Pass the keypoints as a mapping keyed by name, or refer to "
            "them by index.",
        )

    @staticmethod
    def _as_mapping(
        keypoints: Any, labels: Sequence[str] | None
    ) -> dict[str, list[Any]]:
        """Normalize keypoints into a mapping of name to ``[x, y, v]``.

        The keypoints can be a mapping keyed by name. They can also be a
        sequence of ``(x, y)`` or ``(x, y, visibility)`` triplets. A
        sequence takes its keys from ``labels``, or from the position when
        no labels are known. The input stays unchanged.
        """
        if isinstance(keypoints, Mapping):
            items = list(keypoints.items())
        else:
            values = list(keypoints)
            items = (
                list(zip(labels, values, strict=True))
                if labels is not None and len(labels) == len(values)
                else [(str(i), value) for i, value in enumerate(values)]
            )
        return {
            str(label): KeypointAnnotation._as_triplet(value)
            for label, value in items
        }

    @staticmethod
    def _as_triplet(keypoint: Any) -> list[Any]:
        """Return the coordinates as a list, which the caller can clip.

        `Keypoint` supplies the default visibility when the keypoint
        omits it. A keypoint with too many values passes through to
        pydantic, which names the value that does not belong.
        """
        if isinstance(keypoint, Mapping):
            keypoint = Keypoint(**keypoint)
        x, y, *rest = keypoint
        return [float(x), float(y), *rest]

    @model_validator(mode="after")
    def _validate_declared_metadata(self) -> Self:
        declared = self.declared_metadata()
        if declared is None:
            return self
        declared.validate_for(len(self.keypoints))
        # `KeypointMetadata` sorts the edges and orders each flip pair, so
        # take them back to keep the two in step.
        self.edges, self.flip_pairs = declared.edges, declared.flip_pairs
        return self


class SegmentationAnnotation(Annotation):
    """Run-length encoded segmentation mask.

    The encoded mask uses COCO-style `run-length encoding`_.

    This class support parsing segmentation masks from multiple input formats:

        - Run-length encoding (RLE) directly as a list of counts or as a byte string.
        - Binary mask arrays as numpy arrays or saved as ``.npy`` or ``.png`` files.
        - Polygons as lists of normalized points together with the image width and height.

    Example:
        >>> rle = SegmentationAnnotation(height=4, width=4, counts=b"11213ON0")
        >>> mask = SegmentationAnnotation(
        ...     mask=np.array(
        ...         [
        ...             [0, 1, 0, 0],
        ...             [1, 1, 0, 0],
        ...             [0, 0, 0, 0],
        ...             [0, 0, 1, 1],
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
    def _validate_rle(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        if {"counts", "width", "height"} - set(values.keys()):
            return values

        height = values["height"]
        width = values["width"]

        if not check_type(height, int) or not check_type(width, int):
            raise ValueError("Height and width must be integers")

        values = dict(values)
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
                rle = pycocotools.mask.frPyObjects(
                    {"counts": counts, "size": [height, width]},  # type: ignore
                    height,
                    width,
                )
            values["counts"] = rle["counts"]
            values["height"] = rle["size"][0]
            values["width"] = rle["size"][1]

        return values

    @staticmethod
    def _numpy_to_rle(mask: np.ndarray) -> _SerializedRLE:
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
    def _validate_mask(cls, values: Any) -> Any:
        if not isinstance(values, Mapping) or "mask" not in values:
            return values
        values = dict(values)

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
                try:
                    with Image.open(mask_path) as image:
                        mask = np.array(image.convert("L"))
                except Exception as e:
                    raise ValueError(
                        f"Failed to load mask from image at '{mask_path}'"
                    ) from e
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

        # The encoded size wins over any `height` and `width` sent with the
        # mask, otherwise the stored size would not match the counts.
        return {**values, **cls._numpy_to_rle(mask)}

    @model_validator(mode="before")
    @classmethod
    def _validate_polyline(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        if {"points", "width", "height"} - set(values.keys()):
            return values

        values = dict(values)

        width = values.pop("width")
        height = values.pop("height")
        if not check_type(height, int) or not check_type(width, int):
            raise ValueError("Height and width must be integers")

        points = values.pop("points")
        if not check_type(points, list[tuple[float, float]]):
            raise ValueError("Polyline must be a list of float 2D points")

        if len(points) < 3:
            raise ValueError("Polyline must contain at least 3 points")

        # `_clip_points` rewrites the list in place, so never hand it the
        # caller's own one.
        points = list(points)
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

    def to_numpy(self) -> np.ndarray:
        """Load the array from the file path."""
        return np.load(self.path)

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
            # Memory mapping keeps the check from reading the whole array.
            np.load(path, mmap_mode="r")
        except Exception:
            # Not every filesystem supports mmap, so a plain read decides
            # whether the file is really unusable.
            try:
                np.load(path)
            except Exception as e:
                raise ValueError(
                    f"Failed to load array annotation from {path}."
                ) from e
        return path


class DatasetRecord(BaseModelExtraForbid):
    """Dataset record containing file paths and an optional annotation.

    A record is the unit of ingestion for `LuxonisDataset.add`. It may point
    to one media source through ``file`` or to multiple synchronized sources
    through ``files``, but never both -- passing both is an error, where
    ``files`` used to be silently discarded in favor of ``file``.

    ``sample_metadata`` stores **record-level metadata**. It is preserved by
    native import/export and returned by `LuxonisLoader` as
    `LoaderOutput.metadata`. It is intentionally separate from
    `Detection.metadata`, which creates annotation metadata label tasks.

    Attributes:
        files: File paths keyed by source name.
        annotation: Optional detection associated with the dataset record.
        task_name: The name of the task to which the record belongs.
        sample_metadata: JSON-like metadata for the whole sample. Values
            should be JSON-serializable. Missing metadata defaults to an empty
            dictionary.

    Example:
        .. code-block:: json

            {
              "file": "images/frame_001.jpg",
              "task_name": "detection",

              "sample_metadata": {
                "record_id": 123,
                "camera": "left",
                "tags": ["night", "warehouse"]
              },

              "annotation": {
                "class": "person",
                "boundingbox": {
                  "x": 0.1,
                  "y": 0.2,
                  "w": 0.3,
                  "h": 0.4
                }
              }
            }

    """

    files: dict[str, FilePath]
    annotation: Detection | None = None
    task_name: str = ""
    sample_metadata: Params = Field(default_factory=dict)

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

    @model_validator(mode="after")
    def validate_task_name_valid_identifier(self) -> Self:
        Detection._check_valid_identifier(self.task_name, label="Task name")
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_task_name(cls, values: Any) -> Any:
        if not isinstance(values, Mapping) or "task" not in values:
            return values

        values = dict(values)
        task = values.pop("task")
        if values.get("task_name", task) != task:
            raise ValueError(
                "Conflicting values for 'task' and 'task_name'. "
                "Use only 'task_name'."
            )

        log_once(
            logger.warning,
            "The 'task' field is deprecated. Use 'task_name' instead.",
        )
        values["task_name"] = task
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_files(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values

        # A shallow copy is enough: nothing below mutates a nested value,
        # and deep-copying would duplicate any mask the payload carries.
        values = dict(values)
        if "file" in values:
            if "files" in values:
                raise ValueError("Provide either 'file' or 'files', not both.")
            values["files"] = {"image": values.pop("file")}
        if "files" in values:
            files = values["files"]
            # Anything else is left untouched for pydantic to report
            # against `files` instead of failing on the paths below.
            if isinstance(files, Mapping) and all(
                isinstance(path, PathType) for path in files.values()
            ):
                values["files"] = {
                    key: Path(path).absolute() for key, path in files.items()
                }
        return values

    def to_parquet_rows(
        self, keypoint_metadata: Mapping[str, KeypointMetadata] | None = None
    ) -> Iterable[ParquetRecord]:
        """Recursively convert the dataset record and all its
        annotations and sub-annotations to parquet rows.

        Args:
            keypoint_metadata: Keypoint metadata of the written tasks,
                keyed by task name. A keypoint payload is positional. The
                keypoint metadata thus sets the order and pads the omitted
                keypoints.

        Yields:
            Annotation data rows.

        """
        yield from self._to_parquet_rows(
            self.annotation,
            self.task_name,
            json.dumps(self.sample_metadata),
            keypoint_metadata or {},
        )

    def _to_parquet_rows(
        self,
        annotation: Detection | None,
        task_name: str,
        sample_metadata: str,
        keypoint_metadata: Mapping[str, KeypointMetadata],
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
                    "sample_metadata": sample_metadata,
                }
            else:
                for task_type in _LABEL_TASK_TYPES:
                    label: Annotation | None = getattr(annotation, task_type)

                    if label is not None:
                        yield {
                            "file": str(file_path),
                            "source_name": source,
                            "task_name": task_name,
                            "class_name": annotation.class_name,
                            "instance_id": annotation.instance_id,
                            "task_type": task_type,
                            "annotation": label.to_parquet_json(
                                keypoint_metadata.get(task_name)
                            ),
                            "sample_metadata": sample_metadata,
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
                        "sample_metadata": sample_metadata,
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
                        "sample_metadata": sample_metadata,
                    }
                for name, detection in annotation.sub_detections.items():
                    yield from self._to_parquet_rows(
                        detection,
                        f"{task_name}/{name}",
                        sample_metadata,
                        keypoint_metadata,
                    )

    @staticmethod
    def decode_metadata(value: Any) -> Params:
        """Decode serialized record metadata into a dictionary.

        Args:
            value: A metadata dictionary, serialized JSON object, empty string,
                or ``None``.

        Returns:
            Decoded metadata when the value is a dictionary-like JSON object;
            otherwise an empty dictionary.

        """
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            value = json.loads(value)
        return value if isinstance(value, dict) else {}


def load_annotation(
    task_type: Literal[
        "classification",
        "boundingbox",
        "keypoints",
        "segmentation",
        "instance_segmentation",
        "array",
    ],
    data: Mapping[str, Any],
    *,
    keypoint_labels: Sequence[str] | None = None,
) -> "Annotation":
    """Load an annotation from serialized data.

    Args:
        task_type: The type of the annotation task.
        data: Serialized annotation data.
        keypoint_labels: Names of the keypoints of the task, when known.
            A stored keypoint payload is positional. The names key the
            keypoints by name instead of by position.

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
    return classes[task_type].model_validate(
        data, context={"keypoint_labels": keypoint_labels}
    )


def _where(context: str) -> str:
    return f" for {context}" if context else ""


def _is_positional(labels: Iterable[str]) -> bool:
    """Whether keypoint names are the ``"0"``, ``"1"``, ... fallback.

    A plain list of triplets gives its keypoints keys by position. Those
    keys give only the number of keypoints. They are never labels that a
    user chose.
    """
    labels = list(labels)
    return labels == [str(i) for i in range(len(labels))]


def _resolve_pairs(
    values: Mapping[str, Any], labels: Sequence[str] | None, hint: str
) -> Mapping[str, Any]:
    """Replace every keypoint name in ``edges`` and ``flip_pairs``.

    Args:
        values: Raw input values.
        labels: Keypoint names in index order, when they are known.
        hint: Sentence that tells the caller how to supply the names.

    Returns:
        The values, with each name replaced by its index.

    Raises:
        ValueError: If a name occurs but no names are known, or if a name
            is not one of them.

    """
    try:
        pairs_by_field = {
            field: [list(pair) for pair in values[field]]
            for field in ("edges", "flip_pairs")
            if isinstance(values.get(field), Iterable)
            and not isinstance(values[field], (str, bytes))
        }
    except TypeError:
        # Malformed input. Let pydantic report it against the field type.
        return values

    if not any(
        isinstance(endpoint, str)
        for pairs in pairs_by_field.values()
        for pair in pairs
        for endpoint in pair
    ):
        return values

    if not labels:
        raise ValueError(
            "Keypoint names are required in order to refer to the edges or "
            f"the flip pairs by name. {hint}"
        )

    indices = {label: i for i, label in enumerate(labels)}
    resolved = dict(values)
    for field, pairs in pairs_by_field.items():
        resolved[field] = [
            [_resolve_endpoint(endpoint, indices) for endpoint in pair]
            for pair in pairs
        ]
    return resolved


def _resolve_endpoint(endpoint: Any, indices: Mapping[str, int]) -> Any:
    if not isinstance(endpoint, str):
        return endpoint
    if endpoint not in indices:
        raise ValueError(
            f"Unknown keypoint name '{endpoint}'. "
            f"Known keypoints: {', '.join(indices)}."
        )
    return indices[endpoint]


def _split_side(label: str) -> tuple[str, str] | None:
    """Split a keypoint name into its side marker and the remainder.

    A separator must delimit the marker. Thus ``bright_spot`` is not a
    right-side keypoint.

    Returns:
        The side and the remaining name, or ``None`` if the name carries no
        side marker.

    """
    parts = re.split(r"[_\-\s]+", label.strip().lower())
    if len(parts) < 2:
        return None
    if (side := _SIDE_MARKERS.get(parts[0])) is not None:
        return side, "_".join(parts[1:])
    if (side := _SIDE_MARKERS.get(parts[-1])) is not None:
        return side, "_".join(parts[:-1])
    return None


# Also keeps the API docs rooted here: pydoctor moves a re-exported name to
# the re-exporting module unless the module defining it lists it in `__all__`.
__all__ = [
    "Annotation",
    "ArrayAnnotation",
    "BBoxAnnotation",
    "Category",
    "ClassificationAnnotation",
    "DatasetRecord",
    "Detection",
    "InstanceSegmentationAnnotation",
    "Keypoint",
    "KeypointAnnotation",
    "KeypointMetadata",
    "KeypointVisibility",
    "NormalizedFloat",
    "SegmentationAnnotation",
    "load_annotation",
]
