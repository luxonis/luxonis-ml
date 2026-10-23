from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HeadMetadata(BaseModel):
    """Base parser metadata.

    Extra fields are allowed so custom parsers can store their own
    metadata.

    Attributes:
        postprocessor_path: Optional path to the postprocessor.

    """

    model_config = ConfigDict(extra="allow")

    postprocessor_path: str | None = Field(
        None, description="Optional path to the postprocessor."
    )


class HeadObjectDetectionMetadata(HeadMetadata):
    """Metadata for an object detection head.

    Attributes:
        classes: Names of classes detected by the model.
        n_classes: Number of classes detected by the model.
        iou_threshold: Non-maximum suppression IoU threshold for
            filtering overlapping boxes.
        conf_threshold: Confidence threshold above which a detection is
            considered valid.
        max_det: Maximum detections per image.
        anchors: Optional predefined boxes of different sizes and aspect
            ratios. The innermost values describe box sizes, the middle
            values group anchors per output, and the outer values are
            ordered from smallest to largest output.

    """

    classes: list[str] = Field(
        description="Names of classes detected by the model."
    )
    n_classes: int = Field(
        description="Number of classes detected by the model."
    )
    iou_threshold: float = Field(
        description="Non-maximum suppression IoU threshold for filtering overlapping boxes."
    )
    conf_threshold: float = Field(
        description="Confidence threshold above which a detection is considered valid."
    )
    max_det: int = Field(description="Maximum detections per image.")
    anchors: list[list[list[float]]] | None = Field(
        None,
        description=(
            "Predefined boxes of different sizes and aspect ratios, "
            "ordered from smallest to largest output."
        ),
    )


class HeadObjectDetectionSSDMetadata(HeadObjectDetectionMetadata):
    """Metadata for an SSD object detection head.

    Attributes:
        boxes_outputs: Output containing predicted bounding box
            coordinates.
        scores_outputs: Output containing predicted bounding box
            confidence scores.

    """

    boxes_outputs: str = Field(
        description="Output containing predicted bounding box coordinates."
    )
    scores_outputs: str = Field(
        description=(
            "Output containing predicted bounding box confidence scores."
        )
    )


class HeadClassificationMetadata(HeadMetadata):
    """Metadata for a classification head.

    Attributes:
        classes: Names of classes recognized by the model.
        n_classes: Number of classes recognized by the model.
        is_softmax: Whether the output already contains softmax
            probabilities.

    """

    classes: list[str] = Field(
        description="Names of classes recognized by the model."
    )
    n_classes: int = Field(
        description="Number of classes recognized by the model."
    )
    is_softmax: bool = Field(
        description="Whether the output already contains softmax probabilities."
    )


class HeadSegmentationMetadata(HeadMetadata):
    """Metadata for a segmentation head.

    Attributes:
        classes: Names of classes segmented by the model.
        n_classes: Number of classes segmented by the model.
        is_softmax: Whether the output already contains softmax
            probabilities.
        background_class: Whether the pixels classified as index 0
            should be treated as background.

    """

    classes: list[str] = Field(
        description="Names of classes segmented by the model."
    )
    n_classes: int = Field(
        description="Number of classes segmented by the model."
    )
    is_softmax: bool = Field(
        description="Whether the output already contains softmax probabilities."
    )
    background_class: bool = Field(
        default=False,
        description="Whether the pixels classified as index 0 should be treated as background.",
    )


class HeadYOLOMetadata(HeadObjectDetectionMetadata):
    """Metadata for a YOLO head.

    Attributes:
        yolo_outputs: Output names for the YOLO grid outputs.
        mask_outputs: Optional output names for mask coefficients.
        protos_outputs: Optional output name for mask prototypes.
        keypoints_outputs: Optional output names for keypoints.
        angles_outputs: Optional output names for oriented bounding box
            angles.
        subtype: YOLO family decoding subtype, such as ``"yolov5"``.
        n_prototypes: Optional number of prototypes per box for YOLO
            instance segmentation.
        n_keypoints: Optional number of keypoints per box for YOLO
            keypoint detection.
        is_softmax: Optional flag indicating whether YOLO instance
            segmentation outputs already contain softmax probabilities.
        strides: Optional per-head strides.

    """

    yolo_outputs: list[str] = Field(
        description="Output names for the YOLO grid outputs."
    )

    # Instance segmentation
    mask_outputs: list[str] | None = Field(
        None, description="Output names for mask coefficients."
    )
    protos_outputs: str | None = Field(
        None, description="Output name for mask prototypes."
    )

    # Keypoint detection
    keypoints_outputs: list[str] | None = Field(
        None, description="Output names for keypoints."
    )

    # OBB detection
    angles_outputs: list[str] | None = Field(
        None, description="Output names for oriented bounding box angles."
    )

    subtype: str = Field(
        description="YOLO family decoding subtype, such as 'yolov5'."
    )
    n_prototypes: int | None = Field(
        None,
        description="Number of prototypes per box for YOLO instance segmentation.",
    )
    n_keypoints: int | None = Field(
        None,
        description="Number of keypoints per box for YOLO keypoint detection.",
    )
    is_softmax: bool | None = Field(
        None,
        description=(
            "Whether YOLO instance segmentation outputs already contain "
            "softmax probabilities."
        ),
    )
    strides: list[int] | None = Field(
        None,
        description="Optional per-head strides.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_task_specific_fields(
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that YOLO metadata fields match a supported task.

        Args:
            values: Raw YOLO metadata fields.

        Returns:
            The raw metadata fields when they describe a supported YOLO
            task and output set.

        Raises:
            ValueError: If task-specific fields cannot infer a supported
                task, a field is not supported for the inferred task, a
                required output field is missing, or an output field is not
                supported for the inferred task.

        """
        defined_params = {k for k, v in dict(values).items() if v is not None}
        tasks = _infer_yolo_tasks(defined_params - _YOLO_COMMON_FIELDS)
        _check_yolo_outputs(defined_params - _YOLO_NON_OUTPUT_FIELDS, tasks)
        return values


# Fields that do not decide the YOLO task.
_YOLO_COMMON_FIELDS = frozenset(
    {
        "postprocessor_path",
        "yolo_outputs",
        "mask_outputs",
        "protos_outputs",
        "keypoints_outputs",
        "angles_outputs",
        "subtype",
        "iou_threshold",
        "conf_threshold",
        "max_det",
        "classes",
        "n_classes",
        "anchors",
        "strides",
    }
)

# Fields that are not output names.
_YOLO_NON_OUTPUT_FIELDS = frozenset(
    {
        "postprocessor_path",
        "subtype",
        "iou_threshold",
        "conf_threshold",
        "max_det",
        "classes",
        "n_classes",
        "anchors",
        "n_prototypes",
        "n_keypoints",
        "is_softmax",
        "strides",
    }
)

_YOLO_REQUIRED_FIELDS = {
    "object_detection": [],
    "instance_segmentation": ["n_prototypes", "is_softmax"],
    "keypoint_detection": ["n_keypoints"],
}

# Fields that rule out plain object detection.
_YOLO_DETECTION_EXCLUDED_FIELDS = frozenset(
    {"n_prototypes", "n_keypoints", "is_softmax"}
)

_YOLO_OUTPUT_FIELDS = {
    "instance_segmentation": [
        "yolo_outputs",
        "mask_outputs",
        "protos_outputs",
    ],
    "keypoint_detection": ["yolo_outputs", "keypoints_outputs"],
    "object_detection": ["yolo_outputs"],
}


def _infer_yolo_tasks(defined_params: set[str]) -> list[str]:
    """Infer the YOLO tasks that the task-specific fields describe.

    Every defined field must belong to one of the inferred tasks.

    Examples:
        >>> _infer_yolo_tasks(set())
        ['object_detection']
        >>> _infer_yolo_tasks({"n_keypoints"})
        ['keypoint_detection']
        >>> _infer_yolo_tasks({"is_softmax"})
        Traceback (most recent call last):
        ...
        ValueError: Invalid combination of parameters. No specific task can be inferred.

    """
    tasks = [
        task
        for task in ("instance_segmentation", "keypoint_detection")
        if all(f in defined_params for f in _YOLO_REQUIRED_FIELDS[task])
    ]
    if defined_params.isdisjoint(_YOLO_DETECTION_EXCLUDED_FIELDS):
        tasks.append("object_detection")

    if len(tasks) == 0:
        raise ValueError(
            "Invalid combination of parameters. No specific task can be inferred."
        )

    for param in defined_params:
        if not any(param in _YOLO_REQUIRED_FIELDS[task] for task in tasks):
            raise ValueError(
                "Invalid combination of parameters. "
                f"Field {param} is not supported for the tasks {tasks}."
            )
    return tasks


def _check_yolo_outputs(defined_params: set[str], tasks: list[str]) -> None:
    """Check that the output fields are the ones of the tasks.

    Every task needs all of its output fields, and every defined output
    field must belong to a task. Object detection may also set
    ``angles_outputs``.

    Examples:
        >>> _check_yolo_outputs(
        ...     {"yolo_outputs", "angles_outputs"}, ["object_detection"]
        ... )
        >>> _check_yolo_outputs(
        ...     {"yolo_outputs", "mask_outputs"}, ["object_detection"]
        ... )
        Traceback (most recent call last):
        ...
        ValueError: Invalid combination of output parameters. ...

    """
    if not all(
        field in defined_params
        for task in tasks
        for field in _YOLO_OUTPUT_FIELDS[task]
    ):
        raise ValueError(f"Invalid output fields for tasks {tasks}")

    for param in defined_params:
        if param == "angles_outputs" and "object_detection" in tasks:
            continue
        if not any(param in _YOLO_OUTPUT_FIELDS[task] for task in tasks):
            raise ValueError(
                "Invalid combination of output parameters. "
                f"Field {param} is not supported for the tasks {tasks}."
            )
