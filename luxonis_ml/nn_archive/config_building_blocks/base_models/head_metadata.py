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


class HeadYOLOMetadata(HeadObjectDetectionMetadata, HeadSegmentationMetadata):
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

    @model_validator(mode="before")
    @classmethod
    def validate_task_specific_fields(
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that YOLO metadata fields match a supported task."""
        defined_params = {k for k, v in dict(values).items() if v is not None}

        common_fields = [
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
        ]
        defined_params = defined_params.difference(common_fields)

        required_fields = {
            "object_detection": [],
            "instance_segmentation": [
                "n_prototypes",
                "is_softmax",
            ],
            "keypoint_detection": ["n_keypoints"],
        }

        unsupported_fields = {
            "object_detection": [
                "n_prototypes",
                "n_keypoints",
                "is_softmax",
            ],
            "instance_segmentation": ["n_keypoints"],
            "keypoint_detection": [
                "n_prototypes",
                "is_softmax",
            ],
        }

        tasks = []
        # Extract the task type
        if all(
            field in defined_params
            for field in required_fields.get("instance_segmentation", [])
        ):
            tasks.append("instance_segmentation")
        if all(
            field in defined_params
            for field in required_fields.get("keypoint_detection", [])
        ):
            tasks.append("keypoint_detection")
        if all(
            field not in defined_params
            for field in unsupported_fields.get("object_detection", [])
        ):
            tasks.append("object_detection")

        if len(tasks) == 0:
            raise ValueError(
                "Invalid combination of parameters. No specific task can be inferred."
            )

        for param in defined_params:
            if not any(param in required_fields[task] for task in tasks):
                raise ValueError(
                    "Invalid combination of parameters. "
                    f"Field {param} is not supported for the tasks {tasks}."
                )

        # Validate Outputs
        defined_params = {k for k, v in dict(values).items() if v is not None}
        common_fields = [
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
        ]
        defined_params = defined_params.difference(common_fields)

        supported_output_params = {
            "instance_segmentation": [
                "yolo_outputs",
                "mask_outputs",
                "protos_outputs",
            ],
            "keypoint_detection": ["yolo_outputs", "keypoints_outputs"],
            "object_detection": ["yolo_outputs"],
        }

        # Check if all required output fields are present
        if not all(
            field in defined_params
            for task in tasks
            for field in supported_output_params[task]
        ):
            raise ValueError(f"Invalid output fields for tasks {tasks}")

        # Check if all defined fields are supported
        for param in defined_params:
            if param == "angles_outputs" and "object_detection" in tasks:
                continue
            if not any(
                param in supported_output_params[task] for task in tasks
            ):
                raise ValueError(
                    "Invalid combination of output parameters. "
                    f"Field {param} is not supported for the tasks {tasks}."
                )

        return values
