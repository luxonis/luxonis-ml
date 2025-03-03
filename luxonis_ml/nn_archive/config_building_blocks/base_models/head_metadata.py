from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HeadMetadata(BaseModel):
    """Metadata for the basic head. It allows you to specify additional
    fields.

    @type postprocessor_path: str | None
    @ivar postprocessor_path: Path to the postprocessor.
    """

    model_config = ConfigDict(extra="allow")

    postprocessor_path: Optional[str] = Field(
        None, description="Path to the postprocessor."
    )


class HeadObjectDetectionMetadata(HeadMetadata):
    """Metadata for the object detection head.

    @type classes: list
    @ivar classes: Names of object classes detected by the model.
    @type n_classes: int
    @ivar n_classes: Number of object classes detected by the model.
    @type iou_threshold: float
    @ivar iou_threshold: Non-max supression threshold limiting boxes
        intersection.
    @type conf_threshold: float
    @ivar conf_threshold: Confidence score threshold above which a
        detected object is considered valid.
    @type max_det: int
    @ivar max_det: Maximum detections per image.
    @type anchors: list
    @ivar anchors: Predefined bounding boxes of different sizes and
        aspect ratios. The innermost lists are length 2 tuples of box
        sizes. The middle lists are anchors for each output. The outmost
        lists go from smallest to largest output.
    """

    classes: List[str] = Field(
        description="Names of object classes recognized by the model."
    )
    n_classes: int = Field(
        description="Number of object classes recognized by the model."
    )
    iou_threshold: float = Field(
        description="Non-max supression threshold limiting boxes intersection."
    )
    conf_threshold: float = Field(
        description="Confidence score threshold above which a detected object is considered valid."
    )
    max_det: int = Field(description="Maximum detections per image.")
    anchors: Optional[List[List[List[float]]]] = Field(
        None,
        description="Predefined bounding boxes of different sizes and aspect ratios. The innermost lists are length 2 tuples of box sizes. The middle lists are anchors for each output. The outmost lists go from smallest to largest output.",
    )


class HeadObjectDetectionSSDMetadata(HeadObjectDetectionMetadata):
    """Metadata for the SSD object detection head.

    @type boxes_outputs: str
    @ivar boxes_outputs: Output name corresponding to predicted bounding
        box coordinates.
    @type scores_outputs: str
    @ivar scores_outputs: Output name corresponding to predicted
        bounding box confidence scores.
    """

    boxes_outputs: str = Field(
        description="Output name corresponding to predicted bounding box coordinates."
    )
    scores_outputs: str = Field(
        description="Output name corresponding to predicted bounding box confidence scores."
    )


class HeadClassificationMetadata(HeadMetadata):
    """Metadata for the classification head.

    @type classes: list
    @ivar classes: Names of object classes classified by the model.
    @type n_classes: int
    @ivar n_classes: Number of object classes classified by the model.
    @type is_softmax: bool
    @ivar is_softmax: True, if output is already softmaxed
    """

    classes: List[str] = Field(
        description="Names of object classes recognized by the model."
    )
    n_classes: int = Field(
        description="Number of object classes recognized by the model."
    )
    is_softmax: bool = Field(
        description="True, if output is already softmaxed."
    )


class HeadSegmentationMetadata(HeadMetadata):
    """Metadata for the segmentation head.

    @type classes: list
    @ivar classes: Names of object classes segmented by the model.
    @type n_classes: int
    @ivar n_classes: Number of object classes segmented by the model.
    @type is_softmax: bool
    @ivar is_softmax: True, if output is already softmaxed
    """

    classes: List[str] = Field(
        description="Names of object classes recognized by the model."
    )
    n_classes: int = Field(
        description="Number of object classes recognized by the model."
    )
    is_softmax: bool = Field(
        description="True, if output is already softmaxed."
    )


class HeadYOLOMetadata(HeadObjectDetectionMetadata, HeadSegmentationMetadata):
    """Metadata for the YOLO head.

    @type yolo_outputs: list
    @ivar yolo_outputs: A list of output names for each of the different
        YOLO grid sizes.
    @type mask_outputs: list | None
    @ivar mask_outputs: A list of output names for each mask output.
    @type protos_outputs: str | None
    @ivar protos_outputs: Output name for the protos.
    @type keypoints_outputs: list | None
    @ivar keypoints_outputs: A list of output names for the keypoints.
    @type angles_outputs: list | None
    @ivar angles_outputs: A list of output names for the angles.
    @type subtype: str
    @ivar subtype: YOLO family decoding subtype (e.g. yolov5, yolov6,
        yolov7 etc.)
    @type n_prototypes: int | None
    @ivar n_prototypes: Number of prototypes per bbox in YOLO instance
        segmnetation.
    @type n_keypoints: int | None
    @ivar n_keypoints: Number of keypoints per bbox in YOLO keypoint
        detection.
    @type is_softmax: bool | None
    @ivar is_softmax: True, if output is already softmaxed in YOLO
        instance segmentation
    """

    yolo_outputs: List[str] = Field(
        description="A list of output names for each of the different YOLO grid sizes."
    )

    # Instance segmentation
    mask_outputs: Optional[List[str]] = Field(
        None, description="A list of output names for each mask output."
    )
    protos_outputs: Optional[str] = Field(
        None, description="Output name for the protos."
    )

    # Keypoint detection
    keypoints_outputs: Optional[List[str]] = Field(
        None, description="A list of output names for the keypoints."
    )

    # OBB detection
    angles_outputs: Optional[List[str]] = Field(
        None, description="A list of output names for the angles."
    )

    subtype: str = Field(
        description="YOLO family decoding subtype (e.g. yolov5, yolov6, yolov7 etc.)."
    )
    n_prototypes: Optional[int] = Field(
        None,
        description="Number of prototypes per bbox in YOLO instance segmnetation.",
    )
    n_keypoints: Optional[int] = Field(
        None,
        description="Number of keypoints per bbox in YOLO keypoint detection.",
    )
    is_softmax: Optional[bool] = Field(
        None,
        description="True, if output is already softmaxed in YOLO instance segmentation.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_task_specific_fields(
        cls, values: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                    f"Invalid combination of parameters. Field {param} is not supported for the tasks {tasks}."
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
                    f"Invalid combination of output parameters. Field {param} is not supported for the tasks {tasks}."
                )

        return values
