from abc import ABC
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ..enums import ObjectDetectionSubtypeYOLO
from .head_outputs import (
    Outputs,
    OutputsClassification,
    OutputsSegmentation,
    OutputsSSD,
    OutputsYOLO,
)


class Head(BaseModel, ABC):
    """Represents head of a model.

    @type family: str
    @ivar family: Decoding family.
    @type outputs: C{Outputs}
    @ivar outputs: A configuration specifying which output names from the `outputs` block of the archive are fed into the head.
    @type classes: list
    @ivar classes: Names of object classes recognized by the model.
    @type n_classes: int
    @ivar n_classes: Number of object classes recognized by the model.
    """

    family: str = Field(description="Decoding family.")
    outputs: Outputs = Field(
        description="A configuration specifying which output names from the `outputs` block of the archive are fed into the head."
    )
    classes: List[str] = Field(
        description="Names of object classes recognized by the model."
    )
    n_classes: int = Field(
        description="Number of object classes recognized by the model."
    )


class HeadObjectDetection(Head, ABC):
    """Metadata for object detection head.

    @type iou_threshold: float
    @ivar iou_threshold: Non-max supression threshold limiting boxes intersection.
    @type conf_threshold: float
    @ivar conf_threshold: Confidence score threshold above which a detected object is
        considered valid.
    @type max_det: int
    @ivar max_det: Maximum detections per image.
    @type anchors: C{Optional[List[List[List[int]]]]}
    @ivar anchors: Predefined bounding boxes of different sizes and aspect ratios. The
        innermost lists are length 2 tuples of box sizes. The middle lists are anchors
        for each output. The outmost lists go from smallest to largest output.
    """

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


class HeadClassification(Head, ABC):
    """Metadata for classification head.

    @type family: str
    @ivar family: Decoding family.
    @type outputs: C{OutputsClassification}
    @ivar outputs: A configuration specifying which output names from the `outputs` block of the archive are fed into the head.
    @type is_softmax: bool
    @ivar is_softmax: True, if output is already softmaxed.
    """

    family: Literal["Classification"] = Field(..., description="Decoding family.")
    outputs: OutputsClassification = Field(
        description="A configuration specifying which output names from the `outputs` block of the archive are fed into the head."
    )
    is_softmax: bool = Field(description="True, if output is already softmaxed.")

    @field_validator("family")
    def validate_label_type(
        cls,
        value,
    ):
        if value != "Classification":
            raise ValueError("Invalid family")
        return value


class HeadObjectDetectionSSD(HeadObjectDetection, ABC):
    """Metadata for SSD object detection head.

    @type family: str
    @ivar family: Decoding family.
    @type outputs: C{OutputsSSD}
    @ivar outputs: A configuration specifying which output names from the `outputs` block of the archive are fed into the head.
    """

    family: Literal["ObjectDetectionSSD"] = Field(..., description="Decoding family.")
    outputs: OutputsSSD = Field(
        description="A configuration specifying which output names from the `outputs` block of the archive are fed into the head."
    )

    @field_validator("family")
    def validate_label_type(
        cls,
        value,
    ):
        if value != "ObjectDetectionSSD":
            raise ValueError("Invalid family")
        return value


class HeadSegmentation(Head, ABC):
    """Metadata for segmentation head.

    @type family: str
    @ivar family: Decoding family.
    @type outputs: C{OutputsSegmentation}
    @ivar outputs: A configuration specifying which output names from the `outputs` block of the archive are fed into the head.
    @type is_softmax: bool
    @ivar is_softmax: True, if output is already softmaxed.
    """

    family: Literal["Segmentation"] = Field(..., description="Decoding family.")
    outputs: OutputsSegmentation = Field(
        description="A configuration specifying which output names from the `outputs` block of the archive are fed into the head."
    )
    is_softmax: bool = Field(description="True, if output is already softmaxed.")

    @field_validator("family")
    def validate_label_type(
        cls,
        value,
    ):
        if value != "Segmentation":
            raise ValueError("Invalid family")
        return value


class HeadYOLO(HeadObjectDetection, HeadSegmentation, ABC):
    """Metadata for YOLO head.

    @type family: str
    @ivar family: Decoding family.
    @type outputs: C{OutputsYOLO}
    @ivar outputs: A configuration specifying which output names from the `outputs` block of the archive are fed into the head.
    @type subtype: ObjectDetectionSubtypeYOLO
    @ivar subtype: YOLO family decoding subtype (e.g. v5, v6, v7 etc.).
    @type postprocessor_path: str | None
    @ivar postprocessor_path: Path to the secondary executable used in YOLO instance segmentation.
    @type n_prototypes: int | None
    @ivar n_prototypes: Number of prototypes per bbox in YOLO instance segmnetation.
    @type n_keypoints: int | None
    @ivar n_keypoints: Number of keypoints per bbox in YOLO keypoint detection.
    @type is_softmax: bool | None
    @ivar is_softmax: True, if output is already softmaxed in YOLO instance segmentation.
    """

    family: Literal["YOLO",] = Field(..., description="Decoding family.")
    outputs: OutputsYOLO = Field(
        description="A configuration specifying which output names from the `outputs` block of the archive are fed into the head."
    )
    subtype: ObjectDetectionSubtypeYOLO = Field(
        description="YOLO family decoding subtype (e.g. yolov5, yolov6, yolov7 etc.)."
    )
    postprocessor_path: Optional[str] = Field(
        None,
        description="Path to the secondary executable used in YOLO instance segmentation.",
    )
    n_prototypes: Optional[int] = Field(
        None, description="Number of prototypes per bbox in YOLO instance segmnetation."
    )
    n_keypoints: Optional[int] = Field(
        None, description="Number of keypoints per bbox in YOLO keypoint detection."
    )
    is_softmax: Optional[bool] = Field(
        None,
        description="True, if output is already softmaxed in YOLO instance segmentation.",
    )

    @field_validator("family")
    def validate_label_type(
        cls,
        value,
    ):
        if value != "YOLO":
            raise ValueError("Invalid family")
        return value

    @model_validator(mode="before")
    def validate_task_specific_fields(
        cls,
        values,
    ):
        defined_params = {k for k, v in values.items() if v is not None}

        common_fields = [
            "family",
            "outputs",
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
            "instance_segmentation": [
                "postprocessor_path",
                "n_prototypes",
                "is_softmax",
            ],
            "keypoint_detection": ["n_keypoints"],
        }

        unsupported_fields = {
            "object_detection": [
                "postprocessor_path",
                "n_prototypes",
                "n_keypoints",
                "is_softmax",
            ],
            "instance_segmentation": ["n_keypoints"],
            "keypoint_detection": [
                "postprocessor_path",
                "n_prototypes",
                "is_softmax",
            ],
        }

        tasks = []
        # Extract the task type
        if all(
            [
                field in defined_params
                for field in required_fields.get("instance_segmentation", [])
            ]
        ):
            tasks.append("instance_segmentation")
        if all(
            [
                field in defined_params
                for field in required_fields.get("keypoint_detection", [])
            ]
        ):
            tasks.append("keypoint_detection")
        if all(
            [
                field not in defined_params
                for field in unsupported_fields.get("object_detection", [])
            ]
        ):
            tasks.append("object_detection")

        if len(tasks) == 0:
            raise ValueError(
                "Invalid combination of parameters. No specific task can be infered."
            )

        for param in defined_params:
            if not any(param in required_fields[task] for task in tasks):
                raise ValueError(
                    f"Invalid combination of parameters. Field {param} is not supported for the tasks {tasks}."
                )

        # Validate Outputs
        outputs = values.get("outputs", {})
        defined_params = {
            k
            for k, v in (
                outputs.model_dump() if isinstance(outputs, BaseModel) else outputs
            ).items()
            if v is not None
        }

        supported_output_params = {
            "instance_segmentation": ["yolo_outputs", "mask_outputs", "protos"],
            "keypoint_detection": ["yolo_outputs", "keypoints"],
            "object_detection": ["yolo_outputs"],
        }

        # Check if all required output fields are present
        if not all(
            [
                field in defined_params
                for task in tasks
                for field in supported_output_params[task]
            ]
        ):
            raise ValueError(f"Invalid output fields for tasks {tasks}")

        # Check if all defined fields are supported
        for param in defined_params:
            if param == "angles" and "object_detection" in tasks:
                continue
            if not any(param in supported_output_params[task] for task in tasks):
                raise ValueError(
                    f"Invalid combination of output parameters. Field {param} is not supported for the tasks {tasks}."
                )

        return values

    @model_validator(mode="before")
    def validate_anchors(cls, values):
        if (
            "anchors" in values
            and values["anchors"] is not None
            and (
                values["subtype"] == ObjectDetectionSubtypeYOLO.YOLOv6
                or values["subtype"] == ObjectDetectionSubtypeYOLO.YOLOv8
            )
        ):
            raise ValueError("YOLOv6 and YOLOv8 do not support anchors.")
        return values


HeadType = Union[
    HeadClassification,
    HeadObjectDetection,
    HeadObjectDetectionSSD,
    HeadSegmentation,
    HeadYOLO,
]
