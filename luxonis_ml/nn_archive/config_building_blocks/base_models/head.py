from abc import ABC
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ..enums import ObjectDetectionSubtypeYOLO
from .head_outputs import (
    Outputs,
    OutputsClassification,
    OutputsInstanceSegmentationYOLO,
    OutputsKeypointDetectionYOLO,
    OutputsOBBDetectionYOLO,
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


class HeadObjectDetectionYOLO(HeadObjectDetection, HeadSegmentation, ABC):
    """Metadata for YOLO object detection head.

    @type family: str
    @ivar family: Decoding family.
    @type outputs: Union[C{OutputsYOLO}, C{OutputsInstanceSegmentationYOLO}, C{OutputsKeypointDetectionYOLO}, C{OutputsOBBDetectionYOLO}]
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

    family: Literal[
        "ObjectDetectionYOLO",
        "InstanceSegmentationYOLO",
        "KeypointDetectionYOLO",
        "OBBDetectionYOLO",
    ] = Field(..., description="Decoding family.")
    outputs: Union[
        OutputsYOLO,
        OutputsInstanceSegmentationYOLO,
        OutputsKeypointDetectionYOLO,
        OutputsOBBDetectionYOLO,
    ] = Field(
        description="A configuration specifying which output names from the `outputs` block of the archive are fed into the head."
    )
    subtype: ObjectDetectionSubtypeYOLO = Field(
        description="YOLO family decoding subtype (e.g. v5, v6, v7 etc.)."
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
        if value not in [
            "ObjectDetectionYOLO",
            "InstanceSegmentationYOLO",
            "KeypointDetectionYOLO",
            "OBBDetectionYOLO",
        ]:
            raise ValueError("Invalid family")
        return value

    @model_validator(mode="before")
    def validate_task_specific_fields(
        cls,
        values,
    ):
        family = values.get("family")

        required_fields = {
            "InstanceSegmentationYOLO": [
                "postprocessor_path",
                "n_prototypes",
                "is_softmax",
            ],
            "KeypointDetectionYOLO": ["n_keypoints"],
        }

        unsupported_fields = {
            "ObjectDetectionYOLO": [
                "postprocessor_path",
                "n_prototypes",
                "n_keypoints",
                "is_softmax",
            ],
            "OBBDetectionYOLO": [
                "postprocessor_path",
                "n_prototypes",
                "n_keypoints",
                "is_softmax",
            ],
            "InstanceSegmentationYOLO": ["n_keypoints"],
            "KeypointDetectionYOLO": [
                "postprocessor_path",
                "n_prototypes",
                "is_softmax",
            ],
        }

        for field in required_fields.get(family, []):
            if field not in values or values[field] is None:
                raise ValueError(f"{family} requires {field}.")

        for field in unsupported_fields.get(family, []):
            if field in values and values[field] is not None:
                raise ValueError(f"{family} does not support {field}.")

        return values

    @model_validator(mode="before")
    def validate_anchors(cls, values):
        if (
            "anchors" in values
            and values["anchors"] is not None
            and values["subtype"] == ObjectDetectionSubtypeYOLO.YOLOv6
        ):
            raise ValueError("YOLOv6 does not support anchors.")
        return values

    @model_validator(mode="before")
    def validate_outputs(cls, values):
        family = values.get("family")
        outputs = values.get("outputs")

        if family == "ObjectDetectionYOLO":
            if not isinstance(outputs, OutputsYOLO):
                raise ValueError("Invalid outputs for ObjectDetectionYOLO.")
        elif family == "InstanceSegmentationYOLO":
            if not isinstance(outputs, OutputsInstanceSegmentationYOLO):
                raise ValueError("Invalid outputs for InstanceSegmentationYOLO.")
        elif family == "KeypointDetectionYOLO":
            if not isinstance(outputs, OutputsKeypointDetectionYOLO):
                raise ValueError("Invalid outputs for KeypointDetectionYOLO.")
        elif family == "OBBDetectionYOLO":
            if not isinstance(outputs, OutputsOBBDetectionYOLO):
                raise ValueError("Invalid outputs for OBBDetectionYOLO.")
        return values


HeadType = Union[
    HeadClassification,
    HeadObjectDetection,
    HeadObjectDetectionYOLO,
    HeadObjectDetectionSSD,
    HeadSegmentation,
]
