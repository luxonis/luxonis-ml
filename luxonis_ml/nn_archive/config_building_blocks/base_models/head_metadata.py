from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..enums import ObjectDetectionSubtypeYOLO


class HeadMetadata(BaseModel):
    """Metadata for the basic head.

    @type classes: list
    @ivar classes: Names of object classes recognized by the model.
    @type n_classes: int
    @ivar n_classes: Number of object classes recognized by the model.
    """

    model_config = ConfigDict(extra="allow")
    classes: List[str] = Field(
        description="Names of object classes recognized by the model."
    )
    n_classes: int = Field(
        description="Number of object classes recognized by the model."
    )


class HeadObjectDetectionMetadata(HeadMetadata):
    """Metadata for the object detection head.

    @type iou_threshold: float
    @ivar iou_threshold: Non-max supression threshold limiting boxes intersection.
    @type conf_threshold: float
    @ivar conf_threshold: Confidence score threshold above which a detected object is
        considered valid.
    @type max_det: int
    @ivar max_det: Maximum detections per image.
    @type anchors: list
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


class HeadClassificationMetadata(HeadMetadata):
    """Metadata for the classification head.

    @type is_softmax: bool
    @ivar is_softmax: True, if output is already softmaxed
    """

    is_softmax: bool = Field(description="True, if output is already softmaxed.")


class HeadSegmentationMetadata(HeadMetadata):
    """Metadata for the segmentation head.

    @type is_softmax: bool
    @ivar is_softmax: True, if output is already softmaxed
    """

    is_softmax: bool = Field(description="True, if output is already softmaxed.")


class HeadYOLOMetadata(HeadObjectDetectionMetadata, HeadSegmentationMetadata):
    """Metadata for the YOLO head.

    @type subtype: C{ObjectDetectionSubtypeYOLO}
    @ivar subtype: YOLO family decoding subtype (e.g. yolov5, yolov6, yolov7 etc.)
    @type n_prototypes: int | None
    @ivar n_prototypes: Number of prototypes per bbox in YOLO instance segmnetation.
    @type n_keypoints: int | None
    @ivar n_keypoints: Number of keypoints per bbox in YOLO keypoint detection.
    @type is_softmax: bool | None
    @ivar is_softmax: True, if output is already softmaxed in YOLO instance segmentation
    """

    subtype: ObjectDetectionSubtypeYOLO = Field(
        description="YOLO family decoding subtype (e.g. yolov5, yolov6, yolov7 etc.)."
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
