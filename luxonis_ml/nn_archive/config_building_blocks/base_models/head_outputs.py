from typing import List

from pydantic import Field

from .custom_base_model import CustomBaseModel


class Outputs(CustomBaseModel):
    """Parent class for all outputs."""

    pass


class OutputsBasic(Outputs):
    """Represents outputs of a classification model.

    @type predictions: str
    @ivar predictions: Name of the output with predictions.
    """

    predictions: str = Field(description="Name of the output with predictions.")


class OutputsClassification(OutputsBasic):
    pass


class OutputsSegmentation(OutputsBasic):
    pass


class OutputsYOLO(Outputs):
    """Represents outputs of a basic YOLO object detection model.

    @type yolo_outputs: C{List[str]}
    @ivar yolo_outputs: A list of output names for each of the different YOLO grid
        sizes.
    """

    yolo_outputs: List[str] = Field(
        description="A list of output names for each of the different YOLO grid sizes."
    )


class OutputsSSD(Outputs):
    """Represents outputs of a MobileNet SSD object detection model.

    @type boxes: str
    @ivar boxes: Output name corresponding to predicted bounding box coordinates.
    @type scores: str
    @ivar scores: Output name corresponding to predicted bounding box confidence scores.
    """

    boxes: str = Field(
        description="Output name corresponding to predicted bounding box coordinates."
    )
    scores: str = Field(
        description="Output name corresponding to predicted bounding box confidence scores."
    )


class OutputsInstanceSegmentationYOLO(Outputs):
    """Represents outputs of a basic YOLO object detection model.

    @type yolo_outputs: C{List[str]}
    @ivar yolo_outputs: A list of output names for each of the different YOLO grid
        sizes.
    @type mask_outputs: C{List[str]}
    @ivar mask_outputs: A list of output names for each mask output.
    @type protos: str
    @ivar protos: Output name for the protos.
    """

    yolo_outputs: List[str] = Field(
        description="A list of output names for each of the different YOLO grid sizes."
    )
    mask_outputs: List[str] = Field(
        description="A list of output names for each mask output."
    )
    protos: str = Field(description="Output name for the protos.")


class OutputsKeypointDetectionYOLO(OutputsBasic):
    pass
