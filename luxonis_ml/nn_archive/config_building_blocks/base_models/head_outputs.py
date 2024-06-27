from typing import List, Optional

from pydantic import Field

from luxonis_ml.utils import BaseModelExtraForbid


class Outputs(BaseModelExtraForbid):
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
    """Represents outputs of a YOLO model.

    @type yolo_outputs: C{List[str]}
    @ivar yolo_outputs: A list of output names for each of the different YOLO grid
        sizes.
    @type mask_outputs: C{List[str]} | None
    @ivar mask_outputs: A list of output names for each mask output.
    @type protos: str | None
    @ivar protos: Output name for the protos.
    @type keypoints: str | None
    @ivar keypoints: Output name for the keypoints.
    @type angles: str | None
    @ivar angles: Output name for the angles.
    """

    yolo_outputs: List[str] = Field(
        description="A list of output names for each of the different YOLO grid sizes."
    )

    # Instance segmentation
    mask_outputs: Optional[List[str]] = Field(
        description="A list of output names for each mask output."
    )
    protos: Optional[str] = Field(description="Output name for the protos.")

    # Keypoint detection
    keypoints: Optional[str] = Field(description="Output name for the keypoints.")

    # OBB detection
    angles: Optional[str] = Field(description="Output name for the angles.")


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
