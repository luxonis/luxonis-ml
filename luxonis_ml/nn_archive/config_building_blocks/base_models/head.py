from abc import ABC

from pydantic import BaseModel, Field

from .head_metadata import (
    HeadClassificationMetadata,
    HeadMetadata,
    HeadObjectDetectionMetadata,
    HeadObjectDetectionSSDMetadata,
    HeadSegmentationMetadata,
    HeadYOLOMetadata,
)


class Head(BaseModel, ABC):
    """Parser head definition for model outputs.

    Attributes:
        name: Optional name of the head.
        parser: Parser responsible for processing the model outputs.
        metadata: Parser metadata.
        outputs: Optional names of model outputs fed into the parser. If
            omitted, all outputs are used.
    """

    name: str | None = Field(None, description="Optional name of the head.")
    parser: str = Field(
        description="Parser responsible for processing the model outputs."
    )
    metadata: (
        HeadObjectDetectionMetadata
        | HeadClassificationMetadata
        | HeadObjectDetectionSSDMetadata
        | HeadSegmentationMetadata
        | HeadYOLOMetadata
        | HeadMetadata
    ) = Field(description="Metadata of the parser.")
    outputs: list[str] | None = Field(
        None,
        description=(
            "Names of model outputs fed into the parser. If omitted, all "
            "outputs are used."
        ),
    )


HeadType = Head
