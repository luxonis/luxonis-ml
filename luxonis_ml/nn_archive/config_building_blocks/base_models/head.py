from abc import ABC
from typing import List, Optional, Union

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
    """Represents head of a model.

    @type parser: str
    @ivar parser: Name of the parser responsible for processing the models output.
    @type outputs: List[str] | None
    @ivar outputs: Specify which outputs are fed into the parser. If None, all outputs
        are fed.
    @type metadata: C{HeadMetadata} | C{HeadObjectDetectionMetadata} |
        C{HeadClassificationMetadata} | C{HeadObjectDetectionSSDMetadata} |
        C{HeadSegmentationMetadata} | C{HeadYOLOMetadata}
    @ivar metadata: Metadata of the parser.
    """

    parser: str = Field(
        description="Name of the parser responsible for processing the models output."
    )
    metadata: Union[
        HeadMetadata,
        HeadObjectDetectionMetadata,
        HeadClassificationMetadata,
        HeadObjectDetectionSSDMetadata,
        HeadSegmentationMetadata,
        HeadYOLOMetadata,
    ] = Field(description="Metadata of the parser.")
    outputs: Optional[List[str]] = Field(
        None,
        description="Specify which outputs are fed into the parser. If None, all outputs are fed.",
    )


HeadType = Head
