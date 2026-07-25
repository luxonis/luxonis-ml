"""Annotation types and the render machinery they share."""

from .base import Annotation, RenderContext
from .bbox import BBox
from .classification import Classification
from .keypoints import Keypoints, Skeleton
from .mask import Mask, SemanticMask
from .overlay import Corner, CornerStack
from .text import Caption, InfoCard, Legend

# The render annotations carry a self-referential ``children: list[Annotation]``
# field. Resolve that forward reference now that every annotation class exists and
# ``Annotation`` is in scope here (each class's own module may not import it).
for _model in (
    Annotation,
    BBox,
    Keypoints,
    Mask,
    SemanticMask,
    CornerStack,
    Classification,
    Caption,
    InfoCard,
    Legend,
):
    _model.model_rebuild()

__all__ = [
    "Annotation",
    "BBox",
    "Caption",
    "Classification",
    "Corner",
    "CornerStack",
    "InfoCard",
    "Keypoints",
    "Legend",
    "Mask",
    "RenderContext",
    "SemanticMask",
    "Skeleton",
]
