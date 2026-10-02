"""Renderable spatial annotations and image-level overlays.

Most users import these objects from `luxonis_ml.vizlab`. Spatial annotations
(`BBox`, `Keypoints`, and `Mask`) extend the corresponding LDF models with
rendering state such as labels, colors, and styles. `Polyline`, `Arrow`, and
`Ruler` are geometry LDF has no model for — a run of points, a relation between
two annotations, and a measured span — so they subclass `Annotation` directly.
`Classification`, `Caption`, `InfoCard`, `Legend`, and `ScaleBar` are
image-level overlays anchored to a `Corner`.

All annotations can be passed to `Image.add`. Attach nested annotations with
`Annotation.add`; child colors and styles are derived from their parent unless
explicitly overridden.
"""

from .base import Annotation, RenderContext
from .bbox import BBox
from .classification import Classification
from .colorbar import ColorBar, FlowWheel
from .distribution import ClassDistribution
from .fields import (
    ArrayField,
    ArrayImage,
    FlowField,
    NormalMap,
    ScalarField,
    SegmentationScores,
)
from .heatmap import Heatmap
from .keypoints import Keypoints
from .mask import Mask, SemanticMask
from .overlay import Corner, CornerStack
from .polyline import Polyline
from .relation import Arrow
from .scalebar import Ruler, ScaleBar
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
    Polyline,
    Arrow,
    Ruler,
    ScaleBar,
    Heatmap,
    CornerStack,
    Classification,
    ClassDistribution,
    Caption,
    ColorBar,
    FlowWheel,
    InfoCard,
    Legend,
    ArrayField,
    ScalarField,
    FlowField,
    NormalMap,
    ArrayImage,
    SegmentationScores,
):
    _model.model_rebuild()

__all__ = [
    "Annotation",
    "ArrayField",
    "ArrayImage",
    "Arrow",
    "BBox",
    "Caption",
    "ClassDistribution",
    "Classification",
    "ColorBar",
    "Corner",
    "CornerStack",
    "FlowField",
    "FlowWheel",
    "Heatmap",
    "InfoCard",
    "Keypoints",
    "Legend",
    "Mask",
    "NormalMap",
    "Polyline",
    "RenderContext",
    "Ruler",
    "ScalarField",
    "ScaleBar",
    "SegmentationScores",
    "SemanticMask",
]
