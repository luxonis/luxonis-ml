"""vizlab: composition-first, pretty visualization of computer-vision labels.

The visualization engine of luxonis-ml. The public API is small and fluent: an
`Image` is the composition root that annotations
attach to; nothing is drawn until `Image.render`
is called.

vizlab natively ingests Luxonis Data Format objects — pass a
`Detection` (or a whole
`DatasetRecord`) straight to
`Image.add`. The render classes (`BBox`,
`Keypoints`, `Mask`, ...) remain available as lower-level drawing primitives.

Example:
    >>> import numpy as np
    >>> from luxonis_ml.vizlab import Image
    >>> viz = Image(np.zeros((100, 100, 3), np.uint8))
    >>> viz.width, viz.height
    (100, 100)

"""

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("viz"):
    from .annotations import (
        Annotation,
        BBox,
        Caption,
        Classification,
        Corner,
        Keypoints,
        Legend,
        Mask,
        RenderContext,
        SemanticMask,
        Skeleton,
    )
    from .color import Color
    from .compose import blend, grid, hstack, vstack
    from .geometry import Rect
    from .image import Image
    from .ldf import VizConfig, visualize_record
    from .panel import with_panel
    from .presets import COCO_CLASSES
    from .style import (
        DARK_THEME,
        DEFAULT_PALETTE,
        LIGHT_THEME,
        GoldenRatioColors,
        LabelPlacement,
        Palette,
        Style,
        Theme,
        get_default_theme,
        set_default_theme,
    )

__all__ = [
    "COCO_CLASSES",
    "DARK_THEME",
    "DEFAULT_PALETTE",
    "LIGHT_THEME",
    "Annotation",
    "BBox",
    "Caption",
    "Classification",
    "Color",
    "Corner",
    "GoldenRatioColors",
    "Image",
    "Keypoints",
    "LabelPlacement",
    "Legend",
    "Mask",
    "Palette",
    "Rect",
    "RenderContext",
    "SemanticMask",
    "Skeleton",
    "Style",
    "Theme",
    "VizConfig",
    "blend",
    "get_default_theme",
    "grid",
    "hstack",
    "set_default_theme",
    "visualize_record",
    "vstack",
    "with_panel",
]
