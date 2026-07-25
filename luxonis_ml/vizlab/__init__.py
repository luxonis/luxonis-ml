"""Render images and Luxonis Data Format annotations.

``vizlab`` is the visualization layer of ``luxonis-ml``. It renders bounding
boxes, instance and semantic masks, keypoints, classification tags, nested
detections, and image-level metadata with a shared palette and collision-aware
label layout. Install the optional renderer with
``pip install luxonis-ml[viz]``.

The main entry point is `Image`. It accepts NumPy arrays, Pillow images, Torch
tensors, and image paths. Add either native vizlab annotations or LDF
`Detection` and `DatasetRecord` objects; drawing is deferred until `Image.render`
so the renderer can place all labels together.

Examples:
    Render an LDF detection:

    >>> import numpy as np
    >>> from luxonis_ml.ldf import Detection
    >>> from luxonis_ml.vizlab import Image
    >>> detection = Detection(
    ...     class_name="car",
    ...     boundingbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
    ... )
    >>> image = Image(np.zeros((80, 120, 3), np.uint8)).add(detection)
    >>> image.render().shape
    (80, 120, 4)

    Build the same scene with a native render annotation:

    >>> from luxonis_ml.vizlab import BBox
    >>> image = Image(np.zeros((80, 120, 3), np.uint8))
    >>> image.add(BBox(x=0.1, y=0.2, w=0.3, h=0.4, label="car", score=0.9))
    Image(size=120x80, annotations=1)

Notes:
    Spatial coordinates are normalized to the source image in ``[0, 1]``.
    Raster mask fills are painted at source resolution, then the image is scaled
    once and vector strokes, keypoints, and label chips are drawn at display
    resolution. Use ``mode="bgr"`` when the input is an OpenCV array.

    The public API has four layers:

    - `Image`, `visualize_record`, and `VizConfig` cover LDF and loader workflows.
    - `BBox`, `Keypoints`, `Mask`, `SemanticMask`, and `Classification` are
      lower-level render annotations.
    - `Style`, `Palette`, and `Theme` control appearance.
    - `blend`, `hstack`, `vstack`, `grid`, and `with_panel` return new composed
      images and leave their inputs unchanged.

    See ``vizlab_examples/`` for a runnable feature overview.

"""

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("viz"):
    from .annotations import (
        Annotation,
        BBox,
        Caption,
        Classification,
        Corner,
        InfoCard,
        Keypoints,
        Legend,
        Mask,
        RenderContext,
        SemanticMask,
        Skeleton,
    )
    from .color import Color
    from .compose import blend, grid, hstack, vstack
    from .convert import VizConfig, visualize_record
    from .geometry import Rect
    from .image import Image
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
    "InfoCard",
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
