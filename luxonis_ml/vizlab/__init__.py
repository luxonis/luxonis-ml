"""vizlab: composition-first, genuinely pretty visualization of CV labels.

The visualization engine of ``luxonis-ml``. It draws bounding boxes,
instance/semantic masks, keypoints with skeletons, classification tags, and
nested sub-labels with Skia — anti-aliasing, true alpha, rounded corners, soft
shadows, and good typography. Install it with the ``viz`` extra
(``pip install luxonis-ml[viz]``).

LDF-native. vizlab renders Luxonis Data Format objects directly: pass a
`Detection` (or a whole `DatasetRecord`) straight to `Image.add`, or render a
loader sample with `visualize_record`. Each render class (`BBox`, `Keypoints`,
`Mask`, ...) also carries a ``from_ldf`` constructor and stays available as a
lower-level drawing primitive. Spatial coordinates are image-normalized in
``[0, 1]``.

    >>> import numpy as np
    >>> from luxonis_ml.ldf import Detection
    >>> from luxonis_ml.vizlab import Image
    >>> det = Detection(
    ...     class_name="car",
    ...     boundingbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
    ... )
    >>> Image(np.zeros((80, 120, 3), np.uint8)).add(det).render().shape
    (80, 120, 4)

Design. `Image` is the composition root: annotations are collected with
`Image.add` (which returns ``self`` for chaining) and nothing is drawn until
`Image.render`, so labels are laid out with knowledge of the whole scene.
Colors, placement, and style are chosen for you and can all be overridden. A
class gets a stable, well-spread color from golden-ratio hue spacing
(`GoldenRatioColors`); pre-seed a `Palette` with your class names to fix those
colors across images. A sub-label's style is *derived* from its parent — a
lighter shade with a thinner, dashed outline — so nesting reads at a glance.
Label chips are placed to avoid overlapping each other and are drawn on top of
every box and mask.

Label types. `BBox` (axis-aligned, or oriented via ``angle``), `Keypoints`
(with a `Skeleton`), `Mask` (instance: binary array, polygon points, or COCO
RLE), `SemanticMask` (dense label map), and `Classification` (image-level
corner tags), plus the `Caption`, `Legend`, and `InfoCard` overlays. Every
annotation may carry a ``label``, a ``score``, and an arbitrary ``payload`` —
the OCR case is a box plus its transcribed text.

    >>> from luxonis_ml.vizlab import BBox
    >>> img = Image(np.zeros((80, 120, 3), np.uint8))
    >>> img.add(BBox(x=0.1, y=0.2, w=0.3, h=0.4, label="car", score=0.9))
    Image(size=120x80, annotations=1)

Composition. `blend` (mixup), `hstack`/`vstack`, and `grid` each render their
inputs and return a new `Image`, leaving the originals untouched.
`Image.with_panel` appends a "second window" of arbitrary JSON-like metadata
(augmentations, source, tags) beside an image as an indented key/value tree
that never occludes the pixels or labels. Theme with `DARK_THEME` (default) or
`LIGHT_THEME`, or your own via `set_default_theme`.

A single runnable script covering every feature lives in ``vizlab_examples/``.
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
