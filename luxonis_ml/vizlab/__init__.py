"""Render images and Luxonis Data Format annotations.

``vizlab`` is the visualization layer of ``luxonis-ml``. It renders bounding
boxes, instance and semantic masks, keypoints, classification tags, nested
detections, and image-level metadata with a shared palette and collision-aware
label layout. Install the optional renderer with
``pip install luxonis-ml[viz]``.

The main entry point is `Image`. It accepts NumPy arrays, Pillow images, Torch
tensors, and image paths. Add either native vizlab annotations or LDF
`Detection` and `DatasetRecord` objects; drawing is deferred until `Image.render`
so the renderer can place all labels together. A whole dataset sample renders in
one call with `visualize_record` — the record's annotations become the overlays
and its ``sample_metadata`` becomes a side panel.

A runnable tour of every feature lives in ``vizlab_examples/gallery.py``; it
writes the figures shown and linked below to ``vizlab_examples/output/``. TODO:
publish these images and replace each ``TODO-HOST`` path with its hosted URL.

.. image:: TODO-HOST/from_record.png
   :alt: A DatasetRecord dict rendered to an annotated street frame.

Data in, picture out: a whole ``DatasetRecord`` (or a plain dict shaped like one)
rendered by a single `visualize_record` call — boxes, an instance-segmented
vehicle, nested sub-labels, OCR text, keypoints, semantic masks, and a metadata
panel, all inferred from the data (see the first example below).

.. image:: TODO-HOST/showcase.png
   :alt: One richly annotated sample built from native vizlab annotations.

The same kind of scene composed imperatively from `BBox`, `Mask`, `Keypoints`,
`Classification`, `Caption`, and `Legend`.

.. image:: TODO-HOST/gallery.png
   :alt: An at-a-glance grid with one cell per vizlab feature.

Every label type at a glance. Each area has its own focused figure on the
relevant part of the API:

- `BBox` — detection labels (plain, oriented, OCR payload, nested).
- `Keypoints`, `Mask`, `SemanticMask` — point- and pixel-level labels.
- `Classification`, `Legend`, `Caption` — image-level overlays.
- `Heatmap` — dense fields under gradient themes.
- `ClassDistribution` — predictions in every distribution mode.
- `Theme` — the dark and light themes.
- `grid` (with `blend`, `hstack`, `vstack`) — composition.
- `with_panel` — the metadata side panel.
- `InfoCard` — typography and inline markup.

Examples:
    Render a whole record — the data-in, picture-out path. A `DatasetRecord`
    (here built straight from a plain dict) carries every annotation type; one
    `visualize_record` call turns it into a finished frame, and the record's
    ``sample_metadata`` is appended as a side panel:

    >>> import numpy as np
    >>> from luxonis_ml.ldf import DatasetRecord
    >>> from luxonis_ml.vizlab import visualize_record
    >>> record = DatasetRecord.model_validate(
    ...     {
    ...         "files": {},  # pixels are passed to visualize_record separately
    ...         "task_name": "traffic",
    ...         "sample_metadata": {
    ...             "source": "frame_0007.jpg",
    ...             "split": "train",
    ...         },
    ...         "annotation": [
    ...             {
    ...                 "class_name": "car",
    ...                 "boundingbox": {
    ...                     "x": 0.08,
    ...                     "y": 0.5,
    ...                     "w": 0.4,
    ...                     "h": 0.32,
    ...                 },
    ...                 "sub_detections": {
    ...                     "plate": {
    ...                         "class_name": "plate",
    ...                         "boundingbox": {
    ...                             "x": 0.13,
    ...                             "y": 0.73,
    ...                             "w": 0.13,
    ...                             "h": 0.05,
    ...                         },
    ...                         "metadata": {"text": "LJ 82-A31"},  # OCR chip
    ...                     }
    ...                 },
    ...             },
    ...             {
    ...                 "class_name": "person",
    ...                 "boundingbox": {
    ...                     "x": 0.62,
    ...                     "y": 0.38,
    ...                     "w": 0.15,
    ...                     "h": 0.46,
    ...                 },
    ...                 "keypoints": {
    ...                     "keypoints": [(0.69, 0.44, 2), (0.69, 0.6, 2)]
    ...                 },
    ...             },
    ...             {"class_name": "sunny"},  # a class-only image-level tag
    ...         ],
    ...     }
    ... )
    >>> image = visualize_record(record, np.zeros((360, 640, 3), np.uint8))
    >>> rendered = image.render()
    >>> rendered.shape[0], rendered.shape[2]  # image height and RGBA channels
    (360, 4)
    >>> rendered.shape[1] > 640  # a metadata side panel was appended
    True

    Render a single LDF detection:

    >>> from luxonis_ml.ldf import Detection
    >>> from luxonis_ml.vizlab import Image
    >>> detection = Detection(
    ...     class_name="car",
    ...     boundingbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
    ... )
    >>> image = Image(np.zeros((80, 120, 3), np.uint8)).add(detection)
    >>> image.render().shape
    (80, 120, 4)

    Build a scene with a native render annotation:

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
        ClassDistribution,
        Classification,
        Corner,
        Heatmap,
        InfoCard,
        Keypoints,
        Legend,
        Mask,
        RenderContext,
        SemanticMask,
    )
    from .color import Color
    from .compose import blend, grid, hstack, vstack
    from .convert import VizConfig, visualize_record
    from .geometry import Rect
    from .gradient import (
        DEFAULT_GRADIENT,
        GRADIENTS,
        Gradient,
        resolve_gradient,
    )
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
    "DEFAULT_GRADIENT",
    "DEFAULT_PALETTE",
    "GRADIENTS",
    "LIGHT_THEME",
    "Annotation",
    "BBox",
    "Caption",
    "ClassDistribution",
    "Classification",
    "Color",
    "Corner",
    "GoldenRatioColors",
    "Gradient",
    "Heatmap",
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
    "Style",
    "Theme",
    "VizConfig",
    "blend",
    "get_default_theme",
    "grid",
    "hstack",
    "resolve_gradient",
    "set_default_theme",
    "visualize_record",
    "vstack",
    "with_panel",
]
