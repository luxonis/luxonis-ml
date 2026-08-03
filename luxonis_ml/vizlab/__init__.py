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

A runnable tour of every feature lives in ``vizlab_examples/vizlab.ipynb``: it
walks each annotation type beside the LDF data that produces it, and covers the
parts a static figure cannot show — hover tooltips, the interactive HTML export,
and the viewer. ``vizlab_examples/gallery.py`` writes the figures shown and
linked below to ``vizlab_examples/output/``. TODO: publish these images and
replace each ``TODO-HOST`` path with its hosted URL.

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
- `Polyline` — open or closed runs of points (lane lines, boundaries,
  trajectories, wireframes), optionally tapered, arrowed, or colored by a value
  that varies along the run.
- `Arrow` — a relation between two things in the scene, whose endpoints may be
  other annotations and resolve to their edges at render time.
- `ScaleBar`, `Ruler` — a corner-pinned bar of a round length, and a measured
  span between two points, given a caller-supplied pixels-per-unit calibration.
- `Classification`, `Legend`, `Caption` — image-level overlays.
- `Heatmap`, `ColorBar` — dense scalar fields under gradient themes, and the
  key that says which value a color stands for.
- `ScalarField`, `FlowField`, `NormalMap`, `ArrayImage`, `SegmentationScores` —
  the readings of an LDF array label, one class per kind of array: a depth or
  disparity map, a signed error map centered on zero, a two-channel optical flow
  field under the direction wheel (`FlowWheel`), surface normals, an array that
  already is a picture, and a per-class score stack resolved either to a
  `SemanticMask`, to the field of how sure the winning class was
  (`SegmentationScores.confidence`), or to both at once via
  ``weight_by_confidence``, which keeps the class colors but fades the
  pixels the model hesitated on. `luxonis_ml.vizlab.adapters.arrays` works out which one a
  label wants; ``data inspect --array-viz`` draws it, and ``--array-kind`` pins
  it when the shape alone is ambiguous.
- `VideoWriter`, `save_video` — a whole sequence of scenes as one video or
  animated image (see `luxonis_ml.vizlab.video`).
- `Renderable.render_html` — a whole scene as one self-contained interactive
  HTML page: the vector render inlined with working hover tooltips, no
  external request and no companion files. ``scene.save("out.html")`` writes
  it (see `luxonis_ml.vizlab.scene.html`).
- `ClassDistribution` — predictions in every distribution mode.
- `Theme` — the dark and light themes.
- `grid` (with `blend`, `hstack`, `vstack`) — composition.
- `with_panel` — the metadata side panel.
- `InfoCard` — typography.
- `luxonis_ml.vizlab.render.markup` — inline markup. Every string vizlab draws —
  labels, captions, titles, tooltips, panel rows — accepts the same Pango-style
  tags (``<b>``, ``<i>``, ``<u>``, ``<s>``, ``<code>``, ``<span color=… weight=…
  size=…>``); use `escape` for text you did not author.

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
    ...                         "metadata": {
    ...                             "text": "LJ 82-A31"
    ...                         },  # hover meta
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
    >>> (
    ...     rendered.shape[0] >= 360,
    ...     rendered.shape[2],
    ... )  # framed height, RGBA channels
    (True, 4)
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

    - `Image`, `visualize_record`, and `RenderOptions` cover LDF and loader
      workflows.
    - `BBox`, `Keypoints`, `Mask`, `SemanticMask`, and `Classification` are
      lower-level render annotations.
    - `Style`, `Palette`, and `Theme` control appearance; a `RenderOptions`
      bundles them (plus the default gradient and LDF behavior) and is passed via
      ``Image(options=...)`` or installed for a scope with `default_options` /
      `set_default_options`. Per-annotation, `Annotation.styled` layers a partial
      style over the theme and `Annotation.color` / `Palette` pins fix a class's
      color. ``Theme.with_palette("okabe-ito")`` swaps in one of the
      colorblind-safe presets in :data:`PALETTES`; `Palette.from_colormap`
      builds a categorical palette out of a `Gradient`.
    - `blend`, `hstack`, `vstack`, `grid`, and `with_panel` return new composed
      images and leave their inputs unchanged.

    `Renderable.save` writes one finished scene to a file, picking the encoder
    from the extension. `VideoWriter` and `save_video` do the same for a whole
    sequence of scenes, writing one video (``.mp4``, ``.webm``, ``.avi``,
    ``.mkv``, ``.mov``) or one animated image (``.gif``, ``.webp``, ``.apng``,
    ``.avif``); see `luxonis_ml.vizlab.video` for which to reach for.

    Internally those responsibilities live in focused ``adapters``, ``scene``,
    ``render``, ``layout``, ``interaction``, and ``comparison`` packages. Every
    name above is re-exported here, so importing from ``luxonis_ml.vizlab`` is
    enough; reach for a package path only when you want something the top level
    does not export.

    Interactive display lives in the opt-in `luxonis_ml.vizlab.viewer`
    subpackage (import it explicitly: ``from luxonis_ml.vizlab.viewer import
    Viewer``). It is deliberately *not* re-exported here so the core rendering
    package never imports a windowing/OpenCV backend until a viewer is used. Pair
    it with `Frame` (a renderable scene plus the hover/click regions captured by
    `Renderable.frame`). Composition preserves those regions automatically, so
    ``Frame.capture(grid(...))`` is as viewer-ready as ``Frame.capture(image)``.

    See ``vizlab_examples/vizlab.ipynb`` for a runnable feature overview, and
    ``vizlab_examples/README.md`` for what else lives there.

"""

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("viz"):
    from .adapters import visualize_record
    from .annotations import (
        Annotation,
        ArrayField,
        ArrayImage,
        Arrow,
        BBox,
        Caption,
        ClassDistribution,
        Classification,
        ColorBar,
        Corner,
        FlowField,
        FlowWheel,
        Heatmap,
        InfoCard,
        Keypoints,
        Legend,
        Mask,
        NormalMap,
        Polyline,
        RenderContext,
        Ruler,
        ScalarField,
        ScaleBar,
        SegmentationScores,
        SemanticMask,
    )
    from .color import Color
    from .comparison import (
        ComparisonReport,
        ComparisonResult,
        Match,
        Verdict,
        compare,
        confusion_matrix_figure,
        match_detections,
    )
    from .geometry import Rect
    from .gradient import (
        DEFAULT_GRADIENT,
        GRADIENTS,
        Gradient,
        resolve_gradient,
    )
    from .interaction import Frame
    from .layout import (
        Block,
        Controls,
        Hints,
        Swatches,
        blend,
        combine,
        fit_grid,
        grid,
        hstack,
        order_by_position,
        vstack,
        with_panel,
    )
    from .options import (
        RenderOptions,
        current_options,
        default_options,
        set_default_options,
    )
    from .presets import COCO_CLASSES
    from .render import HitMap
    from .render.markup import escape
    from .scene import Composite, Image, Renderable
    from .style import (
        DARK_THEME,
        DEFAULT_PALETTE,
        LIGHT_THEME,
        PALETTES,
        ColormapColors,
        CVDDistinctColors,
        GoldenRatioColors,
        LabelPlacement,
        MaskOutline,
        Palette,
        Style,
        Theme,
        resolve_generator,
    )
    from .tooltip import Tooltip
    from .video import (
        VIDEO_FORMATS,
        VideoWriter,
        is_video_path,
        save_video,
    )

__all__ = [
    "COCO_CLASSES",
    "DARK_THEME",
    "DEFAULT_GRADIENT",
    "DEFAULT_PALETTE",
    "GRADIENTS",
    "LIGHT_THEME",
    "PALETTES",
    "VIDEO_FORMATS",
    "Annotation",
    "ArrayField",
    "ArrayImage",
    "Arrow",
    "BBox",
    "Block",
    "CVDDistinctColors",
    "Caption",
    "ClassDistribution",
    "Classification",
    "Color",
    "ColorBar",
    "ColormapColors",
    "ComparisonReport",
    "ComparisonResult",
    "Composite",
    "Controls",
    "Corner",
    "FlowField",
    "FlowWheel",
    "Frame",
    "GoldenRatioColors",
    "Gradient",
    "Heatmap",
    "Hints",
    "HitMap",
    "Image",
    "InfoCard",
    "Keypoints",
    "LabelPlacement",
    "Legend",
    "Mask",
    "MaskOutline",
    "Match",
    "NormalMap",
    "Palette",
    "Polyline",
    "Rect",
    "RenderContext",
    "RenderOptions",
    "Renderable",
    "Ruler",
    "ScalarField",
    "ScaleBar",
    "SegmentationScores",
    "SemanticMask",
    "Style",
    "Swatches",
    "Theme",
    "Tooltip",
    "Verdict",
    "VideoWriter",
    "blend",
    "combine",
    "compare",
    "confusion_matrix_figure",
    "current_options",
    "default_options",
    "escape",
    "fit_grid",
    "grid",
    "hstack",
    "is_video_path",
    "match_detections",
    "order_by_position",
    "resolve_generator",
    "resolve_gradient",
    "save_video",
    "set_default_options",
    "visualize_record",
    "vstack",
    "with_panel",
]
