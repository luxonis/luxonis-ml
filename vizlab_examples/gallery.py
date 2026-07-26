"""Documentation-figure generator for ``luxonis-ml`` visualization and augments.

Renders a set of ONGs into ``vizlab_examples/output/`` for the docs. Two groups:
the ``vizlab`` feature figures (synthetic, no external assets), and the custom
augmentation figures (real dataset samples run through each transform).

vizlab feature figures — each a self-contained group that drops into the docs:

- ``showcase.png`` — one richly annotated sample (most features at once) on a
  synthetic street frame, with a metadata side panel.
- ``from_record.png`` — the same kind of rich sample produced directly from a
  large ``DatasetRecord``-compatible dict via ``visualize_record`` (data in,
  picture out — no annotation objects built by hand).
- ``gallery.png`` — one at-a-glance grid with a single cell per feature.
- ``detection.png`` — box-based labels (plain, oriented, OCR payload, nested).
- ``masks_keypoints.png`` — pixel- and point-level labels (keypoints, instance
  mask, polygon mask, semantic mask).
- ``overlays.png`` — things drawn *over* the image (classification tags,
  captions + legend, heatmap, class distribution).
- ``themes.png`` — the same scene in the dark and light themes.
- ``heatmaps.png`` — one field under several gradient themes.
- ``distributions.png`` — one prediction under every distribution mode.
- ``compose.png`` — blend / stack / grid composition.
- ``panel.png`` — the metadata side panel.
- ``typography.png`` — bundled fonts and inline ``<b>/<i>/<code>`` markup.

Custom-augmentation figures — one before/after strip per transform in
``luxonis_ml.data.augmentations.custom``, drawn with vizlab so boxes, keypoints,
and masks transform together. They run on the ``D2_ParkingLot_Native`` dataset,
downloaded automatically from the GCS test bucket (a local dump directory in the
repo root is used instead when present), so they need the ``data`` extra and,
for the download, GCS credentials:

- ``aug_letterbox.png`` — `LetterboxResize`.
- ``aug_mixup.png`` — `MixUp`.
- ``aug_mosaic.png`` — `Mosaic4`.
- ``aug_horizontal_flip.png`` / ``aug_vertical_flip.png`` /
  ``aug_transpose.png`` — the symmetric keypoint flips, on a synthetic pose with
  left (cyan) and right (rose) joints colored, so the mirror keeps each side on
  the correct anatomical part (a naive flip would swap them).

The vizlab figures synthesize their own backdrops with numpy (no external
assets); the augmentation figures are skipped with a hint when the ``data``
extra, the dataset, or GCS credentials are unavailable.

Run it from a checkout with the ``viz`` (and, for augments, ``data``) extra::

    python vizlab_examples/gallery.py

All spatial coordinates are image-normalized in ``[0, 1]`` (the Luxonis Data
Format convention): a box is ``x, y`` (top-left) plus ``w, h``; a keypoint is
``(x, y, visibility)`` with COCO visibility ``0``/``1``/``2``.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich import print

if TYPE_CHECKING:
    from luxonis_ml.data import LuxonisDataset
    from luxonis_ml.vizlab import VizConfig

from luxonis_ml.vizlab import (
    LIGHT_THEME,
    BBox,
    Caption,
    ClassDistribution,
    Classification,
    Corner,
    Gradient,
    Heatmap,
    Image,
    InfoCard,
    Keypoints,
    Legend,
    Mask,
    SemanticMask,
    blend,
    grid,
    hstack,
)

OUTPUT_DIR = Path(__file__).parent / "output"
_W, _H = 340, 250


def gradient(width: int, height: int, *, hue: float = 0.58) -> np.ndarray:
    """Build a smooth diagonal gradient to draw annotations on."""
    ys = np.linspace(0.0, 1.0, height)[:, None]
    xs = np.linspace(0.0, 1.0, width)[None, :]
    base = 30 + (xs + ys) / 2.0 * 70
    rgb = np.stack(
        [base * (0.7 + 0.3 * hue), base * 0.9, base * (0.9 + 0.4 * (1 - hue))],
        axis=-1,
    )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save(image: Image, name: str) -> Path:
    """Render an `Image` to the output directory and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    image.save(path)
    return path


def street_scene(width: int, height: int) -> np.ndarray:
    """Paint a believable street backdrop — sky, buildings, road, lane markings.

    A stand-in for a real photo so the showcase sample looks like a genuine
    frame; every shape is drawn with numpy, so no external asset is needed. The
    horizon sits at the vertical midpoint, with a receding lane down the center.
    """
    img = np.zeros((height, width, 3), dtype=np.float64)
    horizon = int(height * 0.5)

    # Sky: a vertical gradient from a deeper blue at the top to a pale haze at
    # the horizon.
    ramp = np.linspace(0.0, 1.0, horizon)[:, None]
    sky = (
        np.array([96, 132, 194]) * (1 - ramp)
        + np.array([206, 214, 224]) * ramp
    )
    img[:horizon] = sky[:, None, :]

    # Road: asphalt, a touch lighter toward the camera.
    road_ramp = np.linspace(0.0, 1.0, height - horizon)[:, None]
    road = (
        np.array([68, 70, 78]) * (1 - road_ramp)
        + np.array([98, 100, 108]) * road_ramp
    )
    img[horizon:] = road[:, None, :]

    # A skyline of flat building silhouettes, leaving a gap at the center for
    # the road's vanishing point.
    band = int(height * 0.16)
    buildings = [
        (0.00, 0.12, 0.70, 58),
        (0.12, 0.09, 0.48, 66),
        (0.21, 0.10, 0.88, 50),
        (0.31, 0.07, 0.42, 72),
        (0.56, 0.09, 0.58, 60),
        (0.65, 0.12, 0.92, 48),
        (0.77, 0.08, 0.52, 68),
        (0.85, 0.15, 0.74, 54),
    ]
    for x0, wf, hf, shade in buildings:
        bx0, bx1 = int(x0 * width), int((x0 + wf) * width)
        by0 = int(horizon - hf * band)
        img[by0:horizon, bx0:bx1] = [shade, shade + 6, shade + 18]

    # Dashed center line, widening toward the camera for a sense of perspective.
    cx = width // 2
    for y in range(horizon + 6, height, 26):
        t = (y - horizon) / (height - horizon)
        half_w = max(1, int(2 + t * 6))
        dash = int(6 + t * 16)
        img[y : y + dash, cx - half_w : cx + half_w] = [212, 206, 178]

    return np.clip(img, 0, 255).astype(np.uint8)


# --- one cell per feature ---------------------------------------------------


def _boxes() -> Image:
    # Colors are assigned from the class name; the same class is always the
    # same color. `score` shows as a percentage, `payload` as free text (OCR).
    return (
        Image(gradient(_W, _H, hue=0.58))
        .add(BBox(x=0.08, y=0.16, w=0.5, h=0.68, label="person", score=0.97))
        .add(BBox(x=0.44, y=0.36, w=0.47, h=0.5, label="dog", score=0.86))
    )


def _oriented() -> Image:
    # `angle` (degrees, about the box center) rotates a box for aerial imagery,
    # scene text, or rotated-object detectors.
    return (
        Image(gradient(_W, _H, hue=0.62))
        .add(
            BBox(
                x=0.1,
                y=0.28,
                w=0.42,
                h=0.24,
                angle=28,
                label="ship",
                score=0.9,
            )
        )
        .add(
            BBox(
                x=0.5,
                y=0.4,
                w=0.35,
                h=0.35,
                angle=-18,
                label="roof",
                score=0.8,
            )
        )
    )


def _payload() -> Image:
    # The OCR case: a box plus its transcribed text on the same chip.
    return (
        Image(gradient(_W, _H, hue=0.12))
        .add(
            BBox(
                x=0.08,
                y=0.2,
                w=0.5,
                h=0.22,
                label="word",
                score=0.99,
                payload="INVOICE",
            )
        )
        .add(
            BBox(x=0.08, y=0.56, w=0.84, h=0.22, label="line", payload="#1042")
        )
    )


def _keypoints() -> Image:
    # COCO visibility: 2 = visible (solid dot), 1 = occluded (hollow ring). The
    # right arm here is occluded.
    pose = [
        (0.5, 0.24, 2),
        (0.44, 0.48, 2),
        (0.56, 0.48, 1),
        (0.38, 0.72, 2),
        (0.62, 0.72, 1),
    ]
    edges = [(0, 1), (0, 2), (1, 3), (2, 4)]
    return Image(gradient(_W, _H, hue=0.68)).add(
        Keypoints(keypoints=pose, edges=edges, label="pose")
    )


def _instance_mask() -> Image:
    # A binary (H, W) array; its outline is traced (OpenCV) and smoothed.
    ys, xs = np.ogrid[:_H, :_W]
    disc = ((xs - 170) ** 2 + (ys - 125) ** 2 <= 95**2).astype(np.uint8)
    return Image(gradient(_W, _H, hue=0.4)).add(Mask(mask=disc, label="moon"))


def _polygon_mask() -> Image:
    # A polygon given as normalized points (rasterized to a mask of `width` x
    # `height`).
    leaf = [
        (0.26, 0.32),
        (0.44, 0.2),
        (0.68, 0.24),
        (0.82, 0.5),
        (0.64, 0.78),
        (0.34, 0.74),
        (0.22, 0.48),
    ]
    return Image(gradient(_W, _H, hue=0.44)).add(
        Mask(points=leaf, width=_W, height=_H, label="leaf")
    )


def _semantic() -> Image:
    # A dense (H, W) integer label map, colored per class id; class 0 is
    # background and left undrawn.
    labels = np.zeros((_H, _W), dtype=np.int32)
    labels[: int(_H * 0.55)] = 1
    labels[int(_H * 0.55) :] = 2
    labels[120:200, 210:300] = 3
    names = {0: "background", 1: "sky", 2: "ground", 3: "car"}
    return Image(gradient(_W, _H, hue=0.5)).add(
        SemanticMask(labels=labels, names=names, ignore_index=0)
    )


def _nested() -> Image:
    # A child's style is derived from its parent — lighter, thinner, dashed —
    # so nesting reads at a glance.
    car = BBox(x=0.08, y=0.16, w=0.66, h=0.72, label="car", score=0.98)
    car.add(BBox(x=0.32, y=0.4, w=0.34, h=0.4, label="driver", score=0.9))
    return Image(gradient(_W, _H, hue=0.55)).add(car)


def _classification() -> Image:
    # Image-level tags stacked in a corner (multi-label, with scores).
    return (
        Image(gradient(_W, _H, hue=0.12))
        .add(BBox(x=0.26, y=0.28, w=0.56, h=0.56, label="beach", score=0.8))
        .add(
            Classification(
                tags=[("outdoor", 0.98), ("sunny", 0.7)],
                corner=Corner.TOP_LEFT,
            )
        )
    )


def _captions_legend() -> Image:
    # Overlays: captions (a filename and a bold title) and a class-color key.
    return (
        Image(gradient(_W, _H, hue=0.05))
        .add(BBox(x=0.08, y=0.16, w=0.44, h=0.68, label="car", score=0.96))
        .add(BBox(x=0.44, y=0.36, w=0.44, h=0.5, label="truck", score=0.88))
        .add(Caption(text="frame_0421.jpg", corner=Corner.TOP_LEFT))
        .add(Caption(text="Detections", title=True, corner=Corner.TOP_RIGHT))
        .add(
            Legend(
                entries=["car", "truck", ("road", "#5566aa")],
                title="classes",
                corner=Corner.BOTTOM_RIGHT,
            )
        )
    )


def _blob_field(
    width: int, height: int, centers: list[tuple[float, float, float]]
) -> np.ndarray:
    """Sum of Gaussian bumps — a smooth field to stand in for a saliency map.

    Each center is ``(cx, cy, sigma)`` in normalized image coordinates.
    """
    ys = np.linspace(0.0, 1.0, height)[:, None]
    xs = np.linspace(0.0, 1.0, width)[None, :]
    field = np.zeros((height, width), dtype=np.float64)
    for cx, cy, sigma in centers:
        field += np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)))
    return field


def _heatmap() -> Image:
    # A dense scalar field (here two hot-spots) colored through a gradient and
    # blended over the image; low values fade to transparent. The default
    # gradient is "turbo"; see `render_heatmap_themes` for the other themes.
    field = _blob_field(_W, _H, [(0.35, 0.4, 0.16), (0.7, 0.68, 0.1)])
    return Image(gradient(_W, _H, hue=0.55)).add(Heatmap(values=field))


# A synthetic softmax over classes, standing in for a model prediction.
_PREDICTION = {
    "husky": 0.58,
    "malamute": 0.24,
    "wolf": 0.09,
    "samoyed": 0.05,
    "corgi": 0.04,
}


def _distribution() -> Image:
    # Model predictions are a probability distribution, not one class. The
    # default "bars" mode ranks the classes; `ground_truth` marks the true one
    # (here the model's top guess is wrong). See `render_distribution_modes`.
    return Image(gradient(_W, _H, hue=0.6)).add(
        ClassDistribution(
            probabilities=_PREDICTION,
            ground_truth="malamute",
            title="prediction",
        )
    )


def _light_theme() -> Image:
    # DARK_THEME is the default; pass LIGHT_THEME (or your own) for a light look.
    return (
        Image(np.full((_H, _W, 3), 236, np.uint8), theme=LIGHT_THEME)
        .add(BBox(x=0.08, y=0.16, w=0.5, h=0.68, label="person", score=0.97))
        .add(BBox(x=0.44, y=0.36, w=0.47, h=0.5, label="dog", score=0.86))
    )


def render_overview() -> Path:
    """One at-a-glance grid with a single cell per feature.

    A less-focused companion to the per-topic figures below: it fits every label
    type on one page so the top-level docs can show the whole surface at once,
    then link out to the focused figures for detail.
    """
    cells = {
        "bounding boxes": _boxes(),
        "oriented boxes": _oriented(),
        "payload (OCR)": _payload(),
        "keypoints": _keypoints(),
        "instance mask": _instance_mask(),
        "polygon mask": _polygon_mask(),
        "semantic mask": _semantic(),
        "heatmap": _heatmap(),
        "class distribution": _distribution(),
        "nested sub-labels": _nested(),
        "classification": _classification(),
        "captions + legend": _captions_legend(),
    }
    return save(
        grid(list(cells.values()), ncols=3, titles=list(cells)),
        "gallery.png",
    )


def render_detection() -> Path:
    """Box-based labels: plain, oriented, OCR payload, and nested sub-labels."""
    cells = {
        "bounding boxes": _boxes(),
        "oriented boxes": _oriented(),
        "payload (OCR)": _payload(),
        "nested sub-labels": _nested(),
    }
    return save(
        grid(list(cells.values()), ncols=2, titles=list(cells)),
        "detection.png",
    )


def render_masks_keypoints() -> Path:
    """Pixel- and point-level labels: keypoints and the three mask kinds."""
    cells = {
        "keypoints": _keypoints(),
        "semantic mask": _semantic(),
        "instance mask": _instance_mask(),
        "polygon mask": _polygon_mask(),
    }
    return save(
        grid(list(cells.values()), ncols=2, titles=list(cells)),
        "masks_keypoints.png",
    )


def render_overlays() -> Path:
    """Overlays drawn on top of the image: tags, chrome, and analytics."""
    cells = {
        "classification": _classification(),
        "captions + legend": _captions_legend(),
        "heatmap": _heatmap(),
        "class distribution": _distribution(),
    }
    return save(
        grid(list(cells.values()), ncols=2, titles=list(cells)),
        "overlays.png",
    )


def render_themes() -> Path:
    """Render the same detections in the default dark theme and the light one."""
    return save(
        grid(
            [_boxes(), _light_theme()],
            ncols=2,
            titles=["dark theme", "light theme"],
        ),
        "themes.png",
    )


def render_compose() -> Path:
    """Blend (mixup), stack, and grid — each returns a new image."""
    cat = Image(gradient(300, 220, hue=0.58)).add(
        BBox(x=0.13, y=0.18, w=0.6, h=0.5, label="cat", score=0.96)
    )
    dog = Image(gradient(300, 220, hue=0.08)).add(
        BBox(x=0.13, y=0.18, w=0.6, h=0.5, label="dog", score=0.91)
    )
    mixed = blend(cat, dog, alpha=0.4)
    return save(
        hstack([cat, dog, mixed], titles=["cat", "dog", "mixup"]),
        "compose.png",
    )


def render_heatmap_themes() -> Path:
    """Show one field under several gradient themes, plus a custom gradient.

    ``Heatmap.gradient`` takes the name of a built-in theme or any `Gradient`;
    build a custom one from a list of colors with ``Gradient.from_colors``.
    """
    field = _blob_field(
        300, 220, [(0.3, 0.35, 0.15), (0.68, 0.62, 0.12), (0.5, 0.85, 0.08)]
    )
    themes: list[str | Gradient] = [
        "turbo",
        "viridis",
        "magma",
        "jet",
        Gradient.from_colors(["#000000", "#00e5ff", "#ffffff"]),  # custom
    ]
    titles = ["turbo", "viridis", "magma", "jet", "custom"]
    cells = [
        Image(gradient(300, 220, hue=0.55)).add(
            Heatmap(values=field, gradient=theme)
        )
        for theme in themes
    ]
    return save(grid(cells, ncols=5, titles=titles), "heatmaps.png")


def render_distribution_modes() -> Path:
    """Show one prediction under every `ClassDistribution` render mode.

    ``mode`` picks the look; ``ground_truth`` highlights the correct class (row,
    chip, segment, wedge, or a ✓/✗ on the gauge) so a wrong top-1 is obvious.
    """
    modes = ["bars", "chips", "gauge", "stacked", "pie", "donut"]
    cells = [
        Image(gradient(300, 260, hue=0.6)).add(
            ClassDistribution(
                probabilities=_PREDICTION,
                mode=mode,
                ground_truth="malamute",
                corner=Corner.TOP_LEFT,
            )
        )
        for mode in modes
    ]
    return save(grid(cells, ncols=3, titles=modes), "distributions.png")


def render_panel() -> Path:
    """Append a metadata sidebar that never occludes the pixels or labels."""
    metadata = {
        "source": "coco/val2017/000000042.jpg",
        "split": "train",
        "augmentations": ["horizontal_flip", "gaussian_blur (0.5)"],
        "tags": {"difficulty": "hard", "verified": True},
    }
    img = (
        Image(gradient(420, 320, hue=0.58))
        .add(BBox(x=0.1, y=0.2, w=0.43, h=0.75, label="person", score=0.97))
        .add(BBox(x=0.43, y=0.34, w=0.48, h=0.53, label="dog", score=0.86))
        .add(Classification(tags=[("outdoor", 0.98)]))
    )
    return save(img.with_panel(metadata, title="metadata"), "panel.png")


# A side-view car silhouette (normalized points), for the instance mask.
_CAR_SILHOUETTE = [
    (0.06, 0.82),
    (0.06, 0.75),
    (0.10, 0.70),
    (0.16, 0.685),
    (0.185, 0.62),
    (0.25, 0.61),
    (0.29, 0.67),
    (0.33, 0.70),
    (0.345, 0.75),
    (0.345, 0.82),
    (0.30, 0.845),
    (0.11, 0.845),
]

# A standing pedestrian pose (image-normalized) and its skeleton edges. COCO
# visibility: 2 = visible, 1 = occluded (the right wrist and ankle here).
_POSE = [
    (0.415, 0.37, 2),  # 0 nose
    (0.400, 0.42, 2),  # 1 left shoulder
    (0.435, 0.42, 2),  # 2 right shoulder
    (0.385, 0.49, 2),  # 3 left elbow
    (0.450, 0.49, 2),  # 4 right elbow
    (0.380, 0.55, 2),  # 5 left wrist
    (0.455, 0.55, 1),  # 6 right wrist
    (0.405, 0.56, 2),  # 7 left hip
    (0.430, 0.56, 2),  # 8 right hip
    (0.400, 0.65, 2),  # 9 left knee
    (0.435, 0.65, 2),  # 10 right knee
    (0.400, 0.75, 2),  # 11 left ankle
    (0.435, 0.75, 1),  # 12 right ankle
]
_POSE_EDGES = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (3, 5),
    (2, 4),
    (4, 6),
    (1, 7),
    (2, 8),
    (7, 8),
    (7, 9),
    (9, 11),
    (8, 10),
    (10, 12),
]


def _ground_segmentation(width: int, height: int) -> SemanticMask:
    """Segment the drivable road and the left sidewalk as a dense label map."""
    labels = np.zeros((height, width), dtype=np.int32)
    horizon = int(height * 0.5)
    ys = np.arange(height)[:, None]
    xs = np.arange(width)[None, :]
    labels[horizon:] = 1  # road: everything below the horizon
    # Sidewalk: a wedge along the lower-left, narrowing toward the horizon.
    edge = (0.30 - (ys / height) * 0.12) * width
    labels[(ys > horizon) & (xs < edge)] = 2
    return SemanticMask(
        labels=labels,
        names={0: "background", 1: "road", 2: "sidewalk"},
        ignore_index=0,
        fill_alpha=0.3,
    )


def render_showcase() -> Path:
    """One richly annotated sample using most features on a synthetic frame.

    Emulates a real street frame (painted with numpy) carrying the full spread
    of annotations at realistic proportions: ground semantic segmentation, an
    instance-segmented car with a nested license plate (OCR payload) and driver,
    an oriented parked car, a distant car, a pedestrian with a keypoint skeleton,
    scene-level classification tags, captions, a class legend, and a metadata
    side panel.
    """
    scene_w, scene_h = 960, 600
    img = Image(street_scene(scene_w, scene_h))

    # Ground truth for the surfaces the car drives and the pedestrian walks on.
    img.add(_ground_segmentation(scene_w, scene_h))

    # Foreground car: an instance mask carrying a nested license plate (with its
    # transcribed text) and the driver — children derive the car's color.
    car = Mask(
        points=_CAR_SILHOUETTE,
        width=scene_w,
        height=scene_h,
        label="car",
        score=0.98,
    )
    car.add(
        BBox(
            x=0.285,
            y=0.79,
            w=0.055,
            h=0.032,
            label="plate",
            payload="5A2 8391",
        )
    )
    car.add(BBox(x=0.165, y=0.635, w=0.05, h=0.045, label="driver", score=0.9))
    img.add(car)

    # A parked car at an angle (oriented box) and a distant one (small box).
    img.add(
        BBox(x=0.58, y=0.60, w=0.28, h=0.16, angle=-12, label="car", score=0.9)
    )
    img.add(BBox(x=0.455, y=0.485, w=0.07, h=0.05, label="car", score=0.68))

    # A pedestrian: a box with its pose skeleton nested inside it.
    person = BBox(x=0.37, y=0.34, w=0.095, h=0.42, label="person", score=0.95)
    person.add(Keypoints(keypoints=_POSE, edges=_POSE_EDGES, label="pose"))
    img.add(person)

    # Image-level chrome: scene tags, a title, the source filename, and a key.
    img.add(
        Classification(
            tags=[("daytime", 0.99), ("urban", 0.94), ("clear", 0.88)],
            corner=Corner.TOP_LEFT,
        )
    )
    img.add(
        Caption(text="Annotated sample", title=True, corner=Corner.TOP_RIGHT)
    )
    img.add(
        Caption(text="seq_000e/frame_000123.jpg", corner=Corner.BOTTOM_LEFT)
    )
    img.add(
        Legend(
            entries=["car", "person", "road", "sidewalk"],
            title="classes",
            corner=Corner.BOTTOM_RIGHT,
        )
    )

    metadata = {
        "source": "seq_000e/frame_000123.jpg",
        "split": "train",
        "resolution": "1920x1080",
        "camera": "OAK-D Pro",
        "weather": "clear",
        "time": "14:32",
        "location": {"lat": 46.0569, "lon": 14.5058},
        "objects": {"car": 3, "person": 1},
        "augmentations": ["horizontal_flip", "color_jitter"],
    }
    return save(
        img.with_panel(metadata, title="Sample metadata"), "showcase.png"
    )


# A pedestrian pose for the record showcase, placed for a person around x≈0.44.
_RECORD_POSE = [
    (0.440, 0.38, 2),  # 0 nose
    (0.425, 0.43, 2),  # 1 left shoulder
    (0.460, 0.43, 2),  # 2 right shoulder
    (0.410, 0.50, 2),  # 3 left elbow
    (0.475, 0.50, 2),  # 4 right elbow
    (0.405, 0.57, 2),  # 5 left wrist
    (0.480, 0.57, 1),  # 6 right wrist
    (0.430, 0.575, 2),  # 7 left hip
    (0.455, 0.575, 2),  # 8 right hip
    (0.425, 0.67, 2),  # 9 left knee
    (0.460, 0.67, 2),  # 10 right knee
    (0.425, 0.77, 2),  # 11 left ankle
    (0.460, 0.77, 1),  # 12 right ankle
]
_POSE_NAMES = [
    "nose",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]


def render_from_record() -> Path:
    """Convert one large ``DatasetRecord``-compatible dict straight to a picture.

    Everything below is plain data — the exact shape a dataset generator yields
    or a loader round-trips (see `loader_output_to_records`). A single
    `DatasetRecord.model_validate` +
    `visualize_record` call turns it into the
    finished frame: boxes, an instance-segmented truck, per-object semantic
    masks, a nested license plate with its OCR text, keypoints, image-level
    classification tags, and a metadata side panel — all inferred from the data,
    with no vizlab annotation objects built by hand.
    """
    from luxonis_ml.ldf import DatasetRecord
    from luxonis_ml.vizlab import VizConfig, visualize_record

    w, h = 960, 600
    record_dict = {
        "files": {},  # the pixels are passed to visualize_record separately
        "task_name": "scene",
        "sample_metadata": {
            "source": "seq_014/frame_000512.jpg",
            "split": "val",
            "city": "Ljubljana",
            "weather": "overcast",
            "annotator": "auto + review",
            "augmentations": ["horizontal_flip", "color_jitter"],
        },
        "annotation": [
            # Per-class semantic segmentation of the ground surfaces.
            {
                "class_name": "road",
                "segmentation": {
                    "points": [
                        (0.0, 0.5),
                        (1.0, 0.5),
                        (1.0, 1.0),
                        (0.0, 1.0),
                    ],
                    "width": w,
                    "height": h,
                },
            },
            {
                "class_name": "sidewalk",
                "segmentation": {
                    "points": [
                        (0.0, 0.6),
                        (0.2, 0.6),
                        (0.1, 1.0),
                        (0.0, 1.0),
                    ],
                    "width": w,
                    "height": h,
                },
            },
            # An instance-segmented truck (polygon) carrying tracking metadata.
            {
                "class_name": "truck",
                "instance_id": 0,
                "instance_segmentation": {
                    "points": [
                        (0.60, 0.50),
                        (0.62, 0.40),
                        (0.87, 0.40),
                        (0.90, 0.50),
                        (0.90, 0.74),
                        (0.60, 0.74),
                    ],
                    "width": w,
                    "height": h,
                },
                "metadata": {"track_id": 11},
            },
            # A car with a nested license plate (its text) and driver.
            {
                "class_name": "car",
                "instance_id": 1,
                "boundingbox": {
                    "x": 0.05,
                    "y": 0.62,
                    "w": 0.28,
                    "h": 0.24,
                },
                "metadata": {"track_id": 4, "speed": 31.2},
                "sub_detections": {
                    "plate": {
                        "class_name": "plate",
                        "boundingbox": {
                            "x": 0.085,
                            "y": 0.79,
                            "w": 0.075,
                            "h": 0.038,
                        },
                        "metadata": {"text": "LJ 82-A31"},
                    },
                    "driver": {
                        "class_name": "driver",
                        "boundingbox": {
                            "x": 0.12,
                            "y": 0.66,
                            "w": 0.06,
                            "h": 0.05,
                        },
                    },
                },
            },
            # A distant car (a plain box).
            {
                "class_name": "car",
                "instance_id": 2,
                "boundingbox": {"x": 0.51, "y": 0.485, "w": 0.09, "h": 0.06},
                "metadata": {"track_id": 9},
            },
            # A pedestrian with a keypoint skeleton.
            {
                "class_name": "person",
                "instance_id": 3,
                "boundingbox": {"x": 0.40, "y": 0.355, "w": 0.09, "h": 0.43},
                "keypoints": {"keypoints": _RECORD_POSE},
            },
            # A traffic sign whose recognized text rides on the label chip.
            {
                "class_name": "sign",
                "instance_id": 4,
                "boundingbox": {"x": 0.33, "y": 0.33, "w": 0.06, "h": 0.085},
                "metadata": {"text": "STOP"},
            },
            # Class-only detections become image-level classification tags.
            {"class_name": "overcast"},
            {"class_name": "urban"},
        ],
    }

    record = DatasetRecord.model_validate(record_dict)
    config = VizConfig(
        skeletons={"scene": (_POSE_NAMES, _POSE_EDGES)},
        draw_skeletons=True,
        keypoint_label_mode="none",
    )
    img = visualize_record(record, street_scene(w, h), config=config)
    return save(img, "from_record.png")


def render_typography() -> Path:
    """Bundled fonts (Inter + JetBrains Mono) and inline ``<b>/<i>/<code>`` tags."""
    markup = Image(gradient(_W, _H, hue=0.62)).add(
        InfoCard(
            rows=[
                "The <b>quick</b> <i>brown</i> <code>fox()</code>",
                "id: <b>42</b>",
                "state: <i>occluded</i>",
                "file: <code>seq/img_0007.jpg</code>",
            ],
            title="<b>inline markup</b>",
            corner=Corner.TOP_LEFT,
        )
    )
    # Values render in JetBrains Mono; numbers align in the bar chart.
    numbers = Image(gradient(_W, _H, hue=0.05)).add(
        ClassDistribution(
            probabilities={
                "person": 1240.0,
                "car": 712.0,
                "bike": 143.0,
                "dog": 56.0,
            },
            value_format="count+percent",
            top_k=None,
        )
    )
    panel = (
        Image(gradient(300, _H, hue=0.58))
        .add(BBox(x=0.1, y=0.2, w=0.5, h=0.6, label="person", score=0.97))
        .with_panel(
            {"source": "img_0007.jpg", "frame": 7, "speed": 12.4},
            title="sample",
        )
    )
    grid_img = grid(
        [markup, numbers, panel],
        ncols=3,
        titles=["markup: bold / italic / mono", "mono numbers", "mono values"],
    )
    return save(grid_img, "typography.png")


# ---------------------------------------------------------------------------
# Augmentation figures — real dataset samples through each custom transform.
#
# These need the ``data`` extra and a native LDF dataset dump (default: the
# ``D2_ParkingLot_Native`` directory in the repo root). Each figure shows the
# same sample before and after one of the custom Albumentations transforms in
# ``luxonis_ml.data.augmentations.custom``, drawn with vizlab so boxes,
# keypoints, and masks are visible transforming together.
# ---------------------------------------------------------------------------

#: Canonical source of the demo dataset: the public GCS test bucket the parser
#: tests use (needs GCS credentials, e.g. ``GOOGLE_APPLICATION_CREDENTIALS``).
_AUG_DATASET_URL = (
    "gs://luxonis-test-bucket/luxonis-ml-test-data/D2_ParkingLot_Native.zip"
)
#: Optional local override — a native LDF dump directory (``annotations.json`` +
#: ``images/`` + ``masks/``). When present it is used directly (fast, offline);
#: otherwise the dataset is downloaded and parsed from ``_AUG_DATASET_URL``.
_AUG_DATASET_DIR = (
    Path(__file__).resolve().parent.parent / "D2_ParkingLot_Native"
)
#: A stable dataset name so the built/downloaded dataset is reused across runs.
_AUG_DATASET_NAME = "vizlab_augs_parkinglot"
#: A keypoint-free variant for the batch transforms (MixUp/Mosaic4): dropping
#: keypoints lets both the car and motorbike tasks be mixed together (their
#: differing keypoint counts otherwise clash), which looks better.
_AUG_BATCH_NAME = _AUG_DATASET_NAME + "_nokp"
#: Common display height so before/after cells line up in a strip.
_AUG_DISPLAY_H = 460
#: Class-label font multiplier for the augmentation figures (the base font is
#: small once the large samples are scaled down to the display height).
_AUG_LABEL_SCALE = 1.9


def _load_aug_dataset() -> "LuxonisDataset":
    """Return the augmentation-demo dataset, building or downloading it once.

    Preference order: a previously built dataset of the same name (instant), then
    a local ``D2_ParkingLot_Native`` dump directory (fast, offline), then a
    download-and-parse from the GCS test bucket (`_AUG_DATASET_URL`, the same
    source the parser tests use). The dataset persists under its name, so later
    runs reuse it.
    """
    from luxonis_ml.data import LuxonisDataset

    if LuxonisDataset.exists(_AUG_DATASET_NAME):
        dataset = LuxonisDataset(_AUG_DATASET_NAME)
        if len(dataset) > 0:
            return dataset

    if (_AUG_DATASET_DIR / "annotations.json").exists():
        return _build_aug_dataset_from_dir(_AUG_DATASET_DIR)
    return _download_aug_dataset()


def _download_aug_dataset() -> "LuxonisDataset":
    """Download and parse the demo dataset from the GCS test bucket."""
    import tempfile

    from luxonis_ml.data import LuxonisParser

    return LuxonisParser(
        _AUG_DATASET_URL,
        dataset_name=_AUG_DATASET_NAME,
        delete_local=True,
        save_dir=Path(tempfile.gettempdir()) / "vizlab_augs_download",
    ).parse()


def _build_aug_dataset_from_dir(
    root: Path, name: str = _AUG_DATASET_NAME, *, keypoints: bool = True
) -> "LuxonisDataset":
    """Build a ``LuxonisDataset`` from a local native LDF dump directory.

    The dump has ``annotations.json`` (a list of ``add()``-style records) plus
    ``images/`` and ``masks/``; relative paths are resolved against it. When
    ``keypoints`` is ``False`` the keypoint annotations are dropped, so tasks
    with different keypoint counts can be mixed together.
    """
    import json

    from luxonis_ml.data import LuxonisDataset

    records = json.loads((root / "annotations.json").read_text())

    def generator() -> Iterator[dict]:
        for raw in records:
            record = dict(raw)
            record["file"] = str((root / record["file"]).resolve())
            annotation = record.get("annotation")
            if isinstance(annotation, dict):
                annotation = dict(annotation)
                if not keypoints:
                    annotation.pop("keypoints", None)
                for mask_type in ("segmentation", "instance_segmentation"):
                    entry = annotation.get(mask_type)
                    if isinstance(entry, dict) and isinstance(
                        entry.get("mask"), str
                    ):
                        annotation[mask_type] = {
                            **entry,
                            "mask": str((root / entry["mask"]).resolve()),
                        }
                record["annotation"] = annotation
            yield record

    dataset = LuxonisDataset(name, delete_local=True).add(generator())
    dataset.make_splits({"train": 1.0})
    return dataset


def _batch_demo_dataset() -> tuple["LuxonisDataset", list[str]]:
    """Return the dataset and task names for the MixUp/Mosaic4 figures.

    Prefers a keypoint-free build (from the local dump) so the car and motorbike
    tasks can be mixed together — both classes then appear in the result. Without
    a local dump to strip keypoints from, it falls back to the full dataset and a
    single task so the batch transforms still run.
    """
    from luxonis_ml.data import LuxonisDataset

    if LuxonisDataset.exists(_AUG_BATCH_NAME):
        dataset = LuxonisDataset(_AUG_BATCH_NAME)
        if len(dataset) > 0:
            return dataset, ["car", "motorbike"]
    if (_AUG_DATASET_DIR / "annotations.json").exists():
        dataset = _build_aug_dataset_from_dir(
            _AUG_DATASET_DIR, _AUG_BATCH_NAME, keypoints=False
        )
        return dataset, ["car", "motorbike"]
    return _load_aug_dataset(), ["car"]


def _aug_viz_config(
    dataset: "LuxonisDataset", *, font_scale: float = 1.0
) -> "VizConfig":
    """Build a `VizConfig` sharing the dataset palette and keypoint skeletons.

    ``font_scale`` enlarges the class-label font; the samples are large and get
    scaled down to the display height, which otherwise leaves the labels small.
    """
    from luxonis_ml.data.loaders.label_converter import _BACKGROUND
    from luxonis_ml.vizlab import (
        Palette,
        Theme,
        VizConfig,
        get_default_theme,
    )

    class_names = [
        name
        for name in dict.fromkeys(
            n for names in dataset.get_class_names().values() for n in names
        )
        if name != _BACKGROUND
    ]
    theme = get_default_theme()
    if font_scale != 1.0:
        theme = Theme(
            style=theme.style.merge(
                font_size=theme.style.font_size * font_scale
            ),
            palette=theme.palette,
            background=theme.background,
        )
    return VizConfig(
        palette=Palette(class_names),
        skeletons=dataset.get_skeletons(),
        draw_skeletons=True,
        keypoint_label_mode="none",
        theme=theme,
    )


def _load_annotated(
    dataset: "LuxonisDataset",
    config: "VizConfig",
    *,
    augmentation: list[dict] | None = None,
    size: int | None = None,
    index: int = 0,
    seed: int = 0,
    tasks: list[str] | None = None,
    label: str | None = None,
) -> tuple[np.ndarray, list]:
    """Load one (optionally augmented) sample as an image plus its annotations.

    ``tasks`` restricts the loader to those task names (see `_batch_demo_dataset`
    for why the batch transforms use a keypoint-free, both-class dataset). When
    ``label`` is given it is written to each detection's ``text`` metadata, which
    vizlab renders on the chip — used to show each MixUp source's blend weight.
    """
    from luxonis_ml.data import LuxonisLoader
    from luxonis_ml.data.loaders.label_converter import (
        loader_output_to_records,
    )
    from luxonis_ml.vizlab.convert import blend_records_to_annotations

    loader = LuxonisLoader(
        dataset,
        view="train",
        height=size,
        width=size,
        keep_aspect_ratio=True,
        augmentation_config=augmentation or [],
        seed=seed,
        filter_task_names=tasks,
    )
    classes = dataset.get_classes()
    encodings = dataset.get_categorical_encodings()
    for i, data in enumerate(loader):
        if i < index:
            continue
        image = next(iter(data.images.values()))
        records = loader_output_to_records(
            data.labels, classes=classes, categorical_encodings=encodings
        )
        if label is not None:
            for record in records.values():
                for annotation in (
                    record.annotation
                    if isinstance(record.annotation, list)
                    else [record.annotation]
                ):
                    if annotation is not None:
                        annotation.metadata["text"] = label
        annotations = blend_records_to_annotations(records.values(), config)
        return image, annotations
    raise RuntimeError("dataset produced no samples")


def _render_aug_sample(
    dataset: "LuxonisDataset",
    config: "VizConfig",
    *,
    augmentation: list[dict] | None = None,
    size: int | None = None,
    index: int = 0,
    seed: int = 0,
    tasks: list[str] | None = None,
) -> Image:
    """Render one (optionally augmented) sample with its tasks blended in."""
    image, annotations = _load_annotated(
        dataset,
        config,
        augmentation=augmentation,
        size=size,
        index=index,
        seed=seed,
        tasks=tasks,
    )
    viz = Image(image, config=config)
    for annotation in annotations:
        viz.add(annotation)
    return viz


def _fit_height(image: Image, height: int) -> Image:
    """Render an image and scale it to a common display height (aspect kept).

    The annotations are already drawn, so the result is a plain raster tile —
    exactly what a before/after strip needs.
    """
    import cv2

    rendered = image.render()
    src_h, src_w = rendered.shape[:2]
    new_w = max(1, round(src_w * height / src_h))
    resized = cv2.resize(
        rendered[..., :3], (new_w, height), interpolation=cv2.INTER_AREA
    )
    return Image(resized)


def _aug_strip(before: Image, after: Image, title: str, name: str) -> Path:
    """Compose a before/after strip at a common height and save it."""
    return save(
        grid(
            [
                _fit_height(before, _AUG_DISPLAY_H),
                _fit_height(after, _AUG_DISPLAY_H),
            ],
            ncols=2,
            titles=["original", title],
        ),
        name,
    )


def render_aug_letterbox() -> Path:
    """LetterboxResize: aspect-preserving resize to a square, padding the rest."""
    dataset = _load_aug_dataset()
    config = _aug_viz_config(dataset, font_scale=_AUG_LABEL_SCALE)
    before = _render_aug_sample(dataset, config, index=0)
    after = _render_aug_sample(
        dataset,
        config,
        augmentation=[
            {
                "name": "LetterboxResize",
                "params": {"height": 512, "width": 512, "p": 1.0},
            }
        ],
        size=512,
        index=0,
    )
    return _aug_strip(
        before,
        after,
        "LetterboxResize",
        "aug_letterbox.png",
    )


#: Hardcoded MixUp weight for the figure (the first sample's contribution). The
#: two samples' chips show ``_MIXUP_ALPHA`` and ``1 - _MIXUP_ALPHA``.
_MIXUP_ALPHA = 0.65
#: A square random-resized crop (``ratio=1`` keeps the aspect, so no squish) that
#: zooms and shifts each MixUp source for variance, since the objects are
#: otherwise always dead-center.
_AUG_SAFE_CROP = {
    "name": "RandomResizedCrop",
    "params": {
        "size": (512, 512),
        "scale": (0.5, 0.85),
        "ratio": (1.0, 1.0),
        "p": 1.0,
    },
}


def _aug_examples(images: list[Image], title: str, name: str) -> Path:
    """Save two example outputs of a batch augmentation, side by side.

    Batch transforms have no meaningful "before" (they combine several samples),
    so the figure shows two independent results instead of an original/after
    pair.
    """
    return save(
        grid(
            [_fit_height(image, _AUG_DISPLAY_H) for image in images],
            ncols=len(images),
            titles=[title] * len(images),
        ),
        name,
    )


def _mixup_image(
    dataset: "LuxonisDataset",
    config: "VizConfig",
    tasks: list[str],
    *,
    index_a: int,
    index_b: int,
    seed_a: int,
    seed_b: int,
) -> Image:
    """Blend two samples with MixUp's formula, stamping each with its weight.

    Built from two real samples so each source's chip can show its blend weight
    (``_MIXUP_ALPHA`` and ``1 - _MIXUP_ALPHA``); the blend is MixUp's own
    ``alpha * a + (1 - alpha) * b``. A square random-resized crop shifts each
    source so the two objects don't just stack in the center.
    """
    import cv2

    crop = [_AUG_SAFE_CROP]
    image_a, labels_a = _load_annotated(
        dataset,
        config,
        augmentation=crop,
        size=640,
        index=index_a,
        seed=seed_a,
        tasks=tasks,
        label=f"α={_MIXUP_ALPHA:.2f}",  # noqa: RUF001
    )
    image_b, labels_b = _load_annotated(
        dataset,
        config,
        augmentation=crop,
        size=640,
        index=index_b,
        seed=seed_b,
        tasks=tasks,
        label=f"α={1 - _MIXUP_ALPHA:.2f}",  # noqa: RUF001
    )
    blended = cv2.addWeighted(
        image_a, _MIXUP_ALPHA, image_b, 1 - _MIXUP_ALPHA, 0.0
    )
    viz = Image(blended, config=config)
    for annotation in (*labels_a, *labels_b):
        viz.add(annotation)
    return viz


def render_aug_mixup() -> Path:
    """MixUp: two blends, each source's chip stamped with its blend weight."""
    dataset, tasks = _batch_demo_dataset()
    config = _aug_viz_config(dataset, font_scale=_AUG_LABEL_SCALE)
    examples = [
        _mixup_image(
            dataset, config, tasks, index_a=0, index_b=3, seed_a=1, seed_b=9
        ),
        _mixup_image(
            dataset, config, tasks, index_a=5, index_b=8, seed_a=4, seed_b=12
        ),
    ]
    return _aug_examples(examples, "MixUp", "aug_mixup.png")


def render_aug_mosaic() -> Path:
    """Mosaic4: two 2x2 compositions, each tiling four samples."""
    dataset, tasks = _batch_demo_dataset()
    config = _aug_viz_config(dataset, font_scale=_AUG_LABEL_SCALE)
    mosaic = {
        "name": "Mosaic4",
        "params": {"out_width": 640, "out_height": 640, "p": 1.0},
    }
    # Mosaic4 randomizes the mosaic center, so many seeds crop down to a single
    # tile; these two keep a balanced 2x2 with both classes visible.
    examples = [
        _render_aug_sample(
            dataset,
            config,
            augmentation=[mosaic],
            size=640,
            index=i,
            seed=s,
            tasks=tasks,
        )
        for i, s in ((0, 2), (2, 3))
    ]
    return _aug_examples(examples, "Mosaic4", "aug_mosaic.png")


# A synthetic, asymmetric person pose (the person's LEFT arm raised) as a
# front-facing figure, so anatomical left is at image-right. Keypoints are
# colored by body side; after a symmetric flip the side colors stay on the
# correct anatomical parts — that is the whole point of these transforms, and
# what naive index-preserving flips get wrong. Order matches ``_FLIP_PAIRS``.
_FLIP_POSE = [
    (0.50, 0.13),  # 0 nose
    (0.61, 0.27),  # 1 left shoulder
    (0.39, 0.27),  # 2 right shoulder
    (0.68, 0.16),  # 3 left elbow (raised)
    (0.33, 0.40),  # 4 right elbow
    (0.73, 0.06),  # 5 left wrist (raised)
    (0.30, 0.52),  # 6 right wrist
    (0.57, 0.54),  # 7 left hip
    (0.43, 0.54),  # 8 right hip
    (0.59, 0.72),  # 9 left knee
    (0.41, 0.72),  # 10 right knee
    (0.60, 0.90),  # 11 left ankle
    (0.40, 0.90),  # 12 right ankle
]
#: Left/right index pairs swapped after a symmetric flip; the nose is its own.
_FLIP_PAIRS = [(0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
_FLIP_SIDE = 360

# Index groups per body side and the (side-local) skeleton edges connecting them.
_FLIP_LEFT = [1, 3, 5, 7, 9, 11]
_FLIP_RIGHT = [2, 4, 6, 8, 10, 12]
_FLIP_CENTER = [0, 1, 2, 7, 8]  # nose + shoulders + hips (the torso)
_LIMB_EDGES = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5)]  # arm then leg
_TORSO_EDGES = [(0, 1), (0, 2), (1, 2), (3, 4), (1, 3), (2, 4)]
_LEFT_COLOR, _RIGHT_COLOR, _CENTER_COLOR = "#22d3ee", "#fb7185", "#94a3b8"


def _flip_backdrop(side: int) -> np.ndarray:
    """Build a gradient with a bright corner ``sun`` so the mirror is obvious."""
    image = gradient(side, side, hue=0.6).copy()
    ys, xs = np.ogrid[:side, :side]
    sun = (xs - 0.15 * side) ** 2 + (ys - 0.15 * side) ** 2 <= (
        0.07 * side
    ) ** 2
    image[sun] = (250, 214, 96)
    return image


def _pose_image(image: np.ndarray, pose: list[tuple[float, float]]) -> Image:
    """Draw the pose with keypoints colored by body side (left/right/torso)."""

    def side(
        indices: list[int], edges: list[tuple[int, int]], color: str
    ) -> Keypoints:
        points = [(pose[i][0], pose[i][1], 2) for i in indices]
        return Keypoints(keypoints=points, edges=edges, color=color)

    return (
        Image(image)
        .add(side(_FLIP_CENTER, _TORSO_EDGES, _CENTER_COLOR))
        .add(side(_FLIP_RIGHT, _LIMB_EDGES, _RIGHT_COLOR))
        .add(side(_FLIP_LEFT, _LIMB_EDGES, _LEFT_COLOR))
    )


def _render_flip(transform: object, title: str, name: str) -> Path:
    """Apply a symmetric-flip transform to the pose and show before/after.

    Left-side joints are cyan, right-side rose. After the flip the image mirrors
    (watch the sun) yet each color stays on the correct anatomical side — the
    left palm travels with the left arm rather than being mislabeled.
    """
    import albumentations as A

    backdrop = _flip_backdrop(_FLIP_SIDE)
    before = _pose_image(backdrop, _FLIP_POSE)

    points = [(x * _FLIP_SIDE, y * _FLIP_SIDE) for x, y in _FLIP_POSE]
    result = A.Compose(
        [transform],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )(image=backdrop, keypoints=points)
    out_image = result["image"]
    out_h, out_w = out_image.shape[:2]
    flipped = [
        (float(p[0]) / out_w, float(p[1]) / out_h) for p in result["keypoints"]
    ]
    after = _pose_image(out_image, flipped)

    return save(
        grid(
            [
                _fit_height(before, _AUG_DISPLAY_H),
                _fit_height(after, _AUG_DISPLAY_H),
            ],
            ncols=2,
            titles=["original", title],
        ),
        name,
    )


def render_aug_horizontal_flip() -> Path:
    """HorizontalSymmetricKeypointsFlip: mirror left<->right, re-indexing joints."""
    from luxonis_ml.data.augmentations.custom import (
        HorizontalSymmetricKeypointsFlip,
    )

    return _render_flip(
        HorizontalSymmetricKeypointsFlip(keypoint_pairs=_FLIP_PAIRS, p=1.0),
        "HorizontalSymmetricKeypointsFlip",
        "aug_horizontal_flip.png",
    )


def render_aug_vertical_flip() -> Path:
    """VerticalSymmetricKeypointsFlip: mirror top<->bottom, re-indexing joints."""
    from luxonis_ml.data.augmentations.custom import (
        VerticalSymmetricKeypointsFlip,
    )

    return _render_flip(
        VerticalSymmetricKeypointsFlip(keypoint_pairs=_FLIP_PAIRS, p=1.0),
        "VerticalSymmetricKeypointsFlip",
        "aug_vertical_flip.png",
    )


def render_aug_transpose() -> Path:
    """TransposeSymmetricKeypoints: swap axes, re-indexing symmetric joints."""
    from luxonis_ml.data.augmentations.custom import (
        TransposeSymmetricKeypoints,
    )

    return _render_flip(
        TransposeSymmetricKeypoints(keypoint_pairs=_FLIP_PAIRS, p=1.0),
        "TransposeSymmetricKeypoints",
        "aug_transpose.png",
    )


def render_augmentations() -> list[Path]:
    """Render every custom-augmentation figure; needs the ``data`` extra."""
    return [
        render_aug_letterbox(),
        render_aug_mixup(),
        render_aug_mosaic(),
        render_aug_horizontal_flip(),
        render_aug_vertical_flip(),
        render_aug_transpose(),
    ]


def main() -> None:
    """Render every example and print where each landed."""
    for path in (
        render_showcase(),
        render_from_record(),
        render_overview(),
        render_detection(),
        render_masks_keypoints(),
        render_overlays(),
        render_themes(),
        render_heatmap_themes(),
        render_distribution_modes(),
        render_compose(),
        render_panel(),
        render_typography(),
    ):
        print(f"wrote {path}")

    # Augmentation figures need the data extra plus the dataset (local dump or a
    # GCS download); skip gracefully with a hint when any is unavailable. GCS
    # auth surfaces as RuntimeError, a missing dump as FileNotFoundError.
    try:
        aug_paths = render_augmentations()
    except (ImportError, FileNotFoundError, RuntimeError) as error:
        print(f"[yellow]skipping augmentation figures: {error}[/yellow]")
    else:
        for path in aug_paths:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
