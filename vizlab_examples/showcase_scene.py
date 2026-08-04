"""One big, richly annotated scene exercising every vizlab feature.

Unlike ``gallery.py`` — which renders many small per-topic figures to disk —
this is a single, standalone, *interactive* scene: run it and a window opens
with everything the package can draw layered onto one synthetic street frame.
It is meant as a smoke test you can eyeball while developing the package.

What it puts on screen:

- a dense **semantic segmentation** ground map (road / sidewalk),
- a translucent **heatmap** saliency field,
- an **instance mask** car carrying a nested **bounding box** license plate
  (with an OCR ``payload``) and a nested driver box — children derive the
  parent's color,
- an **oriented** parked car and a small distant car, one wearing a custom
  per-annotation **style** (dashed stroke),
- a pedestrian **box** with a nested **keypoint skeleton** (named joints),
- image-level **chrome** stacked in every corner: **classification** tags, a
  **class-distribution** prediction chart, **captions** (one with rich
  ``<b>`` markup), an **info card**, and a **legend**,
- a **metadata side panel**, and
- **hover tooltips** on the spatial annotations (via the interactive viewer),
  tinted with each object's class color.

Run it::

    python vizlab_examples/showcase_scene.py            # interactive window
    python vizlab_examples/showcase_scene.py --light    # light theme
    python vizlab_examples/showcase_scene.py --static    # system image viewer
    python vizlab_examples/showcase_scene.py --save out.png

Hover the boxes and the car for tooltips; press ``q`` or ``Esc`` to quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from luxonis_ml.vizlab import (
    DARK_THEME,
    LIGHT_THEME,
    BBox,
    Caption,
    ClassDistribution,
    Classification,
    Corner,
    Frame,
    Heatmap,
    Image,
    InfoCard,
    Keypoints,
    Legend,
    Mask,
    RenderOptions,
    SemanticMask,
    Theme,
    Tooltip,
    with_panel,
)

SCENE_W, SCENE_H = 1280, 800
DEFAULT_OUT = Path(__file__).parent / "output" / "showcase_scene.png"

# Pinned class colors: deterministic, on-brand hues so the scene (and every
# tooltip tinted from the same palette) looks intentional rather than random.
CLASS_COLORS = {
    "car": "#4FA3FF",
    "person": "#FF6B6B",
    "plate": "#FFD166",
    "driver": "#06D6A0",
    "road": "#8892B0",
    "sidewalk": "#C3A6FF",
}

# A car silhouette (image-normalized polygon) for the instance mask.
CAR_SILHOUETTE = [
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

# A standing pedestrian pose (image-normalized) and its skeleton. COCO
# visibility: 2 = visible, 1 = occluded (the right wrist and ankle here).
POSE = [
    (0.415, 0.37, 2),
    (0.400, 0.42, 2),
    (0.435, 0.42, 2),
    (0.385, 0.49, 2),
    (0.450, 0.49, 2),
    (0.380, 0.55, 2),
    (0.455, 0.55, 1),
    (0.405, 0.56, 2),
    (0.430, 0.56, 2),
    (0.400, 0.65, 2),
    (0.435, 0.65, 2),
    (0.400, 0.75, 2),
    (0.435, 0.75, 1),
]
POSE_EDGES = [
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
POSE_NAMES = [
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


def street_scene(width: int, height: int) -> np.ndarray:
    """Paint a believable street backdrop with numpy — no external asset."""
    img = np.zeros((height, width, 3), dtype=np.float64)
    horizon = int(height * 0.5)

    ramp = np.linspace(0.0, 1.0, horizon)[:, None]
    sky = (
        np.array([96, 132, 194]) * (1 - ramp)
        + np.array([206, 214, 224]) * ramp
    )
    img[:horizon] = sky[:, None, :]

    road_ramp = np.linspace(0.0, 1.0, height - horizon)[:, None]
    road = (
        np.array([68, 70, 78]) * (1 - road_ramp)
        + np.array([98, 100, 108]) * road_ramp
    )
    img[horizon:] = road[:, None, :]

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

    cx = width // 2
    for y in range(horizon + 6, height, 26):
        t = (y - horizon) / (height - horizon)
        half_w = max(1, int(2 + t * 6))
        dash = int(6 + t * 16)
        img[y : y + dash, cx - half_w : cx + half_w] = [212, 206, 178]

    return np.clip(img, 0, 255).astype(np.uint8)


def ground_segmentation(width: int, height: int) -> SemanticMask:
    """Build a dense label map: drivable road plus a left-side sidewalk."""
    labels = np.zeros((height, width), dtype=np.int32)
    horizon = int(height * 0.5)
    ys = np.arange(height)[:, None]
    xs = np.arange(width)[None, :]
    labels[horizon:] = 1
    edge = (0.30 - (ys / height) * 0.12) * width
    labels[(ys > horizon) & (xs < edge)] = 2
    return SemanticMask(
        labels=labels,
        names={0: "background", 1: "road", 2: "sidewalk"},
        ignore_index=0,
        fill_alpha=0.3,
    )


def saliency_field(width: int, height: int) -> np.ndarray:
    """Sum of Gaussian bumps — a smooth field standing in for a saliency map."""
    ys = np.linspace(0.0, 1.0, height)[:, None]
    xs = np.linspace(0.0, 1.0, width)[None, :]
    field = np.zeros((height, width), dtype=np.float64)
    for cx, cy, sigma in [(0.41, 0.55, 0.12), (0.21, 0.78, 0.09)]:
        field += np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)))
    return field


def build_theme(*, light: bool) -> Theme:
    """Build the scene's theme, with class colors pinned for a stable look."""
    base = LIGHT_THEME if light else DARK_THEME
    return base.with_class_colors(CLASS_COLORS)


def build_scene(options: RenderOptions) -> Frame:
    """Assemble the full scene, tooltips and all, onto one street frame.

    Returns a `Frame` — the paneled image plus its hover `HitMap`. The map is
    captured *before* the metadata panel is attached (via `Image.frame`); the
    panel sits to the right and the base blits at ``(0, 0)``, so those hit
    regions stay valid on the wider framed image (see `Frame.with_image`).
    """
    palette = options.theme.palette

    def tip(title: str, cls: str | None, **rows: object) -> Tooltip:
        """Return a hover tooltip tinted with ``cls``'s palette color."""
        return Tooltip(
            title=title,
            rows=tuple((k, str(v)) for k, v in rows.items()),
            tint=palette.color_for(cls) if cls is not None else None,
        )

    img = Image(street_scene(SCENE_W, SCENE_H), options=options)

    # Dense fills first so vector labels land on top of them.
    img.add(ground_segmentation(SCENE_W, SCENE_H))
    img.add(Heatmap(values=saliency_field(SCENE_W, SCENE_H), alpha=0.55))

    # Foreground car: an instance mask with a nested plate (OCR payload) and
    # driver; both children derive the car's color.
    car = Mask(
        points=CAR_SILHOUETTE,
        width=SCENE_W,
        height=SCENE_H,
        label="car",
        score=0.98,
    )
    car.tooltip = tip(
        "car #1", "car", instance_id=1, score=0.98, speed="34 km/h"
    )
    plate = BBox(
        x=0.285, y=0.79, w=0.055, h=0.032, label="plate", payload="5A2 8391"
    )
    plate.tooltip = tip("plate", "plate", text="5A2 8391", region="EU")
    car.add(plate)
    driver = BBox(x=0.165, y=0.635, w=0.05, h=0.045, label="driver", score=0.9)
    driver.tooltip = tip("driver", "driver", score=0.9, belt=True)
    car.add(driver)
    img.add(car)

    # A parked car at an angle (oriented box) with a custom dashed style, and a
    # small distant car with a low score.
    parked = BBox(
        x=0.58, y=0.60, w=0.28, h=0.16, angle=-12, label="car", score=0.9
    ).styled(dash=(10.0, 6.0), stroke_width=4.0)
    parked.tooltip = tip(
        "car #2", "car", instance_id=2, score=0.9, state="parked"
    )
    img.add(parked)
    distant = BBox(x=0.455, y=0.485, w=0.07, h=0.05, label="car", score=0.68)
    distant.tooltip = tip("car #3", "car", instance_id=3, score=0.68)
    img.add(distant)

    # A pedestrian: a box with its named pose skeleton nested inside it.
    person = BBox(x=0.37, y=0.34, w=0.11, h=0.44, label="person", score=0.95)
    person.tooltip = tip("person #1", "person", instance_id=1, score=0.95)
    # Numbered joints (not names): left/right joints sit only ~45px apart, so
    # the wider name chips would collide — indices stay legible.
    visible = sum(1 for *_, v in POSE if v == 2)
    pose = Keypoints(
        keypoints=POSE,
        edges=POSE_EDGES,
        keypoint_names=POSE_NAMES,
        point_labels="numbers",
        label="pose",
    )
    # Keypoints are hoverable too — the skeleton carries its own tooltip.
    pose.tooltip = tip(
        "pose",
        "person",
        joints=len(POSE),
        visible=visible,
        occluded=len(POSE) - visible,
    )
    person.add(pose)
    img.add(person)

    _add_chrome(img)

    # Capture the hover map now, while every tooltip-bearing annotation is on
    # the bare image; then attach the panel and carry the map over unchanged.
    frame = img.frame()

    metadata = {
        "source": "seq_000e/frame_000123.jpg",
        "split": "train",
        "resolution": f"{SCENE_W}x{SCENE_H}",
        "camera": "OAK-D Pro",
        "weather": "clear",
        "objects": {"car": 3, "person": 1},
        "augmentations": ["horizontal_flip", "color_jitter"],
    }
    return frame.with_image(img.with_panel(metadata, title="Sample metadata"))


def _add_chrome(img: Image) -> None:
    """Stack image-level overlays into all four corners."""
    # Top-left: scene tags, then a prediction distribution beneath them —
    # two CornerStacks sharing a corner stack instead of colliding.
    img.add(
        Classification(
            tags=[("daytime", 0.99), ("urban", 0.94), ("clear", 0.88)],
            corner=Corner.TOP_LEFT,
        )
    )
    img.add(
        ClassDistribution(
            probabilities={
                "sedan": 0.61,
                "suv": 0.22,
                "van": 0.10,
                "truck": 0.07,
            },
            ground_truth="suv",
            title="vehicle type",
            corner=Corner.TOP_LEFT,
        )
    )
    # Top-right: a titled caption (rich markup) and an info card beneath it.
    img.add(
        Caption(
            text="Annotated <b>showcase</b> scene",
            title=True,
            corner=Corner.TOP_RIGHT,
        )
    )
    img.add(
        InfoCard(
            rows=["cars: 3", "people: 1", "IoU the: 0.50"],
            title="counts",
            corner=Corner.TOP_RIGHT,
        )
    )
    # Bottom corners: the source filename and a class legend.
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


def _view_interactive(frame: Frame) -> None:
    """Open the scene in the interactive viewer with hover tooltips."""
    try:
        from luxonis_ml.vizlab.viewer import Viewer

        viewer = Viewer()
        name = "vizlab showcase"
        viewer.show(name, frame)
        print("Hover the boxes/car for tooltips. Press 'q' or Esc to quit.")
        try:
            while viewer.wait() not in ("q", "\x1b"):
                pass
        finally:
            viewer.close()
    except Exception as exc:  # noqa: BLE001 - headless / no cv2 fallback
        path = DEFAULT_OUT
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.image.save(path)
        print(
            f"Interactive viewer unavailable ({exc}); saved {path} instead. "
            f"Use --static for the system image viewer, or --save PATH."
        )


def main() -> None:
    """Build the scene and display (or save) it per the CLI flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        nargs="?",
        const=str(DEFAULT_OUT),
        help="save a PNG instead of opening a window (default path if bare)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="open in the system image viewer (Pillow) — no hover",
    )
    parser.add_argument(
        "--light", action="store_true", help="render with the light theme"
    )
    args = parser.parse_args()

    options = RenderOptions(theme=build_theme(light=args.light))
    frame = build_scene(options)

    if args.save is not None:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.image.save(path)
        print(f"Saved {path}")
    elif args.static:
        frame.image.show()
    else:
        _view_interactive(frame)


if __name__ == "__main__":
    main()
