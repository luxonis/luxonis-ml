"""vizlab gallery — one runnable script covering every feature.

Renders a set of ONGs into ``vizlab_examples/output/``: a grid with one cell per
label type, plus standalone images for compositing and the metadata panel. It
synthesizes its own backdrops with numpy, so it needs no external assets.

Run it from a checkout with the ``viz`` extra installed::

    python vizlab_examples/gallery.py

All spatial coordinates are image-normalized in ``[0, 1]`` (the Luxonis Data
Format convention): a box is ``x, y`` (top-left) plus ``w, h``; a keypoint is
``(x, y, visibility)`` with COCO visibility ``0``/``1``/``2``.
"""

from pathlib import Path

import numpy as np
from rich import print

from luxonis_ml.vizlab import (
    LIGHT_THEME,
    BBox,
    Caption,
    Classification,
    Corner,
    Image,
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


def _light_theme() -> Image:
    # DARK_THEME is the default; pass LIGHT_THEME (or your own) for a light look.
    return (
        Image(np.full((_H, _W, 3), 236, np.uint8), theme=LIGHT_THEME)
        .add(BBox(x=0.08, y=0.16, w=0.5, h=0.68, label="person", score=0.97))
        .add(BBox(x=0.44, y=0.36, w=0.47, h=0.5, label="dog", score=0.86))
    )


def render_gallery() -> Path:
    """Render the grid with one cell per feature."""
    cells = {
        "bounding boxes": _boxes(),
        "oriented boxes": _oriented(),
        "payload (OCR)": _payload(),
        "keypoints": _keypoints(),
        "instance mask": _instance_mask(),
        "polygon mask": _polygon_mask(),
        "semantic mask": _semantic(),
        "nested sub-labels": _nested(),
        "classification": _classification(),
        "captions + legend": _captions_legend(),
        "light theme": _light_theme(),
    }
    return save(
        grid(list(cells.values()), ncols=3, titles=list(cells)),
        "gallery.png",
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


def main() -> None:
    """Render every example and print where each landed."""
    for path in (render_gallery(), render_compose(), render_panel()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
