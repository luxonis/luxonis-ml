"""Render a single gallery image covering every label type and composition.

A visual smoke test: one grid cell per feature — boxes, keypoints, instance and
semantic masks, nested sub-labels, and classification tags — plus a light-theme
cell to show theming. Run with ``python examples/gallery.py``.
"""

import numpy as np
from _common import gradient, save

from luxonis_ml.vizlab import (
    LIGHT_THEME,
    BBox,
    Classification,
    Corner,
    Image,
    Keypoints,
    Legend,
    Mask,
    SemanticMask,
    Skeleton,
    grid,
)

_W, _H = 340, 250


def _boxes() -> Image:
    img = Image(gradient(_W, _H, hue=0.58))
    img.add(BBox((30, 40, 170, 170), label="person", score=0.97))
    img.add(BBox((150, 90, 160, 130), label="dog", score=0.86))
    return img


def _keypoints() -> Image:
    pts = np.array(
        [
            [170, 60, 1.0],
            [150, 120, 0.9],
            [190, 120, 0.9],
            [130, 180, 0.8],
            [210, 180, 0.8],
        ],
        dtype=float,
    )
    skeleton = Skeleton(edges=((0, 1), (0, 2), (1, 3), (2, 4)))
    return Image(gradient(_W, _H, hue=0.68)).add(
        Keypoints(pts, skeleton=skeleton, label="pose")
    )


def _instance_mask() -> Image:
    ys, xs = np.ogrid[:_H, :_W]
    disc = (xs - 170) ** 2 + (ys - 125) ** 2 <= 95**2
    return Image(gradient(_W, _H, hue=0.4)).add(Mask(mask=disc, label="moon"))


def _semantic() -> Image:
    labels = np.zeros((_H, _W), dtype=np.int32)
    labels[: int(_H * 0.55)] = 1
    labels[int(_H * 0.55) :] = 2
    labels[120:200, 210:300] = 3
    names = {0: "background", 1: "sky", 2: "ground", 3: "car"}
    return Image(gradient(_W, _H, hue=0.5)).add(
        SemanticMask(labels, names=names, ignore_index=0)
    )


def _nested() -> Image:
    car = BBox((30, 40, 230, 180), label="car", score=0.98)
    car.add(BBox((110, 100, 130, 100), label="driver", score=0.9))
    return Image(gradient(_W, _H, hue=0.55)).add(car)


def _tags() -> Image:
    img = Image(gradient(_W, _H, hue=0.12))
    # Order-independent: box labels avoid the corner tags, which draw on top.
    img.add(BBox((90, 70, 190, 140), label="beach", score=0.8))
    img.add(
        Classification(
            tags=[("outdoor", 0.98), ("sunny", 0.7)], corner=Corner.TOP_LEFT
        )
    )
    return img


def _light_theme() -> Image:
    img = Image(np.full((_H, _W, 3), 236, np.uint8), theme=LIGHT_THEME)
    img.add(BBox((30, 40, 170, 170), label="person", score=0.97))
    img.add(BBox((150, 90, 160, 130), label="dog", score=0.86))
    return img


def _oriented() -> Image:
    img = Image(gradient(_W, _H, hue=0.62))
    img.add(
        BBox(
            (120, 110, 150, 70),
            format="cxcywh",
            angle=28,
            label="ship",
            score=0.93,
        )
    )
    img.add(
        BBox(
            (230, 180, 120, 120),
            format="cxcywh",
            angle=-18,
            label="roof",
            score=0.85,
        )
    )
    return img


def _legend() -> Image:
    img = Image(gradient(_W, _H, hue=0.05))
    img.add(BBox((30, 40, 150, 170), label="car", score=0.96))
    img.add(BBox((150, 90, 150, 130), label="truck", score=0.88))
    img.add(
        Legend(
            entries=["car", "truck", "road"],
            title="classes",
            corner=Corner.BOTTOM_RIGHT,
        )
    )
    return img


def main() -> None:
    """Render the full gallery grid to the output directory."""
    cells = [
        _boxes(),
        _keypoints(),
        _instance_mask(),
        _semantic(),
        _nested(),
        _oriented(),
        _tags(),
        _legend(),
        _light_theme(),
    ]
    titles = [
        "bounding boxes",
        "keypoints",
        "instance mask",
        "semantic mask",
        "nested sub-labels",
        "oriented boxes",
        "classification",
        "legend",
        "light theme",
    ]
    save(grid(cells, ncols=3, titles=titles), "gallery.png")


if __name__ == "__main__":
    main()
