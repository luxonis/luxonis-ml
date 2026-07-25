"""Instance masks: a polygon segmentation and a binary-array mask.

Two ways to give a mask: a polygon (vector, so the edge is crisp and anti-aliased)
and a binary ``(H, W)`` array (its contour is traced with OpenCV when installed).
Both draw a translucent fill plus an outline in the instance's color.
"""

import numpy as np
from _common import gradient, save

from luxonis_ml.vizlab import Image, Mask

# A blobby polygon standing in for a segmentation outline.
LEAF = [
    (90, 120),
    (150, 70),
    (230, 60),
    (300, 95),
    (330, 170),
    (300, 250),
    (220, 300),
    (140, 285),
    (95, 220),
    (80, 170),
]


def _disc(
    width: int, height: int, cx: int, cy: int, radius: int
) -> np.ndarray:
    """Return an ``(H, W)`` boolean mask with a filled disc."""
    ys, xs = np.ogrid[:height, :width]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2


def main() -> None:
    """Render one polygon mask and one binary-array mask on a single image."""
    width, height = 640, 380
    img = Image(gradient(width, height, hue=0.4))
    img.add(Mask(polygon=LEAF, label="leaf"))
    img.add(Mask(mask=_disc(width, height, 480, 200, 120), label="moon"))
    save(img, "06_masks.png")


if __name__ == "__main__":
    main()
