"""Semantic segmentation: a dense label map colored per class.

``SemanticMask`` takes an ``(H, W)`` integer label map and colors each class id
from the palette (using the class names, so colors stay stable). Class 0 is the
background here and is left undrawn.
"""

import numpy as np
from _common import gradient, save

from luxonis_ml.vizlab import Image, SemanticMask

NAMES = {0: "background", 1: "sky", 2: "road", 3: "car", 4: "tree"}


def _label_map(width: int, height: int) -> np.ndarray:
    """Build a small synthetic scene as an integer label map."""
    labels = np.zeros((height, width), dtype=np.int32)
    horizon = int(height * 0.55)
    labels[:horizon] = 1  # sky
    labels[horizon:] = 2  # road
    labels[horizon - 90 : horizon + 10, 90:150] = 4  # tree trunk region
    ys, xs = np.ogrid[:height, :width]
    labels[(xs - 110) ** 2 + (ys - (horizon - 120)) ** 2 <= 70**2] = (
        4  # tree canopy
    )
    labels[horizon - 10 : horizon + 70, 360:560] = 3  # car
    return labels


def main() -> None:
    """Render a semantic label map over a backdrop."""
    width, height = 640, 400
    img = Image(gradient(width, height, hue=0.5))
    img.add(
        SemanticMask(_label_map(width, height), names=NAMES, ignore_index=0)
    )
    save(img, "07_semantic.png")


if __name__ == "__main__":
    main()
