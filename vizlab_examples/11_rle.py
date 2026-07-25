"""COCO RLE mask input.

Detectors and COCO annotations often store instance masks as run-length encoding
(RLE) rather than dense arrays. ``Mask(rle=...)`` accepts a COCO RLE dict directly:
uncompressed ``counts`` (a list of run lengths) is decoded with numpy; compressed
``counts`` needs the optional ``pycocotools`` (the ``rle`` extra).
"""

import numpy as np
from _common import gradient, save

from luxonis_ml.vizlab import Image, Mask


def _uncompressed_rle(binary: np.ndarray) -> dict:
    """Encode a binary mask as an uncompressed COCO RLE dict."""
    flat = binary.flatten(order="F")  # COCO RLE is column-major
    counts: list[int] = []
    prev, run = 0, 0
    for value in flat:
        if int(value) == prev:
            run += 1
        else:
            counts.append(run)
            prev, run = int(value), 1
    counts.append(run)
    return {"size": list(binary.shape), "counts": counts}


def main() -> None:
    """Render two instances supplied as COCO RLE masks."""
    width, height = 620, 360
    ys, xs = np.ogrid[:height, :width]

    ring = ((xs - 200) ** 2 + (ys - 180) ** 2 <= 120**2) & (
        (xs - 200) ** 2 + (ys - 180) ** 2 >= 70**2
    )
    blob = (xs - 450) ** 2 + (ys - 180) ** 2 <= 110**2

    img = Image(gradient(width, height, hue=0.72))
    img.add(Mask(rle=_uncompressed_rle(ring.astype(np.uint8)), label="ring"))
    img.add(Mask(rle=_uncompressed_rle(blob.astype(np.uint8)), label="blob"))
    save(img, "11_rle.png")


if __name__ == "__main__":
    main()
