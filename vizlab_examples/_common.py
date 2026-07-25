"""Shared helpers for the example scripts.

These keep the examples self-contained: they synthesize backdrop images with numpy
so the gallery runs without any external image assets, and resolve a common output
directory next to this file.
"""

from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"


def gradient(
    width: int = 640, height: int = 400, *, hue: float = 0.58
) -> np.ndarray:
    """Build a smooth diagonal gradient image to draw annotations on.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        hue: Base hue in ``[0, 1]`` controlling the overall tint.

    Returns:
        An ``(H, W, 3)`` ``uint8`` RGB array.

    """
    ys = np.linspace(0.0, 1.0, height)[:, None]
    xs = np.linspace(0.0, 1.0, width)[None, :]
    diag = (xs + ys) / 2.0
    base = 30 + diag * 70
    r = base * (0.7 + 0.3 * hue)
    g = base * 0.9
    b = base * (0.9 + 0.4 * (1.0 - hue))
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save(img: object, name: str) -> Path:
    """Render a `vizlab.Image` to the output directory and report the path.

    Args:
        img: A `vizlab.Image` or ``PIL.Image`` to write.
        name: Output file name, e.g. ``"boxes.png"``.

    Returns:
        The path the image was written to.

    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    img.save(path)  # type: ignore[attr-defined]
    return path
