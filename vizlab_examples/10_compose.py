"""Compositing: blend (mixup), hstack with titles, and a grid.

Each composition renders its inputs at native resolution and returns a new image,
so the originals are untouched. Titles are drawn above each cell.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Image, blend, grid, hstack


def _scene(hue: float, label: str, score: float) -> Image:
    """A small labeled scene to compose."""
    return Image(gradient(300, 220, hue=hue)).add(
        BBox((40, 40, 220, 150), label=label, score=score)
    )


def main() -> None:
    """Render a blend, a titled row, and a grid."""
    cat = _scene(0.58, "cat", 0.96)
    dog = _scene(0.08, "dog", 0.91)

    # blend / mixup: the two scenes averaged.
    mixed = blend(cat, dog, alpha=0.4)
    save(mixed, "10_blend.png")

    # side by side with per-cell titles.
    row = hstack([cat, dog, mixed], titles=["cat", "dog", "mixup"])
    save(row, "10_hstack.png")

    # a grid of four scenes.
    scenes = [
        _scene(0.0, "person", 0.99),
        _scene(0.3, "bike", 0.84),
        _scene(0.6, "car", 0.77),
        _scene(0.85, "sign", 0.63),
    ]
    save(
        grid(scenes, ncols=2, titles=["person", "bike", "car", "sign"]),
        "10_grid.png",
    )


if __name__ == "__main__":
    main()
