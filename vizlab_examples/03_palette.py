"""Palette determinism: the same class name always gets the same color.

The boxes are laid out on a grid and colored purely from their class names — no
color is passed in. Re-running the script (or rendering these classes in any other
image) produces the identical color for each class, which is what keeps a batch of
visualizations readable together.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Image

CLASSES = [
    "car",
    "truck",
    "bus",
    "person",
    "bicycle",
    "dog",
    "cat",
    "traffic light",
    "stop sign",
    "backpack",
    "umbrella",
    "bench",
]


def main() -> None:
    """Render a swatch box per class, colored from the class name alone."""
    cols, rows = 4, 3
    cell_w, cell_h, pad = 150, 90, 20
    width = cols * cell_w + (cols + 1) * pad
    height = rows * cell_h + (rows + 1) * pad
    img = Image(gradient(width, height, hue=0.7))

    for i, name in enumerate(CLASSES):
        col, row = i % cols, i // cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        img.add(
            BBox((x, y, x + cell_w, y + cell_h), format="xyxy", label=name)
        )

    save(img, "03_palette.png")


if __name__ == "__main__":
    main()
