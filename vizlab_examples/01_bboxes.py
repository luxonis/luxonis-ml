"""Basic object-detection scene: a handful of labeled, scored boxes.

Run with ``python examples/01_bboxes.py`` (inside the dev shell). Colors are
assigned automatically from each class name; the same class always gets the same
color.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Image


def main() -> None:
    """Render a few labeled boxes onto a synthetic backdrop."""
    img = Image(gradient(720, 460))
    img.add(BBox((40, 60, 260, 320), label="person", score=0.98))
    img.add(BBox((260, 180, 260, 220), label="dog", score=0.87))
    img.add(BBox((470, 40, 220, 210), label="frisbee", score=0.63))
    img.add(BBox((150, 30, 100, 90), label="hat", score=0.55))
    save(img, "01_bboxes.png")


if __name__ == "__main__":
    main()
