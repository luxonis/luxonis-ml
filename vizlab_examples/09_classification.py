"""Image-level classification chips in the corners.

``Classification`` stacks class-tag chips in a chosen corner. Use it for
whole-image predictions (multi-label here, with scores) — each chip is colored by
its class name, matching how the same class would look on a box.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Classification, Corner, Image


def main() -> None:
    """Render multi-label tags in a corner alongside a detection box."""
    img = Image(gradient(640, 420, hue=0.62))
    img.add(BBox((220, 120, 250, 240), label="beach", score=0.81))
    img.add(
        Classification(
            tags=[("outdoor", 0.98), ("beach", 0.81), ("sunny", 0.66)],
            corner=Corner.TOP_LEFT,
        )
    )
    img.add(
        Classification(tags=[("daytime", 0.9)], corner=Corner.BOTTOM_RIGHT)
    )
    save(img, "09_classification.png")


if __name__ == "__main__":
    main()
