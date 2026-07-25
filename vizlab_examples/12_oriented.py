"""Oriented (rotated) bounding boxes.

Any ``BBox`` can be rotated: pass an ``angle`` (about the box center) with a rect
format, a ``cxcywha`` box that carries its own angle, or four explicit corner points
via ``xyxyxyxy``. Common in aerial imagery, scene text, and rotated-object detectors.
Label chips are placed against each box's axis-aligned bounds, so they stay clear of
each other just like upright boxes.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Image


def main() -> None:
    """Render several oriented boxes at different angles."""
    img = Image(gradient(680, 440, hue=0.6))
    # angle= rotates a rect-format box about its center.
    img.add(
        BBox(
            (160, 150, 200, 90),
            format="cxcywh",
            angle=30,
            label="ship",
            score=0.94,
        )
    )
    img.add(
        BBox(
            (470, 140, 150, 150),
            format="cxcywh",
            angle=-20,
            label="roof",
            score=0.88,
        )
    )
    # cxcywha carries the angle in the coords (as a rotated detector emits).
    img.add(
        BBox(
            (250, 340, 260, 70, 8),
            format="cxcywha",
            label="runway",
            score=0.79,
        )
    )
    # Four explicit corner points (a slightly irregular quad).
    img.add(
        BBox(
            (470, 300, 620, 320, 600, 410, 450, 380),
            format="xyxyxyxy",
            label="lot",
        )
    )
    save(img, "12_oriented.png")


if __name__ == "__main__":
    main()
