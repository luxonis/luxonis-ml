"""Coordinate-format resolution.

vizlab accepts boxes in several conventions — ``xyxy``, ``xywh``, ``cxcywh`` — in
either pixel or normalized units, and resolves them all to pixels at render time.
This example draws four boxes, each describing its rectangle in a different format;
they all land exactly where their numbers say.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Image


def main() -> None:
    """Render one box per coordinate format on a single image."""
    img = Image(gradient(640, 480, hue=0.3))

    # xyxy: left, top, right, bottom (pixels).
    img.add(
        BBox((40, 40, 280, 220), format="xyxy", label="xyxy", color="#4c8dff")
    )
    # xywh: left, top, width, height (pixels).
    img.add(
        BBox((360, 40, 240, 180), format="xywh", label="xywh", color="#33c58a")
    )
    # cxcywh: center-x, center-y, width, height (pixels).
    img.add(
        BBox(
            (170, 360, 240, 180),
            format="cxcywh",
            label="cxcywh",
            color="#ffa94d",
        )
    )
    # normalized xyxy: fractions of the image size, auto-detected (all <= 1).
    img.add(
        BBox(
            (0.58, 0.6, 0.94, 0.95),
            format="xyxy",
            label="normalized",
            color="#b085ff",
        )
    )

    save(img, "02_formats.png")


if __name__ == "__main__":
    main()
