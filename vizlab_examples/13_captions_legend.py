"""Image-level captions and a class-color legend.

Captions are short text cards in a corner (a plain caption plus a bolder title
here); a Legend is a class-color key. All are overlays: drawn on top of the boxes
and reserved so box labels route around them.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Caption, Corner, Image, Legend


def main() -> None:
    """Render detections with a filename caption, a title, and a legend."""
    img = Image(gradient(640, 420, hue=0.58))
    img.add(BBox((60, 90, 240, 270), label="car", score=0.95))
    img.add(BBox((330, 130, 230, 210), label="truck", score=0.88))

    img.add(Caption(text="frame_0421.jpg", corner=Corner.TOP_LEFT))
    img.add(Caption(text="Detections", title=True, corner=Corner.TOP_RIGHT))
    img.add(
        Legend(
            entries=["car", "truck", ("road", "#5566aa")],
            title="classes",
            corner=Corner.BOTTOM_RIGHT,
        )
    )
    save(img, "13_captions_legend.png")


if __name__ == "__main__":
    main()
