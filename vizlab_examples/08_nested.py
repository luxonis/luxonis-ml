"""Nested sub-labels with parent-derived styling.

A ``car`` box carries a ``driver`` box, which carries a ``phone`` box. None of the
children set a color or style: each derives from its parent — lighter and slightly
hue-shifted, with a thinner *dashed* stroke — so the nesting reads at a glance.
Nested masks derive the same way; the truck carries a dashed ``cargo`` mask. Compare
with the truck itself, an independent top-level box with its own palette color.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Image, Mask

CARGO = [(480, 190), (650, 200), (640, 340), (500, 330)]


def main() -> None:
    """Render a three-level nested box next to an independent one."""
    img = Image(gradient(720, 460, hue=0.55))

    car = BBox((40, 90, 340, 310), label="car", score=0.99)
    car.add(
        BBox((150, 150, 180, 210), label="driver", score=0.94).add(
            BBox((250, 250, 70, 70), label="phone", score=0.71)
        )
    )
    img.add(car)

    truck = BBox((440, 120, 240, 260), label="truck", score=0.88)
    truck.add(Mask(polygon=CARGO, label="cargo"))
    img.add(truck)
    save(img, "08_nested.png")


if __name__ == "__main__":
    main()
