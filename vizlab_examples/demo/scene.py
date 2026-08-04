"""The demo's backdrops — synthetic frames with something actually in them.

An annotation renderer has to be judged against image content: a box drawn
around empty asphalt demonstrates a rectangle, which nobody doubts. So the
street here carries a car and a pedestrian, and :data:`CAR` and :data:`PERSON`
give their exact normalized boxes — every slide that annotates this frame uses
those constants, so the boxes land on the objects rather than near them.

`vizlab_examples.gallery` has a street of its own, and a dozen figures are
composed against its coordinates; this one is separate so the demo can put
actors wherever its slides need them without moving anything in the docs.

Everything is drawn with `Canvas` — the same Skia surface the annotations use —
so the edges are antialiased and the whole frame, backdrop included, is a vizlab
render.
"""

from collections.abc import Callable

import numpy as np

from luxonis_ml.vizlab import Color, Rect
from luxonis_ml.vizlab.render.canvas import Canvas

#: Where the horizon sits, as a fraction of frame height, and where the road
#: converges on it.
HORIZON, VANISHING_X = 0.44, 0.52

#: The two actors' boxes, normalized. Slides annotate the frame through these,
#: so a box is always drawn around the thing it names.
CAR = {"x": 0.52, "y": 0.44, "w": 0.30, "h": 0.26}
PERSON = {"x": 0.16, "y": 0.45, "w": 0.09, "h": 0.31}
#: A sign on the verge, standing between the other two — so an `Arrow` drawn
#: from one to the other has something real to bow around.
SIGN = {"x": 0.30, "y": 0.36, "w": 0.05, "h": 0.26}

#: Class ids the label map paints, and the names that decode them. 0 is left
#: unlabelled — the verges either side of the road — so `ignore_index` has
#: something to visibly drop out.
CLASSES = {
    0: "void",
    1: "sky",
    2: "building",
    3: "road",
    4: "car",
    5: "person",
    6: "sign",
}

_SKY_TOP, _SKY_HAZE = (68, 104, 166), (196, 208, 226)
_ROAD_FAR, _ROAD_NEAR = (72, 74, 82), (106, 108, 118)
_VERGE = Color(59, 68, 62)
_KERB = Color(96, 100, 104)
_BUILDING = Color(58, 62, 76)
_WINDOW = Color(120, 132, 152)
_PAINT = Color(226, 226, 214)
_SHADOW = Color(0, 0, 0, 70)
_CAR_BODY, _CAR_DARK = Color(147, 65, 74), Color(104, 44, 52)
_GLASS, _TYRE = Color(28, 39, 51), Color(30, 32, 38)
_JACKET, _JEANS, _HAIR = (
    Color(61, 90, 99),
    Color(47, 59, 82),
    Color(58, 50, 46),
)

#: The skyline: ``(x, width, height)`` as fractions, leaving the road's
#: vanishing point clear.
_BUILDINGS = [
    (0.00, 0.13, 0.74),
    (0.11, 0.10, 0.50),
    (0.20, 0.11, 0.92),
    (0.30, 0.08, 0.44),
    (0.58, 0.10, 0.62),
    (0.67, 0.13, 0.96),
    (0.79, 0.09, 0.54),
    (0.87, 0.15, 0.78),
]


def _sky_and_asphalt(width: int, height: int, horizon: int) -> np.ndarray:
    """Paint the two big vertical ramps the rest of the frame is drawn over."""
    img = np.empty((height, width, 3), np.float64)
    up = np.linspace(0.0, 1.0, max(horizon, 1))[:, None]
    img[:horizon] = (np.array(_SKY_TOP) * (1 - up) + np.array(_SKY_HAZE) * up)[
        :, None, :
    ]
    down = np.linspace(0.0, 1.0, max(height - horizon, 1))[:, None]
    img[horizon:] = (
        np.array(_ROAD_FAR) * (1 - down) + np.array(_ROAD_NEAR) * down
    )[:, None, :]
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    alpha = np.full((height, width, 1), 255, np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def _road_edges(width: int, t: float) -> tuple[float, float]:
    """Return the road's left and right edge at depth ``t`` (0 far, 1 near).

    Both edges run to the same vanishing point, so the lane markings and the
    verges recede together instead of each keeping its own perspective.
    """
    center = VANISHING_X * width
    half = 0.014 * width + t * (0.66 - 0.014) * width
    return center - half, center + half


def _depth_at(height: int, t: float) -> float:
    """Return the y of depth ``t``, where 0 is the horizon and 1 the camera."""
    horizon = HORIZON * height
    return horizon + t * (height - horizon)


def _draw_verges(canvas: Canvas, width: int, height: int) -> None:
    """Fill the wedges either side of the road, so the road has edges."""
    horizon = HORIZON * height
    left_far, right_far = _road_edges(width, 0.0)
    left_near, right_near = _road_edges(width, 1.0)
    canvas.polygon(
        [(0, horizon), (left_far, horizon), (left_near, height), (0, height)],
        fill=_VERGE,
    )
    canvas.polygon(
        [
            (width, horizon),
            (right_far, horizon),
            (right_near, height),
            (width, height),
        ],
        fill=_VERGE,
    )
    # A kerb along both edges, so the road meets the verge on an edge rather
    # than on a colour change.
    for side in (0, 1):
        canvas.polygon(
            [
                (_road_edges(width, 0.0)[side], horizon),
                (_road_edges(width, 1.0)[side], height),
            ],
            stroke=_KERB,
            stroke_width=max(2.0, 0.009 * width),
            closed=False,
        )


def _draw_skyline(canvas: Canvas, width: int, height: int) -> None:
    """Draw the building silhouettes, each with a few lit windows."""
    horizon = HORIZON * height
    band = 0.30 * height
    for x0, wide, tall in _BUILDINGS:
        left, right = x0 * width, (x0 + wide) * width
        top = horizon - tall * band
        canvas.polygon(
            [(left, top), (right, top), (right, horizon), (left, horizon)],
            fill=_BUILDING,
        )
        # A lighter cap, so the faces are not flat fills against a flat sky.
        canvas.polygon(
            [
                (left, top),
                (right, top),
                (right, top + 0.012 * height),
                (left, top + 0.012 * height),
            ],
            fill=Color(74, 79, 95),
        )
        step = 0.026 * height
        row = top + step
        while row < horizon - step:
            column = left + step
            while column < right - step * 0.6:
                canvas.rounded_rect(
                    Rect(column, row, column + step * 0.4, row + step * 0.5),
                    radius=1.0,
                    fill=_WINDOW,
                )
                column += step
            row += step * 1.7


def _draw_lane(canvas: Canvas, width: int, height: int) -> None:
    """Draw the centre dashes and the two solid edge lines, all converging."""
    for side in (0, 1):
        canvas.polygon(
            [
                (_road_edges(width, 0.0)[side], _depth_at(height, 0.0)),
                (_road_edges(width, 1.0)[side], _depth_at(height, 1.0)),
            ],
            stroke=_PAINT,
            stroke_width=max(1.5, 0.004 * width),
            closed=False,
        )
    center = VANISHING_X * width
    # Dashes bunch toward the horizon: even steps in `t` would space them
    # evenly on screen, which is exactly what perspective does not do.
    for index in range(11):
        near = ((index + 1) / 11.0) ** 2.1
        far = (index / 11.0) ** 2.1
        half_far = 0.002 * width + far * 0.010 * width
        half_near = 0.002 * width + near * 0.010 * width
        canvas.polygon(
            [
                (center - half_far, _depth_at(height, far)),
                (center + half_far, _depth_at(height, far)),
                (center + half_near, _depth_at(height, near)),
                (center - half_near, _depth_at(height, near)),
            ],
            fill=_PAINT,
        )


def _box(width: int, height: int, spec: dict) -> tuple[float, ...]:
    """Turn a normalized box spec into pixel ``(left, top, w, h)``."""
    return (
        spec["x"] * width,
        spec["y"] * height,
        spec["w"] * width,
        spec["h"] * height,
    )


def _draw_car(canvas: Canvas, width: int, height: int) -> None:
    """Draw a car, seen from behind, filling :data:`CAR`."""
    left, top, wide, tall = _box(width, height, CAR)

    def at(u: float, v: float) -> tuple[float, float]:
        return left + u * wide, top + v * tall

    canvas.rounded_rect(
        Rect(*at(-0.07, 0.93), *at(1.07, 1.11)),
        radius=tall * 0.09,
        fill=_SHADOW,
    )
    for u0, u1 in ((0.05, 0.27), (0.73, 0.95)):
        canvas.rounded_rect(
            Rect(*at(u0, 0.70), *at(u1, 1.0)), radius=tall * 0.07, fill=_TYRE
        )
    canvas.polygon(
        [at(0.16, 0.40), at(0.84, 0.40), at(0.74, 0.02), at(0.26, 0.02)],
        fill=_CAR_BODY,
    )
    canvas.polygon(
        [at(0.24, 0.35), at(0.76, 0.35), at(0.69, 0.09), at(0.31, 0.09)],
        fill=_GLASS,
    )
    canvas.rounded_rect(
        Rect(*at(0.0, 0.36), *at(1.0, 0.92)),
        radius=tall * 0.12,
        fill=_CAR_BODY,
    )
    canvas.rounded_rect(
        Rect(*at(0.0, 0.78), *at(1.0, 0.92)),
        radius=tall * 0.10,
        fill=_CAR_DARK,
    )
    for u0, u1 in ((0.05, 0.24), (0.76, 0.95)):
        canvas.rounded_rect(
            Rect(*at(u0, 0.50), *at(u1, 0.63)),
            radius=tall * 0.05,
            fill=Color(216, 84, 63),
        )
    canvas.rounded_rect(
        Rect(*at(0.40, 0.66), *at(0.60, 0.76)),
        radius=2.0,
        fill=Color(214, 218, 226),
    )


def _draw_person(canvas: Canvas, width: int, height: int) -> None:
    """Draw a standing pedestrian, seen from behind, filling :data:`PERSON`."""
    left, top, wide, tall = _box(width, height, PERSON)

    def at(u: float, v: float) -> tuple[float, float]:
        return left + u * wide, top + v * tall

    canvas.rounded_rect(
        Rect(*at(0.02, 0.96), *at(0.98, 1.04)),
        radius=tall * 0.03,
        fill=_SHADOW,
    )
    canvas.polygon(
        [at(0.30, 0.52), at(0.47, 0.52), at(0.45, 1.0), at(0.30, 1.0)],
        fill=_JEANS,
    )
    canvas.polygon(
        [at(0.53, 0.52), at(0.70, 0.52), at(0.70, 1.0), at(0.55, 1.0)],
        fill=_JEANS,
    )
    canvas.polygon(
        [at(0.13, 0.26), at(0.27, 0.26), at(0.25, 0.62), at(0.11, 0.60)],
        fill=_JACKET,
    )
    canvas.polygon(
        [at(0.73, 0.26), at(0.87, 0.26), at(0.89, 0.60), at(0.75, 0.62)],
        fill=_JACKET,
    )
    canvas.polygon(
        [at(0.22, 0.25), at(0.78, 0.25), at(0.72, 0.60), at(0.28, 0.60)],
        fill=_JACKET,
    )
    canvas.circle(at(0.50, 0.13), radius=tall * 0.11, fill=_HAIR)


def _draw_sign(canvas: Canvas, width: int, height: int) -> None:
    """Draw a round road sign on a post, filling :data:`SIGN`."""
    left, top, wide, tall = _box(width, height, SIGN)

    def at(u: float, v: float) -> tuple[float, float]:
        return left + u * wide, top + v * tall

    canvas.rounded_rect(
        Rect(*at(0.44, 0.36), *at(0.56, 1.0)),
        radius=1.0,
        fill=Color(126, 130, 136),
    )
    canvas.circle(at(0.5, 0.22), radius=wide * 0.5, fill=Color(196, 62, 58))
    canvas.circle(at(0.5, 0.22), radius=wide * 0.34, fill=Color(232, 234, 238))


def traffic(width: int = 840, height: int = 500) -> np.ndarray:
    """Render the street frame: sky, skyline, road, a car and a pedestrian.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        An ``(H, W, 4)`` RGBA frame, ready to annotate.

    """
    horizon = int(HORIZON * height)
    canvas = Canvas.from_rgba(_sky_and_asphalt(width, height, horizon))
    _draw_skyline(canvas, width, height)
    _draw_verges(canvas, width, height)
    _draw_lane(canvas, width, height)
    _draw_sign(canvas, width, height)
    _draw_person(canvas, width, height)
    _draw_car(canvas, width, height)
    return canvas.to_rgba()


def _stencil(
    width: int, height: int, draw: Callable[[Canvas, int, int], None]
) -> np.ndarray:
    """Rasterize ``draw`` to a boolean mask, so a shape can become a class id."""
    canvas = Canvas.blank(width, height)
    draw(canvas, width, height)
    return canvas.to_rgba()[:, :, 3] > 127


def label_map(width: int, height: int) -> np.ndarray:
    """Return the same scene as a semantic label map, keyed by :data:`CLASSES`.

    The classes are painted in the order they occlude one another, and the
    verges are left as 0 so an ``ignore_index`` has real area to drop out.

    Args:
        width: Map width in pixels.
        height: Map height in pixels.

    Returns:
        An ``(H, W)`` int32 array of class ids.

    """
    labels = np.zeros((height, width), np.int32)
    labels[: int(HORIZON * height)] = 1
    labels[_stencil(width, height, _draw_skyline)] = 2
    left, right = _road_edges(width, 0.0)
    left_near, right_near = _road_edges(width, 1.0)
    horizon = HORIZON * height

    def road(canvas: Canvas, _width: int, _height: int) -> None:
        del _width, _height
        canvas.polygon(
            [
                (left, horizon),
                (right, horizon),
                (right_near, height),
                (left_near, height),
            ],
            fill=Color(255, 255, 255),
        )

    labels[_stencil(width, height, road)] = 3
    labels[_stencil(width, height, _draw_sign)] = 6
    labels[_stencil(width, height, _draw_person)] = 5
    labels[_stencil(width, height, _draw_car)] = 4
    return labels
