"""Paint synthetic street frames, with exact labels for every pixel.

A frame is a pinhole view down a straight street: road, curbs, sidewalks,
grass, two rows of buildings, trees, signs, vehicles seen from behind, and
people who face either way. For each pixel the painter keeps the object that
owns it, its semantic class, its depth, its surface normal and its motion. The
masks, the depth map, the normals and the optical flow so come from the same
geometry as the picture.

World units are meters: ``x`` to the right of the street axis, ``y`` up from
the ground, ``z`` forward from the camera. Each object is a flat cut-out that
faces the camera at its own depth, and the painter tests each pixel against
the depth of what is already there.

Example:
    >>> import numpy as np
    >>> from luxonis_ml.testing.street import Camera, random_street
    >>> street = random_street(np.random.default_rng(0), vehicles=2)
    >>> frame = street.render(Camera(width=160, height=100))
    >>> frame.image.shape, frame.depth.shape, frame.flow.shape
    ((100, 160, 3), (100, 160), (100, 160, 2))

"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from math import cos, pi, sin
from typing import Literal, TypeAlias

import cv2
import numpy as np

Color: TypeAlias = tuple[int, int, int]
"""An RGB color."""

Point: TypeAlias = tuple[float, float]
"""A point in meters, or a normalized image position."""

Box: TypeAlias = tuple[float, float, float, float]
"""A normalized ``(x, y, w, h)`` box."""

LabeledPoint: TypeAlias = tuple[float, float, int]
"""A normalized keypoint ``(x, y, visibility)`` in the COCO convention."""

#: Semantic classes in label-map order. Label 0 is unlabeled: the curbs, the
#: sign poles and the traffic cone. The loader turns it into ``background``.
SEMANTIC_CLASSES = (
    "sky",
    "building",
    "road",
    "sidewalk",
    "vegetation",
    "lane_marking",
    "vehicle",
    "person",
    "sign",
)
_SEMANTIC_ID = {name: i + 1 for i, name in enumerate(SEMANTIC_CLASSES)}

#: The 17 COCO keypoints. The left of a person who faces the camera is on the
#: right of the image.
POSE_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

#: The COCO skeleton, as pairs of keypoint names.
POSE_EDGES = [
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
]

#: The keypoints that a horizontal flip swaps, as pairs of names.
POSE_FLIP_PAIRS = [
    (f"left_{part}", f"right_{part}")
    for part in (
        "eye",
        "ear",
        "shoulder",
        "elbow",
        "wrist",
        "hip",
        "knee",
        "ankle",
    )
]

#: The COCO OKS sigmas, in `POSE_NAMES` order.
POSE_SIGMAS = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
]  # fmt: skip

#: Five face landmarks. Their names carry the side words that let LDF infer
#: the flip pairs.
FACE_NAMES = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")

#: The edges between the face landmarks, as pairs of names.
FACE_EDGES = [
    ("left_eye", "right_eye"),
    ("left_eye", "nose"),
    ("right_eye", "nose"),
    ("nose", "mouth_left"),
    ("nose", "mouth_right"),
    ("mouth_left", "mouth_right"),
]

#: The paint colors of the vehicles, by name.
VEHICLE_COLORS: dict[str, Color] = {
    "red": (170, 34, 40),
    "blue": (38, 72, 152),
    "white": (226, 226, 222),
    "black": (32, 32, 36),
    "silver": (160, 166, 174),
    "yellow": (222, 178, 36),
    "green": (36, 112, 72),
}

#: Width and height of each vehicle class, in meters.
VEHICLE_SIZE = {"car": (1.8, 1.45), "truck": (2.45, 3.3), "bus": (2.55, 3.1)}

#: Length of each vehicle class, in meters.
VEHICLE_LENGTH = {"car": 4.5, "truck": 9.0, "bus": 12.0}

#: Rectangle of a number plate, in meters from the foot of the vehicle.
PLATE_RECT = (-0.26, 0.4, 0.26, 0.52)

#: Half width of the road, the curbs, the sidewalks and the grass, in meters.
ROAD, CURB, WALK, FACADE = 3.5, 3.65, 6.2, 8.5

#: Seconds between two frames of a sequence.
DT = 0.1

#: How far behind an object a surface must be for the object to cover it.
#: Without the slack, the ground just under a foot would hide the foot.
_DEPTH_SLACK = 0.6
_NEAR, _FAR = 2.5, 400.0
_SUPERSAMPLE = 2

_INK: Color = (24, 24, 28)
_GLASS: Color = (52, 66, 82)
_TYRE: Color = (26, 26, 30)
_TAIL_LIGHT: Color = (214, 32, 28)
_PLATE: Color = (236, 236, 226)
_SKY: dict[str, tuple[Color, Color]] = {
    "day": ((70, 118, 190), (190, 208, 230)),
    "dusk": ((52, 46, 104), (238, 146, 92)),
    "night": ((6, 8, 22), (30, 34, 60)),
}


@dataclass(frozen=True)
class Camera:
    """A level pinhole camera that looks down the street.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fov: Horizontal field of view in degrees.
        horizon: Row of the horizon, as a fraction of the height.
        mount: Height of the camera above the ground, in meters.
        offset: Lateral position of the camera. A stereo rig moves it.

    """

    width: int = 640
    height: int = 400
    fov: float = 64.0
    horizon: float = 0.45
    mount: float = 1.4
    offset: float = 0.0

    @property
    def focal(self) -> float:
        """The focal length in pixels."""
        return self.width / 2 / float(np.tan(np.radians(self.fov) / 2))

    def project(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        """Project world points that share one depth to pixel positions.

        Args:
            x: Lateral positions in meters.
            y: Heights above the ground in meters.
            z: The depth of every point, in meters.

        Returns:
            An ``(N, 2)`` array of ``(u, v)`` pixel positions.

        """
        f = self.focal / z
        u = self.width / 2 + (x - self.offset) * f
        v = self.horizon * self.height + (self.mount - y) * f
        return np.stack([u, v], axis=-1)


@dataclass
class Part:
    """One flat shape of an object, in meters from the foot of the object.

    Attributes:
        shape: ``poly`` fills ``points``, ``line`` strokes them ``size``
            wide, and ``circle`` fills a disc of radius ``size`` around the
            first point.
        points: The points of the shape.
        color: The fill color.
        size: The stroke width or the radius, in meters.
        owned: Whether the pixels belong to the instance mask of the object.
            A part that is not owned takes the class ``semantic``, or none.
        semantic: The class of a part that is not owned.
        normal: How the surface normal varies across the shape.
        emissive: Whether the part glows at night.
        text: Text printed into the shape when the shape is large enough.
        text_color: The color of the text.

    """

    shape: Literal["poly", "line", "circle"]
    points: Sequence[Point]
    color: Color
    size: float = 0.0
    owned: bool = True
    semantic: str | None = None
    normal: Literal["flat", "sphere", "cylinder"] = "flat"
    emissive: bool = False
    text: str = ""
    text_color: Color = _INK


def _box(x0: float, y0: float, x1: float, y1: float, color: Color) -> Part:
    """Make an axis-aligned rectangle."""
    return Part("poly", [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color)


def _regular(n: int, r: float, cy: float, turn: float = 0.0) -> list[Point]:
    """Return the corners of a regular ``n``-gon centered at ``(0, cy)``."""
    return [
        (r * cos(turn + 2 * pi * k / n), cy + r * sin(turn + 2 * pi * k / n))
        for k in range(n)
    ]


def _shade(color: Color, k: float) -> Color:
    """Scale the brightness of a color."""
    r, g, b = (int(min(255.0, c * k)) for c in color)
    return r, g, b


@dataclass
class Thing:
    """An object on the street.

    Attributes:
        kind: What the object is. Only vehicles, people and signs become
            annotations. Trees and the traffic cone are scenery.
        label: The class name of the annotation.
        x: Lateral position of the foot of the object.
        z: Depth of the foot of the object.
        track: Instance ID, the same in every frame of a sequence.
        vx: Lateral speed in meters per second.
        vz: Forward speed in meters per second.
        color: The paint of a vehicle, a key of `VEHICLE_COLORS`.
        height: The height of a person, in meters.
        pose: ``standing``, ``walking``, ``waving`` or ``pointing``.
        facing: Whether a person faces the camera.
        phase: The gait phase of a walking person, in radians.
        clothes: The skin, shirt, trousers and hair colors of a person.
        text: The text painted on a plate, a sign or a bus display.
        caption: The text of a sign as metadata. It can hold characters that
            the painter cannot print.
        note: Free text for the metadata, to test how it is escaped.

    """

    kind: Literal["vehicle", "person", "sign", "tree", "cone"]
    label: str
    x: float
    z: float
    track: int = -1
    vx: float = 0.0
    vz: float = 0.0
    color: str = "silver"
    height: float = 1.75
    pose: str = "standing"
    facing: bool = True
    phase: float = 0.0
    clothes: tuple[Color, Color, Color, Color] = (
        (224, 182, 150),
        (60, 90, 150),
        (50, 50, 60),
        (50, 36, 30),
    )
    text: str = ""
    caption: str = ""
    note: str = ""

    def parts(self) -> list[Part]:
        """Return the shapes that draw the object, from back to front."""
        return _PARTS[self.kind](self)


_BASE_POSE: dict[str, Point] = {
    "nose": (0.0, 1.60),
    "left_eye": (0.035, 1.645),
    "right_eye": (-0.035, 1.645),
    "left_ear": (0.078, 1.625),
    "right_ear": (-0.078, 1.625),
    "left_shoulder": (0.19, 1.44),
    "right_shoulder": (-0.19, 1.44),
    "left_elbow": (0.235, 1.16),
    "right_elbow": (-0.235, 1.16),
    "left_wrist": (0.25, 0.9),
    "right_wrist": (-0.25, 0.9),
    "left_hip": (0.11, 0.95),
    "right_hip": (-0.11, 0.95),
    "left_knee": (0.115, 0.52),
    "right_knee": (-0.115, 0.52),
    "left_ankle": (0.12, 0.08),
    "right_ankle": (-0.12, 0.08),
}
_FACE_POINTS: dict[str, Point] = {
    "left_eye": (0.035, 1.645),
    "right_eye": (-0.035, 1.645),
    "nose": (0.0, 1.60),
    "mouth_left": (0.032, 1.565),
    "mouth_right": (-0.032, 1.565),
}


def person_joints(thing: Thing) -> dict[str, Point]:
    """Return the 17 COCO joints of a person.

    Args:
        thing: The person.

    Returns:
        The joints in meters from the feet, keyed by `POSE_NAMES`.

    """
    pose = dict(_BASE_POSE)
    swing = sin(thing.phase)
    if thing.pose == "walking":
        for side, lift in (
            ("left", max(0.0, swing)),
            ("right", -min(0.0, swing)),
        ):
            x, y = pose[f"{side}_knee"]
            pose[f"{side}_knee"] = (x, y + 0.08 * lift)
            x, y = pose[f"{side}_ankle"]
            pose[f"{side}_ankle"] = (x, y + 0.16 * lift)
        for side, k in (("left", swing), ("right", -swing)):
            x, y = pose[f"{side}_wrist"]
            pose[f"{side}_wrist"] = (x * (1 + 0.15 * k), y + 0.05 * abs(k))
    elif thing.pose == "waving":
        pose["right_elbow"] = (-0.38, 1.52)
        pose["right_wrist"] = (-0.45 + 0.06 * swing, 1.84)
    elif thing.pose == "pointing":
        pose["left_elbow"] = (0.43, 1.42)
        pose["left_wrist"] = (0.68, 1.42)
    scale, mirror = thing.height / 1.75, 1 if thing.facing else -1
    return {
        name: (mirror * x * scale, y * scale) for name, (x, y) in pose.items()
    }


def face_points(thing: Thing) -> dict[str, Point]:
    """Return the five face landmarks of a person.

    Args:
        thing: The person.

    Returns:
        The landmarks in meters from the feet, keyed by `FACE_NAMES`.

    """
    scale = thing.height / 1.75
    return {n: (x * scale, y * scale) for n, (x, y) in _FACE_POINTS.items()}


def head_circle(thing: Thing) -> tuple[float, float, float]:
    """Return the center and the radius of the head of a person, in meters.

    Args:
        thing: The person.

    Returns:
        The ``(x, y, radius)`` of the head.

    """
    scale = thing.height / 1.75
    return 0.0, 1.63 * scale, 0.115 * scale


def _person_parts(thing: Thing) -> list[Part]:
    """Draw a person as legs, a torso, arms and a head."""
    skin, shirt, trousers, hair = thing.clothes
    joints = person_joints(thing)
    s = thing.height / 1.75
    hx, hy, hr = head_circle(thing)
    parts = []
    for side in ("left", "right"):
        hip, knee, ankle = (
            joints[f"{side}_{p}"] for p in ("hip", "knee", "ankle")
        )
        parts += [
            Part("line", [hip, knee, ankle], trousers, size=0.14 * s),
            Part(
                "circle", [(ankle[0], ankle[1] - 0.03)], _TYRE, size=0.065 * s
            ),
        ]
    out = 0.035 * s * (1 if thing.facing else -1)
    ls, rs = joints["left_shoulder"], joints["right_shoulder"]
    lh, rh = joints["left_hip"], joints["right_hip"]
    torso = [
        (ls[0] + out, ls[1] + 0.03 * s),
        (rs[0] - out, rs[1] + 0.03 * s),
        (rh[0] - out, rh[1] - 0.06 * s),
        (lh[0] + out, lh[1] - 0.06 * s),
    ]
    parts += [
        Part("poly", torso, shirt, normal="cylinder"),
        Part("line", [(0, 1.42 * s), (hx, hy - 0.08 * s)], skin, size=0.1 * s),
    ]
    for side in ("left", "right"):
        shoulder, elbow, wrist = (
            joints[f"{side}_{p}"] for p in ("shoulder", "elbow", "wrist")
        )
        parts += [
            Part("line", [shoulder, elbow], shirt, size=0.1 * s),
            Part("line", [elbow, wrist], skin, size=0.08 * s),
            Part("circle", [wrist], skin, size=0.05 * s),
        ]
    parts.append(Part("circle", [(hx, hy)], skin, size=hr, normal="sphere"))
    if not thing.facing:
        parts.append(
            Part("circle", [(hx, hy)], hair, size=hr * 1.02, normal="sphere")
        )
        return parts
    cap = [
        (hx + hr * cos(a), hy + 0.02 * s + hr * sin(a))
        for a in np.linspace(0, pi, 12)
    ]
    face = face_points(thing)
    return [
        *parts,
        Part("poly", cap, hair, normal="sphere"),
        Part("circle", [face["left_eye"]], _INK, size=0.014 * s),
        Part("circle", [face["right_eye"]], _INK, size=0.014 * s),
        Part(
            "line",
            [face["mouth_left"], face["mouth_right"]],
            (150, 70, 70),
            size=0.012 * s,
        ),
    ]


def _vehicle_parts(thing: Thing) -> list[Part]:
    """Draw a car, a truck or a bus from behind."""
    width, height = VEHICLE_SIZE[thing.label]
    hw = width / 2
    paint = VEHICLE_COLORS[thing.color]
    parts = [
        _box(-hw + 0.06, 0, -hw + 0.4, 0.44, _TYRE),
        _box(hw - 0.4, 0, hw - 0.06, 0.44, _TYRE),
    ]
    if thing.label == "car":
        body = [(-hw, 0.25), (hw, 0.25), (hw, 0.88), (hw - 0.07, 0.98)]
        cabin = [(-hw + 0.1, 0.96), (hw - 0.1, 0.96), (hw - 0.34, height)]
        glass = [
            (-hw + 0.2, 1.02),
            (hw - 0.2, 1.02),
            (hw - 0.4, height - 0.07),
        ]
        parts += [
            Part("poly", _mirror(body), paint, normal="cylinder"),
            Part(
                "poly", _mirror(cabin), _shade(paint, 0.9), normal="cylinder"
            ),
            Part("poly", _mirror(glass), _GLASS),
            _box(-hw + 0.02, 0.22, hw - 0.02, 0.33, (44, 44, 48)),
        ]
    elif thing.label == "truck":
        seam = Part(
            "line", [(0, 0.55), (0, height - 0.06)], _shade(paint, 0.6)
        )
        seam.size = 0.04
        parts += [
            _box(-hw + 0.1, 0.3, hw - 0.1, 0.5, (40, 40, 44)),
            replace(_box(-hw, 0.48, hw, height, paint), normal="cylinder"),
            seam,
            _box(-hw, height - 0.08, hw, height, _shade(paint, 0.75)),
        ]
    else:
        display = replace(
            _box(-0.55, 2.62, 0.55, 2.95, (18, 18, 20)),
            text=thing.caption or "42",
            text_color=(255, 176, 40),
            emissive=True,
        )
        parts += [
            replace(_box(-hw, 0.3, hw, height, paint), normal="cylinder"),
            _box(-hw + 0.15, 1.55, hw - 0.15, 2.5, _GLASS),
            display,
        ]
    return [
        *parts,
        replace(
            _box(-hw + 0.05, 0.68, -hw + 0.32, 0.86, _TAIL_LIGHT),
            emissive=True,
        ),
        replace(
            _box(hw - 0.32, 0.68, hw - 0.05, 0.86, _TAIL_LIGHT), emissive=True
        ),
        replace(_box(*PLATE_RECT, _PLATE), text=thing.text),
    ]


def _mirror(half: list[Point]) -> list[Point]:
    """Close a left-to-right outline with its mirror image."""
    return half + [(-x, y) for x, y in reversed(half)]


def sign_outline(thing: Thing) -> list[Point]:
    """Return the outer outline of a sign panel.

    Args:
        thing: The sign.

    Returns:
        The outline in meters from the foot of the pole.

    """
    if thing.label == "stop":
        return _regular(8, 0.36, 2.35, pi / 8)
    if thing.label == "yield":
        return [(-0.42, 2.62), (0.42, 2.62), (0.0, 1.92)]
    if thing.label == "speed_limit":
        return _regular(24, 0.34, 2.35)
    return [(-0.66, 2.05), (0.66, 2.05), (0.66, 2.55), (-0.66, 2.55)]


def _sign_parts(thing: Thing) -> list[Part]:
    """Draw a sign panel on a pole."""
    outline = sign_outline(thing)
    top = min(y for _, y in outline) + 0.05
    pole = Part("line", [(0, 0), (0, top)], (120, 124, 128), 0.07, owned=False)
    white, red = (244, 244, 244), (192, 28, 34)
    if thing.label == "stop":
        inner = _regular(8, 0.32, 2.35, pi / 8)
        panel = [
            Part("poly", outline, white),
            Part("poly", inner, red, text="STOP", text_color=white),
        ]
    elif thing.label == "yield":
        inner = [(-0.3, 2.55), (0.3, 2.55), (0.0, 2.05)]
        panel = [Part("poly", outline, red), Part("poly", inner, white)]
    elif thing.label == "speed_limit":
        inner = _regular(24, 0.26, 2.35)
        panel = [
            Part("poly", outline, red),
            Part("poly", inner, white, text=thing.text),
        ]
    else:
        blue = (30, 86, 166)
        panel = [
            Part("poly", outline, blue, text=thing.text, text_color=white)
        ]
    return [pole, *panel]


def _tree_parts(thing: Thing) -> list[Part]:
    """Draw a tree, which has a semantic class but no instance."""
    green = (52, 104 + int(abs(thing.x * 7 + thing.z)) % 40, 56)
    return [
        Part("line", [(0, 0), (0, 2.6)], (92, 66, 44), 0.26, owned=False),
        Part("circle", [(0.0, 3.8)], green, 1.6, normal="sphere", owned=False),
        Part(
            "circle",
            [(0.7, 3.2)],
            _shade(green, 0.85),
            1.0,
            normal="sphere",
            owned=False,
        ),
    ]


def _cone_parts(_: Thing) -> list[Part]:
    """Draw a traffic cone. No task labels it; the anomaly map finds it."""
    body = [(-0.2, 0.0), (0.2, 0.0), (0.05, 0.72), (-0.05, 0.72)]
    return [
        Part("poly", body, (240, 110, 20), owned=False),
        replace(_box(-0.12, 0.3, 0.12, 0.42, (245, 245, 245)), owned=False),
    ]


_PARTS: dict[str, Callable[[Thing], list[Part]]] = {
    "person": _person_parts,
    "vehicle": _vehicle_parts,
    "sign": _sign_parts,
    "tree": _tree_parts,
    "cone": _cone_parts,
}


@dataclass(frozen=True)
class Building:
    """One building along a side of the street.

    Attributes:
        side: ``-1`` for the left side, ``1`` for the right side.
        z0: The depth where the facade starts.
        z1: The depth where the facade ends.
        height: The height of the facade, in meters.
        color: The color of the facade.

    """

    side: int
    z0: float
    z1: float
    height: float
    color: Color


@dataclass
class Instance:
    """A visible annotated object of one frame, in normalized coordinates.

    Attributes:
        thing: The object.
        mask: The visible pixels of the object.
        box: The box around the visible pixels.
        keypoints: The joints of a person.
        face_box: The box around the head of a person who faces the camera.
        face_keypoints: The face landmarks of a person who faces the camera.
        plate_box: The box around the number plate of a vehicle.
        outline: The outline of a sign panel.

    """

    thing: Thing
    mask: np.ndarray
    box: Box
    keypoints: dict[str, LabeledPoint] = field(default_factory=dict)
    face_box: Box | None = None
    face_keypoints: dict[str, LabeledPoint] = field(default_factory=dict)
    plate_box: Box | None = None
    outline: list[Point] = field(default_factory=list)


@dataclass
class Frame:
    """One rendered frame and every label that the painter knows for it.

    Attributes:
        camera: The camera that took the frame.
        image: The ``(H, W, 3)`` RGB picture.
        albedo: The ``(H, W, 3)`` RGB picture before light and weather.
        semantic: The ``(H, W)`` label map, indexed as `SEMANTIC_CLASSES`
            plus one. Label 0 is unlabeled.
        depth: The ``(H, W)`` depth in meters, ``inf`` in the sky.
        normals: The ``(H, W, 3)`` unit surface normals in camera axes, with
            ``x`` right, ``y`` down and ``z`` forward. NaN in the sky.
        flow: The ``(H, W, 2)`` motion of each pixel to the next frame, in
            pixels.
        cone: The ``(H, W)`` pixels of the traffic cone.
        instances: The visible annotated objects, from near to far.

    """

    camera: Camera
    image: np.ndarray
    albedo: np.ndarray
    semantic: np.ndarray
    depth: np.ndarray
    normals: np.ndarray
    flow: np.ndarray
    cone: np.ndarray
    instances: list[Instance]


@dataclass
class Street:
    """A street and everything on it, at one moment.

    Attributes:
        things: The objects on the street.
        buildings: The buildings along both sides.
        light: ``day``, ``dusk`` or ``night``.
        weather: ``clear``, ``rain`` or ``fog``.
        speed: The forward speed of the camera, in meters per second.
        travel: How far the camera has moved since the first frame.
        seed: The seed of the sensor noise and the rain.

    """

    things: list[Thing]
    buildings: list[Building]
    light: str = "day"
    weather: str = "clear"
    speed: float = 8.0
    travel: float = 0.0
    seed: int = 0

    def at(self, frame: int) -> "Street":
        """Return the street a number of frames later.

        Args:
            frame: How many frames of `DT` seconds pass.

        Returns:
            A new street with every object moved.

        """
        t = frame * DT
        things = [
            replace(
                thing,
                x=thing.x + thing.vx * t,
                z=thing.z + (thing.vz - self.speed) * t,
                phase=thing.phase + 6.0 * t,
            )
            for thing in self.things
        ]
        return replace(
            self,
            things=things,
            travel=self.travel + self.speed * t,
            seed=self.seed * 1000 + frame,
        )

    def render(self, camera: Camera) -> Frame:
        """Paint the street as a camera sees it.

        Args:
            camera: The camera.

        Returns:
            The frame and its labels.

        """
        return _render(self, camera)


class _Canvas:
    """Supersampled frame buffers that a depth test paints into."""

    def __init__(self, camera: Camera) -> None:
        h, w = camera.height, camera.width
        self.camera = camera
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.glow = np.zeros((h, w, 3), np.float32)
        self.owner = np.full((h, w), -1, np.int32)
        self.semantic = np.zeros((h, w), np.uint8)
        self.depth = np.full((h, w), np.inf, np.float32)
        self.normal = np.full((h, w, 3), np.nan, np.float32)
        self.motion = np.zeros((h, w, 2), np.float32)
        self.cone = np.zeros((h, w), bool)
        rng = np.random.default_rng(7)
        self.grain = rng.normal(0, 4, (h, w, 1)).astype(np.float32)

    def fill(
        self,
        columns: slice,
        region: np.ndarray,
        color: np.ndarray,
        semantic: np.ndarray | int,
        depth: np.ndarray,
        normal: tuple[float, float, float],
        motion: tuple[float, float],
    ) -> np.ndarray:
        """Paint a background region where it is nearer than the paint.

        The arrays cover the given columns of the canvas only. Returns the
        pixels that were painted.
        """
        window = np.s_[:, columns]
        region = region & (depth < self.depth[window])
        self.rgb[window][region] = color[region] + self.grain[window][region]
        self.semantic[window][region] = (
            semantic if isinstance(semantic, int) else semantic[region]
        )
        self.depth[window][region] = depth[region]
        self.normal[window][region] = normal
        self.motion[window][region] = motion
        self.owner[window][region] = -1
        return region


def _pixel_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the column and row of each pixel center."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    return u, v


def _paint_ground(canvas: _Canvas, street: Street) -> None:
    """Paint the sky, the road, the curbs, the sidewalks and the grass."""
    cam = canvas.camera
    h, w = cam.height, cam.width
    u, v = _pixel_grid(h, w)
    du, dv = u - w / 2, v - cam.horizon * h

    top, low = _SKY[street.light]
    t = np.clip(v[:, :1] / max(cam.horizon * h, 1), 0, 1)[..., None]
    canvas.rgb[:] = np.asarray(top) * (1 - t) + np.asarray(low) * t
    canvas.semantic[:] = _SEMANTIC_ID["sky"]

    depth = cam.focal * cam.mount / np.maximum(dv, 1e-3)
    ground = (dv > 0.5) & (depth < _FAR)
    depth = np.minimum(depth, _FAR)
    side = np.abs(cam.offset + du * depth / cam.focal)
    along = depth + street.travel

    color = np.zeros((h, w, 3), np.float32)
    semantic = np.zeros((h, w), np.uint8)
    dash = (side < 0.08) & (along % 9 < 3.5)
    marking = dash | ((side > 3.15) & (side < 3.3))
    walk = (side >= CURB) & (side < WALK)
    joint = walk & ((along % 1.2 < 0.05) | (side % 1.2 < 0.05))
    # A region without a name keeps the class beneath it: the curb stays
    # unlabeled, and the joints of the paving stay sidewalk.
    for region, rgb, name in (
        (side < ROAD, (78, 80, 88), "road"),
        (marking, (224, 224, 214), "lane_marking"),
        ((side >= ROAD) & (side < CURB), (150, 152, 156), None),
        (walk, (128, 126, 122), "sidewalk"),
        (joint, (98, 96, 92), None),
        (side >= WALK, (66, 112, 58), "vegetation"),
    ):
        color[region] = rgb
        if name is not None:
            semantic[region] = _SEMANTIC_ID[name]
    canvas.fill(
        slice(None),
        ground,
        color,
        semantic,
        depth,
        (0.0, -1.0, 0.0),
        (0.0, -street.speed * DT),
    )


def _paint_facades(canvas: _Canvas, street: Street, side: int) -> None:
    """Paint the buildings of one side of the street, windows included."""
    buildings = sorted(
        (b for b in street.buildings if b.side == side), key=lambda b: b.z0
    )
    if not buildings:
        return
    cam = canvas.camera
    h, w = cam.height, cam.width
    # A facade of the left side can only be seen left of the image center.
    columns = slice(0, w // 2) if side < 0 else slice(w // 2, w)
    u, v = _pixel_grid(h, w)
    du = (u - w / 2)[:, columns]
    dv = (v - cam.horizon * h)[:, columns]
    starts = np.array([b.z0 - street.travel for b in buildings])
    ends = np.array([b.z1 - street.travel for b in buildings])
    heights = np.array([b.height for b in buildings], np.float32)
    colors = np.array([b.color for b in buildings], np.float32)

    reach = side * FACADE - cam.offset
    depth = np.clip(cam.focal * reach / np.where(du == 0, 1e-3, du), 0, _FAR)
    y = cam.mount - dv * depth / cam.focal
    index = np.searchsorted(starts, depth, "right") - 1
    index = np.clip(index, 0, len(buildings) - 1)
    hit = (
        (depth > _NEAR)
        & (depth < _FAR)
        & (depth >= starts[index])
        & (depth < ends[index])
        & (y >= 0)
        & (y <= heights[index])
    )
    s = depth - starts[index]
    window = (
        (s % 3.0 > 0.7)
        & (s % 3.0 < 2.1)
        & (y % 3.2 > 1.0)
        & (y % 3.2 < 2.4)
        & (y < heights[index] - 0.6)
    )
    color = colors[index] * (0.8 if side < 0 else 1.0)
    color[window] = _GLASS
    painted = canvas.fill(
        columns,
        hit,
        color,
        _SEMANTIC_ID["building"],
        depth,
        (-side, 0.0, 0.0),
        (0.0, -street.speed * DT),
    )
    if street.light == "night":
        cell = np.floor(s / 3) * 7 + np.floor(y / 3.2) * 13 + index
        lit = painted & window & (cell % 5 < 2)
        canvas.glow[:, columns][lit] = (255, 210, 130)


def _rasterize(
    part: Part, points: np.ndarray, scale: float, size: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Rasterize one part to a mask over its window, and the window corner."""
    reach = part.size * scale
    pad = 0.0 if part.shape == "poly" else reach
    lo = np.floor(points.min(0) - pad).astype(int)
    hi = np.ceil(points.max(0) + pad).astype(int) + 1
    x0, y0 = max(int(lo[0]), 0), max(int(lo[1]), 0)
    x1, y1 = min(int(hi[0]), size[0]), min(int(hi[1]), size[1])
    if x1 <= x0 or y1 <= y0:
        return None
    buf = np.zeros((y1 - y0, x1 - x0), np.uint8)
    fixed = np.round((points - (x0, y0)) * 16).astype(np.int32)
    if part.shape == "circle":
        center = (int(fixed[0, 0]), int(fixed[0, 1]))
        cv2.circle(buf, center, max(int(reach * 16), 8), 1, -1, cv2.LINE_8, 4)
    elif part.shape == "line":
        thickness = max(1, round(reach))
        cv2.polylines(buf, [fixed], False, 1, thickness, cv2.LINE_8, 4)
    else:
        cv2.fillPoly(buf, [fixed], 1, cv2.LINE_8, 4)
    return buf.astype(bool), (x0, y0)


def _normals(
    part: Part,
    points: np.ndarray,
    radius: float,
    shape: tuple[int, int],
    corner: tuple[int, int],
) -> np.ndarray:
    """Return the surface normals of a part over its raster window."""
    normal = np.zeros((*shape, 3), np.float32)
    normal[..., 2] = -1.0
    if part.normal == "flat":
        return normal
    u, v = _pixel_grid(*shape)
    u, v = u + corner[0], v + corner[1]
    if part.normal == "sphere":
        nx = np.clip((u - points[0, 0]) / max(radius, 1e-3), -1, 1)
        ny = np.clip((v - points[0, 1]) / max(radius, 1e-3), -1, 1)
    else:
        lo, hi = points[:, 0].min(), points[:, 0].max()
        half = max((hi - lo) / 2, 1e-3)
        nx = np.clip((u - (lo + hi) / 2) / half, -1, 1) * 0.7
        ny = np.zeros_like(nx)
    normal[..., 0], normal[..., 1] = nx, ny
    normal[..., 2] = -np.sqrt(1 - np.clip(nx**2 + ny**2, 0, 0.98))
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def _print(
    rgb: np.ndarray, hit: np.ndarray, corner: tuple[int, int], part: Part
) -> None:
    """Print the text of a part into its painted pixels, sized to fit."""
    h, w = hit.shape
    if h < 10 or not part.text:
        return
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(part.text, font, 1.0, 2)
    scale = min(w * 0.78 / tw, h * 0.5 / th)
    if scale * th < 5:
        return
    thickness = max(1, round(scale * 2))
    (tw, th), _ = cv2.getTextSize(part.text, font, scale, thickness)
    layer = np.zeros((h, w), np.uint8)
    origin = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(
        layer, part.text, origin, font, scale, 255, thickness, cv2.LINE_AA
    )
    alpha = (layer.astype(np.float32) / 255 * hit)[..., None]
    x0, y0 = corner
    window = rgb[y0 : y0 + h, x0 : x0 + w]
    window[:] = window * (1 - alpha) + np.asarray(part.text_color) * alpha


def _paint_thing(
    canvas: _Canvas, thing: Thing, index: int, speed: float
) -> None:
    """Paint one object part by part, behind anything that is nearer."""
    cam = canvas.camera
    scale = cam.focal / thing.z
    motion = (thing.vx * DT, (thing.vz - speed) * DT)
    owned_class = _SEMANTIC_ID.get(thing.kind, 0)
    for part in thing.parts():
        metric = np.asarray(part.points, np.float64)
        points = cam.project(thing.x + metric[:, 0], metric[:, 1], thing.z)
        raster = _rasterize(part, points, scale, (cam.width, cam.height))
        if raster is None:
            continue
        mask, (x0, y0) = raster
        window = np.s_[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
        hit = mask & (thing.z < canvas.depth[window] + _DEPTH_SLACK)
        if not hit.any():
            continue
        canvas.rgb[window][hit] = part.color
        canvas.depth[window][hit] = thing.z
        canvas.motion[window][hit] = motion
        canvas.owner[window][hit] = index if part.owned else -1
        canvas.semantic[window][hit] = (
            owned_class
            if part.owned
            else _SEMANTIC_ID.get(part.semantic or "", 0)
        )
        shape = (hit.shape[0], hit.shape[1])
        normal = _normals(part, points, part.size * scale, shape, (x0, y0))
        canvas.normal[window][hit] = normal[hit]
        canvas.glow[window][hit] = part.color if part.emissive else (0, 0, 0)
        canvas.cone[window][hit] = thing.kind == "cone"
        _print(canvas.rgb, hit, (x0, y0), part)


def _weather(canvas: _Canvas, street: Street) -> np.ndarray:
    """Return the painted colors under the light and the weather."""
    rgb = canvas.rgb.copy()
    sky = ~np.isfinite(canvas.depth)
    distance = np.where(sky, _FAR, canvas.depth)[..., None]
    rng = np.random.default_rng(street.seed)
    glow = canvas.glow
    if street.light == "day":
        haze = 1 - np.exp(-distance / 320)
        rgb = rgb * (1 - haze) + np.array((196, 206, 222)) * haze
    elif street.light == "dusk":
        rgb[~sky] *= (0.78, 0.66, 0.6)
        rgb += cv2.GaussianBlur(glow, (0, 0), 3) * 0.5
    else:
        rgb[~sky] *= 0.17
        lit = glow.max(-1) > 0
        rgb[lit] = glow[lit]
        rgb += cv2.GaussianBlur(glow, (0, 0), 7) * 0.9
    if street.weather == "fog":
        clear = np.exp(-np.where(sky[..., None], np.inf, distance) / 28)
        fog = (172, 176, 182) if street.light == "day" else (70, 72, 82)
        rgb = rgb * clear + np.array(fog) * (1 - clear)
    elif street.weather == "rain":
        rgb = rgb * 0.75 + rgb.mean(-1, keepdims=True) * 0.15
        h, w = sky.shape
        streaks = np.zeros((h, w), np.float32)
        for _ in range(h * w // 1200):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            length = int(rng.integers(h // 40, h // 12))
            end = (x - length // 5, y + length)
            cv2.line(streaks, (x, y), end, 1.0, 1, cv2.LINE_AA)
        rgb += cv2.GaussianBlur(streaks, (0, 0), 0.8)[..., None] * 60
    return rgb + rng.normal(0, 2.5, rgb.shape)


def _flow(camera: Camera, depth: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Turn the world motion of each pixel into its image motion."""
    h, w = depth.shape
    u, v = _pixel_grid(h, w)
    finite = np.isfinite(depth)
    z = np.where(finite, depth, 1.0)
    grow = z / np.maximum(z + motion[..., 1], 0.5) - 1
    flow = np.stack(
        [
            (u - w / 2) * grow
            + camera.focal * motion[..., 0] * (grow + 1) / z,
            (v - camera.horizon * h) * grow,
        ],
        axis=-1,
    )
    flow[~finite] = 0
    return flow.astype(np.float32)


def _clip_box(points: np.ndarray, width: int, height: int) -> Box | None:
    """Clip the bounds of pixel points to the image, normalized."""
    x0, y0 = np.clip(points.min(0), 0, (width, height))
    x1, y1 = np.clip(points.max(0), 0, (width, height))
    # A far plate is thinner than a pixel, and it still needs its box: a
    # sub-task nests under its parents only when every parent has one.
    if x1 <= x0 or y1 <= y0:
        return None
    return (
        float(x0 / width),
        float(y0 / height),
        float((x1 - x0) / width),
        float((y1 - y0) / height),
    )


def _label_points(
    metric: dict[str, Point],
    thing: Thing,
    camera: Camera,
    depth: np.ndarray,
    hidden: Sequence[str] = (),
) -> dict[str, LabeledPoint]:
    """Project named points and grade them visible, occluded or unlabeled.

    A point outside the image, or one listed in ``hidden``, is unlabeled.
    A point behind a nearer surface is occluded.
    """
    h, w = depth.shape
    xy = np.asarray(list(metric.values()))
    pixels = camera.project(thing.x + xy[:, 0], xy[:, 1], thing.z)
    points: dict[str, LabeledPoint] = {}
    for name, (u, v) in zip(metric, pixels, strict=True):
        if name in hidden or not (0 <= u < w and 0 <= v < h):
            points[name] = (0.0, 0.0, 0)
            continue
        covered = depth[int(v), int(u)] < thing.z - 0.3
        points[name] = (float(u / w), float(v / h), 1 if covered else 2)
    return points


def _instance(
    thing: Thing,
    index: int,
    owner: np.ndarray,
    depth: np.ndarray,
    camera: Camera,
) -> Instance | None:
    """Collect the labels of one object, or None when it is not visible."""
    mask = owner == index
    if mask.sum() < 6:
        return None
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    box = (
        float(xs.min() / w),
        float(ys.min() / h),
        float((xs.max() + 1 - xs.min()) / w),
        float((ys.max() + 1 - ys.min()) / h),
    )
    instance = Instance(thing, mask, box)
    if thing.kind == "person":
        hidden = () if thing.facing else ("nose", "left_eye", "right_eye")
        joints = person_joints(thing)
        instance.keypoints = _label_points(
            joints, thing, camera, depth, hidden
        )
        if thing.facing:
            hx, hy, hr = head_circle(thing)
            corners = camera.project(
                thing.x + np.array([hx - hr, hx + hr]),
                np.array([hy - hr, hy + hr]),
                thing.z,
            )
            instance.face_box = _clip_box(corners, w, h)
            instance.face_keypoints = _label_points(
                face_points(thing), thing, camera, depth
            )
    elif thing.kind == "vehicle":
        x0, y0, x1, y1 = PLATE_RECT
        corners = camera.project(
            thing.x + np.array([x0, x1]), np.array([y0, y1]), thing.z
        )
        instance.plate_box = _clip_box(corners, w, h)
    elif thing.kind == "sign":
        xy = np.asarray(sign_outline(thing))
        pixels = camera.project(thing.x + xy[:, 0], xy[:, 1], thing.z)
        pixels = np.clip(pixels / (w, h), 0, 1)
        instance.outline = [(float(x), float(y)) for x, y in pixels]
    return instance


def _render(street: Street, camera: Camera) -> Frame:
    """Paint a street at the supersampled size, then reduce it."""
    big = replace(
        camera,
        width=camera.width * _SUPERSAMPLE,
        height=camera.height * _SUPERSAMPLE,
    )
    canvas = _Canvas(big)
    _paint_ground(canvas, street)
    _paint_facades(canvas, street, -1)
    _paint_facades(canvas, street, 1)
    order = sorted(
        (i for i, thing in enumerate(street.things) if thing.z > _NEAR),
        key=lambda i: -street.things[i].z,
    )
    for index in order:
        _paint_thing(canvas, street.things[index], index, street.speed)

    size = (camera.width, camera.height)
    image = cv2.resize(
        _weather(canvas, street), size, interpolation=cv2.INTER_AREA
    )
    albedo = cv2.resize(canvas.rgb, size, interpolation=cv2.INTER_AREA)
    pick = np.s_[
        _SUPERSAMPLE // 2 :: _SUPERSAMPLE, _SUPERSAMPLE // 2 :: _SUPERSAMPLE
    ]
    owner, depth = canvas.owner[pick], canvas.depth[pick]
    instances = [
        instance
        for index in reversed(order)
        if street.things[index].kind in ("vehicle", "person", "sign")
        and (
            instance := _instance(
                street.things[index], index, owner, depth, camera
            )
        )
    ]
    return Frame(
        camera=camera,
        image=np.clip(image, 0, 255).astype(np.uint8),
        albedo=np.clip(albedo, 0, 255).astype(np.uint8),
        semantic=canvas.semantic[pick],
        depth=depth,
        normals=canvas.normal[pick],
        flow=_flow(camera, depth, canvas.motion[pick]),
        cone=canvas.cone[pick],
        instances=instances,
    )


_SKIN: list[Color] = [
    (241, 204, 170),
    (224, 182, 150),
    (176, 124, 90),
    (110, 72, 50),
]
_SHIRTS: list[Color] = [
    (60, 90, 150),
    (180, 60, 60),
    (230, 200, 70),
    (60, 140, 90),
    (230, 230, 230),
    (120, 70, 140),
    (40, 40, 44),
]
_TROUSERS: list[Color] = [
    (50, 50, 60),
    (40, 60, 100),
    (110, 96, 80),
    (30, 30, 30),
]
_HAIR: list[Color] = [
    (50, 36, 30),
    (20, 20, 20),
    (150, 110, 60),
    (200, 190, 170),
]
_FACADES: list[Color] = [
    (150, 120, 100),
    (170, 160, 140),
    (120, 110, 105),
    (180, 150, 120),
    (110, 120, 130),
    (160, 100, 80),
]
_SIGN_TEXTS = {
    "stop": ("STOP", "STOP"),
    "yield": ("", "Kein Vortritt"),
    "speed_limit": ("50", "50 km/h"),
    "info": ("HBF", "Zürich Hbf →"),
}


def random_plate(rng: np.random.Generator) -> str:
    """Make up a number plate.

    Args:
        rng: The random generator.

    Returns:
        Two letters and four digits, such as ``"LX 4821"``.

    """
    letters = "".join(rng.choice(list("ABCDEFGHKLMNPRSTUVXZ"), 2))
    return f"{letters} {int(rng.integers(1000, 10000))}"


def random_buildings(
    rng: np.random.Generator, length: float = 360.0
) -> list[Building]:
    """Line both sides of the street with buildings.

    Args:
        rng: The random generator.
        length: How far down the street the buildings reach, in meters.

    Returns:
        The buildings, with gaps between some of them.

    """
    buildings = []
    for side in (-1, 1):
        z = -10.0
        while z < length:
            size = float(rng.uniform(8, 24))
            base = np.asarray(_FACADES[int(rng.integers(len(_FACADES)))])
            r, g, b = np.clip(base + rng.normal(0, 10, 3), 0, 255).astype(int)
            height = float(rng.uniform(6, 24))
            buildings.append(
                Building(side, z, z + size, height, (int(r), int(g), int(b)))
            )
            z += size + (
                float(rng.uniform(3, 10)) if rng.random() < 0.25 else 0
            )
    return buildings


def random_trees(rng: np.random.Generator) -> list[Thing]:
    """Plant a row of trees on the grass of each side.

    Args:
        rng: The random generator.

    Returns:
        The trees.

    """
    trees = []
    for side in (-1, 1):
        z = float(rng.uniform(3, 10))
        while z < 160:
            trees.append(Thing("tree", "tree", side * (WALK + 0.7), z))
            z += float(rng.uniform(9, 15))
    return trees


def random_vehicle(
    rng: np.random.Generator, x: float, z: float, track: int
) -> Thing:
    """Make up a vehicle that drives away from the camera.

    Args:
        rng: The random generator.
        x: Lateral position.
        z: Depth.
        track: Instance ID.

    Returns:
        A car, a truck or a bus.

    """
    label = str(rng.choice(["car", "truck", "bus"], p=[0.7, 0.15, 0.15]))
    return Thing(
        "vehicle",
        label,
        x,
        z,
        track=track,
        vz=float(rng.uniform(5, 12)),
        color=str(rng.choice(list(VEHICLE_COLORS))),
        text=random_plate(rng),
        caption=str(int(rng.integers(1, 99))) if label == "bus" else "",
    )


def random_person(
    rng: np.random.Generator, x: float, z: float, track: int
) -> Thing:
    """Make up a person who stands, walks, waves or points.

    Args:
        rng: The random generator.
        x: Lateral position.
        z: Depth.
        track: Instance ID.

    Returns:
        A person or a child.

    """
    child = rng.random() < 0.2
    pose = str(
        rng.choice(
            ["walking", "standing", "waving", "pointing"],
            p=[0.55, 0.3, 0.1, 0.05],
        )
    )
    return Thing(
        "person",
        "child" if child else "person",
        x,
        z,
        track=track,
        vz=float(rng.choice([-1.4, 1.4])) if pose == "walking" else 0.0,
        height=float(
            rng.uniform(1.05, 1.3) if child else rng.uniform(1.6, 1.92)
        ),
        pose=pose,
        phase=float(rng.uniform(0, 2 * pi)),
        clothes=(
            _SKIN[int(rng.integers(len(_SKIN)))],
            _SHIRTS[int(rng.integers(len(_SHIRTS)))],
            _TROUSERS[int(rng.integers(len(_TROUSERS)))],
            _HAIR[int(rng.integers(len(_HAIR)))],
        ),
    )


def make_sign(label: str, x: float, z: float, track: int) -> Thing:
    """Make a sign with the text of its class.

    Args:
        label: ``stop``, ``yield``, ``speed_limit`` or ``info``.
        x: Lateral position of the pole.
        z: Depth of the pole.
        track: Instance ID.

    Returns:
        The sign.

    """
    text, caption = _SIGN_TEXTS[label]
    return Thing("sign", label, x, z, track=track, text=text, caption=caption)


def _free_depth(
    rng: np.random.Generator,
    taken: list[tuple[float, float]],
    near: float,
    far: float,
    gap: float,
) -> float | None:
    """Draw a depth at least ``gap`` from every taken depth of a lane."""
    for _ in range(30):
        z = float(rng.uniform(near, far))
        if all(abs(z - other) > max(gap, size) for other, size in taken):
            return z
    return None


def random_street(
    rng: np.random.Generator,
    *,
    vehicles: int = 4,
    people: int = 4,
    signs: int = 2,
    near: float = 6.0,
    far: float = 90.0,
    light: str = "day",
    weather: str = "clear",
    cone: bool = False,
) -> Street:
    """Make up a street with vehicles, people, signs, trees and buildings.

    Vehicles keep to the two lanes. People walk or stand on the sidewalks,
    and one in five crosses the road. Signs stand at the curbs.

    Args:
        rng: The random generator.
        vehicles: How many vehicles to place.
        people: How many people to place.
        signs: How many signs to place.
        near: The smallest depth of an annotated object.
        far: The largest depth of an annotated object.
        light: ``day``, ``dusk`` or ``night``.
        weather: ``clear``, ``rain`` or ``fog``.
        cone: Whether to put an unannotated traffic cone on the road.

    Returns:
        The street. Its instance IDs count up from 1.

    """
    things = random_trees(rng)
    lanes: dict[float, list[tuple[float, float]]] = {-1.75: [], 1.75: []}
    track = 1
    for _ in range(vehicles):
        lane = float(rng.choice(list(lanes)))
        z = _free_depth(rng, lanes[lane], near, far, gap=14.0)
        if z is None:
            continue
        vehicle = random_vehicle(
            rng, lane + float(rng.uniform(-0.2, 0.2)), z, track
        )
        lanes[lane].append((z, VEHICLE_LENGTH[vehicle.label] + 4))
        things.append(vehicle)
        track += 1
    for _ in range(people):
        crossing = rng.random() < 0.2
        side = float(rng.choice([-1, 1]))
        x = (
            float(rng.uniform(-3, 3))
            if crossing
            else side * float(rng.uniform(CURB + 0.4, WALK - 0.4))
        )
        person = random_person(
            rng, x, float(rng.uniform(near * 0.7, far * 0.5)), track
        )
        if crossing:
            person = replace(person, pose="walking", vz=0.0, vx=-side * 1.2)
        things.append(person)
        track += 1
    curbs: dict[float, list[tuple[float, float]]] = {-1.0: [], 1.0: []}
    for _ in range(signs):
        label = str(rng.choice(list(_SIGN_TEXTS)))
        side = float(rng.choice([-1, 1]))
        z = _free_depth(rng, curbs[side], near + 2, far * 0.7, gap=6.0)
        if z is None:
            continue
        curbs[side].append((z, 0.0))
        things.append(make_sign(label, side * (CURB + 0.35), z, track))
        track += 1
    if cone:
        things.append(
            Thing(
                "cone",
                "cone",
                float(rng.uniform(-2.5, 2.5)),
                float(rng.uniform(9, 22)),
            )
        )
    return Street(
        things=things,
        buildings=random_buildings(rng),
        light=light,
        weather=weather,
        speed=float(rng.uniform(6, 10)),
        seed=int(rng.integers(1 << 30)),
    )
