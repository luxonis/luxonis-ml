"""The engine gives each class the name its meaning asks for.

Each check reads the picture, not the engine. Each box holds marks in
two colors: the red mark is the top of the class, and the green mark is
its right. After the pipeline, the class must name what the marks show:
an arrow points to the red mark, and a left hand keeps the turn from the
red mark to the green mark. Random transforms run with several seeds.
"""

from typing import Any

import numpy as np
import pytest

from luxonis_ml.data.augmentations import AlbumentationsEngine
from luxonis_ml.data.augmentations.class_naming import ClassTable
from luxonis_ml.ldf import ClassMetadata

SIZE = 64
BOX = np.array([0.3, 0.3, 0.4, 0.4])
RED, GREEN = 0, 1

ARROWS = ["arrow_up", "arrow_right", "arrow_down", "arrow_left"]
HANDS = ["left_hand", "right_hand"]
CLASSES = {
    "arrows": ClassMetadata(
        direction_sets=[
            {
                "up": "arrow_up",
                "right": "arrow_right",
                "down": "arrow_down",
                "left": "arrow_left",
            },
        ]
    ),
    "hands": ClassMetadata(mirror_pairs=[("left_hand", "right_hand")]),
    "digits": ClassMetadata(upright=["digit_6"]),
}
NAMES = {
    "arrows": ARROWS,
    "hands": HANDS,
    "digits": ["digit_6"],
}


def id_table(task: str) -> ClassTable:
    """Turn the class names of a task into the IDs of `NAMES`."""
    ids = {name: i for i, name in enumerate(NAMES[task])}
    return {
        ids[name]: {
            symmetry: None if target is None else ids[target]
            for symmetry, target in row.items()
        }
        for name, row in CLASSES[task].table().items()
    }


def picture() -> np.ndarray:
    """Draw the red mark at the top of the box, the green at its right."""
    image = np.zeros((SIZE, SIZE, 3))
    image[20:25, 30:34, RED] = 255
    image[30:34, 39:44, GREEN] = 255
    return image


def augment(
    config: list[dict[str, Any]], task: str, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Augment one box of class 0, and return the image and the box."""
    target = f"{task}/boundingbox"
    engine = AlbumentationsEngine(
        SIZE,
        SIZE,
        {target: "boundingbox"},
        {target: len(NAMES[task])},
        ["image"],
        config,
        class_symmetry={task: id_table(task)},
        seed=seed,
    )
    labels = {target: np.array([[0, *BOX]])}
    images, out = engine.apply([({"image": picture()}, labels)])
    return images["image"], out[target][0]


def mark(image: np.ndarray, box: np.ndarray, color: int) -> np.ndarray:
    """Return where a mark is, from the center of the box, in pixels."""
    ys, xs = np.nonzero(image[..., color] > 128)
    center = (box[1:3] + box[3:5] / 2) * SIZE
    return np.array([xs.mean(), ys.mean()]) - center


def pointing(offset: np.ndarray) -> str:
    """Return the arrow class that points at an offset."""
    x, y = offset
    if abs(x) > abs(y):
        return "arrow_right" if x > 0 else "arrow_left"
    return "arrow_down" if y > 0 else "arrow_up"


def is_mirrored(image: np.ndarray, box: np.ndarray) -> bool:
    """Return whether the turn from the red mark to the green one changed."""
    red, green = mark(image, box, RED), mark(image, box, GREEN)
    return bool(red[0] * green[1] - red[1] * green[0] < 0)


SYMMETRIC = [
    pytest.param([{"name": "HorizontalFlip", "params": {"p": 1}}], id="h"),
    pytest.param([{"name": "VerticalFlip", "params": {"p": 1}}], id="v"),
    pytest.param([{"name": "Transpose", "params": {"p": 1}}], id="t"),
    pytest.param([{"name": "RandomRotate90", "params": {"p": 1}}], id="r90"),
    pytest.param([{"name": "D4", "params": {"p": 1}}], id="d4"),
    pytest.param(
        [{"name": "Rotate", "params": {"limit": (60, 70), "p": 1}}],
        id="rotate_65",
    ),
]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("config", SYMMETRIC)
def test_an_arrow_points_where_its_tip_is(
    config: list[dict[str, Any]], seed: int
) -> None:
    image, box = augment(config, "arrows", seed)

    assert ARROWS[int(box[0])] == pointing(mark(image, box, RED))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("config", SYMMETRIC)
def test_a_hand_swaps_side_after_each_mirror(
    config: list[dict[str, Any]], seed: int
) -> None:
    image, box = augment(config, "hands", seed)

    expected = "right_hand" if is_mirrored(image, box) else "left_hand"
    assert HANDS[int(box[0])] == expected


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("config", SYMMETRIC[:5])
def test_an_upright_class_comes_back_upright(
    config: list[dict[str, Any]], seed: int
) -> None:
    image, box = augment(config, "digits", seed)

    assert box[0] == 0
    assert pointing(mark(image, box, RED)) == "arrow_up"
    assert not is_mirrored(image, box)


def test_a_small_turn_keeps_the_classes() -> None:
    config = [{"name": "Rotate", "params": {"limit": (20, 20), "p": 1}}]

    image, box = augment(config, "arrows")

    assert ARROWS[int(box[0])] == "arrow_up"
    assert pointing(mark(image, box, RED)) == "arrow_up"
