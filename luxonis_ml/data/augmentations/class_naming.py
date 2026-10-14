"""Give augmented labels the classes that their class meanings ask for.

A class can say what each turn and mirror of the image makes of it, see
`luxonis_ml.ldf.class_meaning`. `AlbumentationsEngine` measures the
symmetry of the whole image with three probe points, which move with the
image through every spatial transform:

    1. Before the spatial transforms, the probes go to the center of the
       image, as an L of three points.
    2. After them, the probes give the measured map, and
       `luxonis_ml.ldf.class_meaning.nearest_symmetry` rounds it to a turn
       or a mirror.
    3. When a class of the sample cannot show that symmetry, `correction`
       finds one more flip or turn that brings the sample to a symmetry
       that every class can show. The engine applies it to the image and
       to every label.
    4. After the pipeline, the ``relabel`` functions give each box, each
       image-level class and each segmentation channel its new class.

The engine works on class IDs. `LuxonisLoader` turns the class names of
each task into the IDs of its output, including a ``background`` class
that it adds, and gives the engine one `ClassTable` for each task.
"""

from collections.abc import Iterable, Mapping
from typing import Final, TypeAlias

import numpy as np

from luxonis_ml.ldf.class_meaning import (
    SquareSymmetry,
    compose,
    nearest_symmetry,
)

from .keypoint_naming import tag, untag

#: The name of the target that carries the probes of the image.
IMAGE_PROBES: Final = "_luxonis_image_probes"

#: The distance between the probes, as a part of the shorter image side.
_PROBE_SPACING: Final = 0.05

#: The order in which `correction` tries the extra symmetries: no change
#: first, then the mirrors, then the turns.
_CORRECTIONS: Final[tuple[SquareSymmetry, ...]] = (
    "identity",
    "hflip",
    "vflip",
    "rotate_180",
    "transpose",
    "anti_transpose",
    "rotate_90",
    "rotate_270",
)

ClassTable: TypeAlias = Mapping[int, Mapping[SquareSymmetry, int | None]]
"""The class ID that each symmetry makes of each class ID of one task.

``None`` means that the symmetry makes no valid sample of the class. A
class without an entry keeps its ID.
"""


def image_probes(height: int, width: int) -> np.ndarray:
    """Return the tagged probe rows of an image.

    Example:
        >>> image_probes(40, 20).tolist()
        [[10.0, 20.0, 1.0, 0.0, 2.0], [11.0, 20.0, 1.0, 1.0, 2.0], [10.0, 21.0, 1.0, 2.0, 2.0]]

    """
    spacing = _PROBE_SPACING * min(height, width)
    x, y = width / 2, height / 2
    rows = np.array(
        [
            [x, y, 1.0, 2.0],
            [x + spacing, y, 1.0, 2.0],
            [x, y + spacing, 1.0, 2.0],
        ]
    )
    return tag(rows)


def measured_symmetry(probes: np.ndarray) -> SquareSymmetry | None:
    """Return the symmetry that the image probes show after a pipeline.

    Returns:
        The symmetry, or ``None`` when a probe was lost or the map is no
        turn or mirror.

    Example:
        >>> probes = np.array(
        ...     [[9.0, 5, 1, 0, 2], [8, 5, 1, 1, 2], [9, 6, 1, 2, 2]]
        ... )
        >>> measured_symmetry(probes)
        'hflip'

    """
    rows, lost = untag(probes, 3)
    if lost.any():
        return None
    origin = rows[0, :2]
    return nearest_symmetry(
        np.column_stack([rows[1, :2] - origin, rows[2, :2] - origin])
    )


def correction(
    present: Iterable[tuple[str, int]],
    tables: Mapping[str, ClassTable],
    measured: SquareSymmetry,
) -> SquareSymmetry:
    """Return the extra symmetry that makes a valid sample.

    Args:
        present: The task and the class ID of each class in the sample.
        tables: The class table of each task.
        measured: The symmetry that the pipeline gave the sample.

    Returns:
        ``"identity"`` when every class can show ``measured``. Otherwise
        the first extra symmetry, in the order mirrors before turns, after
        which every class can show the result. The identity always works,
        so a result exists.

    Example:
        A digit 6 (class 0) cannot show a mirror, so the sample gets a
        second horizontal flip:

        >>> tables = {
        ...     "digits": {
        ...         0: {
        ...             "identity": 0,
        ...             "hflip": None,
        ...         },
        ...     },
        ... }
        >>> correction([("digits", 0)], tables, "hflip")
        'hflip'

    """
    present = list(present)
    for extra in _CORRECTIONS:
        result = compose(extra, measured)
        if all(
            tables[task].get(class_id, {}).get(result, class_id) is not None
            for task, class_id in present
        ):
            return extra
    raise AssertionError("The identity is valid for every class.")


def _new_id(table: ClassTable, class_id: int, symmetry: SquareSymmetry) -> int:
    """Return the class ID of ``class_id`` after ``symmetry``."""
    new = table.get(class_id, {}).get(symmetry, class_id)
    return class_id if new is None else new


def relabel_boxes(
    boxes: np.ndarray, table: ClassTable, symmetry: SquareSymmetry
) -> np.ndarray:
    """Give each LDF box row ``[c, x, y, w, h]`` its new class.

    Example:
        >>> boxes = np.array([[0.0, 0.1, 0.1, 0.2, 0.2]])
        >>> table = {
        ...     0: {
        ...         "hflip": 1,
        ...     },
        ... }
        >>> relabel_boxes(boxes, table, "hflip")[:, 0].tolist()
        [1.0]

    """
    boxes = boxes.copy()
    for row in boxes:
        row[0] = _new_id(table, int(row[0]), symmetry)
    return boxes


def relabel_vector(
    vector: np.ndarray, table: ClassTable, symmetry: SquareSymmetry
) -> np.ndarray:
    """Move each entry of a class vector to its new class.

    Example:
        >>> vector = np.array([1.0, 0.0, 0.0])
        >>> table = {
        ...     0: {
        ...         "hflip": 1,
        ...     },
        ... }
        >>> relabel_vector(vector, table, "hflip").tolist()
        [0.0, 1.0, 0.0]

    """
    result = np.zeros_like(vector)
    for class_id, value in enumerate(vector):
        new = _new_id(table, class_id, symmetry)
        result[new] = max(result[new], value)
    return result


def relabel_channels(
    masks: np.ndarray, table: ClassTable, symmetry: SquareSymmetry
) -> np.ndarray:
    """Move each channel of a ``(C, H, W)`` segmentation to its new class.

    Example:
        >>> masks = np.stack([np.ones((2, 2)), np.zeros((2, 2))])
        >>> table = {
        ...     0: {
        ...         "hflip": 1,
        ...     },
        ... }
        >>> relabel_channels(masks, table, "hflip")[1].sum()
        np.float64(4.0)

    """
    result = np.zeros_like(masks)
    for class_id, channel in enumerate(masks):
        new = _new_id(table, class_id, symmetry)
        result[new] = np.maximum(result[new], channel)
    return result
