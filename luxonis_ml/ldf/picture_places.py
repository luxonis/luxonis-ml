r"""Keypoints named after a place in the picture.

Most keypoint names follow the object: the left eye of a person stays the
left eye when the picture turns. A few names follow the picture instead:
the corner of a parking space or of a floor tile has no front, so its
name only says where the corner is, such as ``top_left``. A record gives
such a keypoint a `PicturePlace`, and after any augmentation the keypoint
takes the name of the place where it is in the new picture.

The places form a :math:`3 \times 3` grid. The engine compares the
declared grid with where the keypoints are, and rounds the difference to
the nearest of the 8 symmetries of a square: the rotations by multiples of
:math:`90°` and the mirrors. Places are thus renamed by a whole symmetry
of the grid, so the declared places must form full sets: the four
corners, the four sides, and the center, each set complete or absent.

Example:
    The corners of a tile after a rotation by :math:`90°` clockwise. The
    point declared ``top_left`` is now at the top right, so it takes that
    name:

    >>> import numpy as np
    >>> places = ["top_left", "top_right", "bottom_right", "bottom_left"]
    >>> turned = np.array([[1.0, 0], [1, 1], [0, 1], [0, 0]])
    >>> symmetry = places_symmetry(places, turned)
    >>> [places[i] for i in renaming(places, symmetry)]
    ['top_right', 'bottom_right', 'bottom_left', 'top_left']

"""

from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

import numpy as np

PicturePlace: TypeAlias = Literal[
    "top_left",
    "top",
    "top_right",
    "left",
    "center",
    "right",
    "bottom_left",
    "bottom",
    "bottom_right",
]
r"""A place on the :math:`3 \times 3` grid of the picture."""

#: Column and row of each place, with ``y`` pointing down as in a picture.
_GRID: Final[dict[str, tuple[int, int]]] = {
    "top_left": (-1, -1),
    "top": (0, -1),
    "top_right": (1, -1),
    "left": (-1, 0),
    "center": (0, 0),
    "right": (1, 0),
    "bottom_left": (-1, 1),
    "bottom": (0, 1),
    "bottom_right": (1, 1),
}

#: The places that a symmetry of the square can swap with each other.
_SETS: Final = (
    ("top_left", "top_right", "bottom_right", "bottom_left"),
    ("top", "right", "bottom", "left"),
    ("center",),
)


def _square_symmetries() -> list[np.ndarray]:
    """Return the 8 symmetries of a square as integer matrices."""
    turn = np.array([[0, -1], [1, 0]])
    mirror = np.diag([1, -1])
    symmetries = []
    rotation = np.eye(2, dtype=int)
    for _ in range(4):
        symmetries += [rotation, rotation @ mirror]
        rotation = turn @ rotation
    return symmetries


_SYMMETRIES: Final = _square_symmetries()


def missing_places(places: Sequence[PicturePlace | None]) -> list[str]:
    """Return the places that complete the sets the declared places start.

    Example:
        >>> missing_places(["top_left", "top_right", None])
        ['bottom_left', 'bottom_right']

    """
    declared = {place for place in places if place is not None}
    return sorted(
        place
        for group in _SETS
        if declared & set(group)
        for place in group
        if place not in declared
    )


def places_symmetry(
    places: Sequence[PicturePlace | None],
    points: np.ndarray,
    usable: np.ndarray | None = None,
) -> np.ndarray | None:
    r"""Return the symmetry of the grid that matches where the points are.

    It fits the linear map from the declared grid to the points, and
    rounds it to the nearest symmetry of a square. Only the points with a
    place and with ``usable`` set count.

    Args:
        places: The place of each keypoint, or ``None``.
        points: The ``(x, y)`` of each keypoint in the picture.
        usable: Which points count, for example the labeled ones.

    Returns:
        A :math:`2 \times 2` integer matrix, or ``None`` when the usable
        points with a place do not span the grid in both directions.

    """
    chosen = [
        (i, place)
        for i, place in enumerate(places)
        if place is not None and (usable is None or usable[i])
    ]
    if len(chosen) < 3:
        return None
    grid = np.array([_GRID[place] for _, place in chosen], dtype=float)
    seen = np.asarray(points, dtype=float)[[i for i, _ in chosen], :2]
    grid -= grid.mean(axis=0)
    seen -= seen.mean(axis=0)
    if np.linalg.matrix_rank(grid, tol=1e-6) < 2:
        return None
    fitted, *_ = np.linalg.lstsq(grid, seen, rcond=None)
    u, _, vt = np.linalg.svd(fitted.T)
    nearest = u @ vt
    return max(_SYMMETRIES, key=lambda s: float(np.trace(s.T @ nearest)))


def renaming(
    places: Sequence[PicturePlace | None], symmetry: np.ndarray
) -> list[int]:
    """Return the column each keypoint moves to under ``symmetry``.

    A keypoint declared at place :math:`p` is now at place :math:`S p`,
    so it takes the column of the keypoint declared there. A keypoint
    without a place keeps its column.

    Args:
        places: The place of each keypoint, or ``None``. The places form
            full sets, see `missing_places`.
        symmetry: A symmetry from `places_symmetry`.

    Returns:
        For each column, the column that its point moves to.

    """
    column = {place: i for i, place in enumerate(places) if place is not None}
    by_position = {position: place for place, position in _GRID.items()}
    target = []
    for i, place in enumerate(places):
        if place is None:
            target.append(i)
            continue
        moved = symmetry @ np.array(_GRID[place])
        target.append(column[by_position[(int(moved[0]), int(moved[1]))]])
    return target


__all__ = ["PicturePlace", "missing_places", "places_symmetry", "renaming"]
