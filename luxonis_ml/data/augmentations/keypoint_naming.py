"""Give augmented keypoints the names that their meaning asks for.

A keypoint name can follow the object or the picture (see
`KeypointMetadata`). Any spatial augmentation moves the keypoints but
keeps their columns, so after the pipeline some names are wrong:

    - A mirror partner, such as ``left_eye``, must swap with its partner
      when the pipeline mirrored the picture, and only then. Three probe
      points ride along with each instance. If the triangle they form comes
      out reversed, the map at that instance was a mirror. No transform has
      to say what it does, so a custom transform works too.
    - A keypoint with a place in the picture takes the name of the place
      where it is, see `luxonis_ml.ldf.keypoint_positions`. Only the keypoints
      themselves count, so no probe is needed.

Some Albumentations transforms, such as ``ElasticTransform``, drop the
keypoints outside the image. Each keypoint therefore carries an id while
the transforms run, and the dropped ones come back as unlabeled.

The engine keeps each keypoint as a row ``[x, y, labeled, v]`` while it
augments, with the probes as the last rows of each instance, and adds the
id in front of ``v`` for the spatial transforms. ``v`` stays the last
column, where the transforms that mark keypoints outside the image look
for it.
"""

from collections.abc import Sequence
from typing import Final

import numpy as np

from luxonis_ml.ldf.keypoint_positions import (
    KeypointPosition,
    places_symmetry,
    renaming,
)

#: The number of probe rows that follow the keypoints of an instance.
PROBES: Final = 3

#: The distance between the probes, as a part of the shorter image side.
_PROBE_SPACING: Final = 0.05


def to_rows(
    keypoints: np.ndarray,
    n_keypoints: int,
    n_probes: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Add the ``labeled`` column and the probe rows of each instance.

    The probes are an L of three points at the center of the labeled
    keypoints of the instance, or at the center of the image when it has
    none.

    Args:
        keypoints: Pixel rows ``[x, y, v]``, :math:`K` for each instance.
        n_keypoints: :math:`K`.
        n_probes: `PROBES`, or ``0`` when the task needs no probes.
        height: The image height.
        width: The image width.

    Returns:
        Rows ``[x, y, labeled, v]``, :math:`K` keypoints and then the
        probes for each instance.

    Example:
        >>> rows = to_rows(np.array([[2.0, 4, 2], [6, 4, 0]]), 2, 3, 40, 20)
        >>> rows.tolist()
        [[2.0, 4.0, 1.0, 2.0], [6.0, 4.0, 0.0, 0.0], [2.0, 4.0, 0.0, 2.0], [3.0, 4.0, 0.0, 2.0], [2.0, 5.0, 0.0, 2.0]]

    """
    keypoints = np.asarray(keypoints, dtype=float).reshape(-1, n_keypoints, 3)
    labeled = (keypoints[..., 2] > 0).astype(float)
    rows = np.concatenate(
        [keypoints[..., :2], labeled[..., None], keypoints[..., 2:]], axis=-1
    )
    if n_probes:
        spacing = _PROBE_SPACING * min(height, width)
        probes = np.empty((len(rows), n_probes, 4))
        for i, instance in enumerate(rows):
            placed = instance[instance[:, 2] > 0, :2]
            center = (
                placed.mean(axis=0) if len(placed) else (width / 2, height / 2)
            )
            corner = np.array([[0, 0], [spacing, 0], [0, spacing]])
            probes[i, :, :2] = center + corner[:n_probes]
            probes[i, :, 2] = 0
            probes[i, :, 3] = 2
        rows = np.concatenate([rows, probes], axis=1)
    return rows.reshape(-1, 4)


def tag(rows: np.ndarray) -> np.ndarray:
    """Insert the id of each row in front of ``v``.

    Example:
        >>> tag(np.array([[1.0, 2, 1, 2], [3, 4, 0, 0]])).tolist()
        [[1.0, 2.0, 1.0, 0.0, 2.0], [3.0, 4.0, 0.0, 1.0, 0.0]]

    """
    rows = np.asarray(rows, dtype=float)
    ids = np.arange(len(rows), dtype=float)[:, None]
    return np.concatenate([rows[:, :3], ids, rows[:, 3:]], axis=1)


def untag(rows: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Put each row back in its place and drop the id.

    A row that a transform dropped comes back unlabeled, just outside the
    image, so that the loader marks it as not visible. Reflect padding
    adds reflected copies of a row after the row itself, with the same
    id. The first row of each id is the original, so it wins.

    Returns:
        The rows ``[x, y, labeled, v]`` and which of them were dropped.

    Example:
        >>> rows, lost = untag(np.array([[5.0, 6, 1, 1, 2]]), 2)
        >>> rows.tolist(), lost.tolist()
        ([[-1.0, -1.0, 0.0, 0.0], [5.0, 6.0, 1.0, 2.0]], [True, False])
        >>> copies = np.array([[5.0, 6, 1, 0, 2], [9.0, 6, 1, 0, 2]])
        >>> untag(copies, 1)[0].tolist()
        [[5.0, 6.0, 1.0, 2.0]]

    """
    rows = np.asarray(rows, dtype=float).reshape(-1, 5)
    restored = np.tile([-1.0, -1.0, 0.0, 0.0], (n_rows, 1))
    lost = np.ones(n_rows, dtype=bool)
    ids, first = np.unique(rows[:, 3].astype(int), return_index=True)
    restored[ids] = rows[first][:, [0, 1, 2, 4]]
    lost[ids] = False
    return restored, lost


def is_mirrored(probes: np.ndarray) -> bool | None:
    """Whether the probes came out as a reversed triangle.

    Returns:
        ``None`` when the triangle collapsed to a line, or a probe is
        missing, so that the map at the instance is unknown.

    Example:
        >>> is_mirrored(np.array([[10.0, 0], [9, 0], [10, 1]]))
        True
        >>> is_mirrored(np.array([[10.0, 0], [10, 1], [9, 0]]))
        False

    """
    first, second = probes[1] - probes[0], probes[2] - probes[0]
    area = first[0] * second[1] - first[1] * second[0]
    scale = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not np.isfinite(area) or abs(area) <= 1e-6 * scale or scale == 0:
        return None
    return bool(area < 0)


def rename(
    rows: np.ndarray,
    lost: np.ndarray,
    n_keypoints: int,
    n_probes: int,
    mirror_pairs: Sequence[tuple[int, int]],
    places: Sequence[KeypointPosition | None],
) -> np.ndarray:
    """Give each keypoint the name its meaning asks for, and drop probes.

    Args:
        rows: Rows ``[x, y, labeled, v]`` from `untag`.
        lost: Which rows a transform dropped.
        n_keypoints: :math:`K`.
        n_probes: The probe rows that follow each instance.
        mirror_pairs: The mirror partners of the task.
        places: The place of each keypoint in the picture, or empty.

    Returns:
        Rows ``[x, y, v]``, :math:`K` for each instance.

    Example:
        A mirrored instance swaps its partners:

        >>> rows = np.array(
        ...     [
        ...         [8.0, 0, 1, 2],
        ...         [2, 0, 1, 2],
        ...         [9, 5, 0, 2],
        ...         [8, 5, 0, 2],
        ...         [9, 6, 0, 2],
        ...     ]
        ... )
        >>> lost = np.zeros(5, dtype=bool)
        >>> rename(rows, lost, 2, 3, [(0, 1)], []).tolist()
        [[2.0, 0.0, 2.0], [8.0, 0.0, 2.0]]

    """
    width = n_keypoints + n_probes
    instances = rows.reshape(-1, width, 4)
    lost = lost.reshape(-1, width)
    renamed = instances[:, :n_keypoints].copy()
    for i, instance in enumerate(instances):
        target = list(range(n_keypoints))
        probes_kept = n_probes and not lost[i, n_keypoints:].any()
        if (
            mirror_pairs
            and probes_kept
            and is_mirrored(instance[n_keypoints:, :2])
        ):
            for a, b in mirror_pairs:
                target[a], target[b] = b, a
        if places:
            usable = (instance[:n_keypoints, 2] > 0) & ~lost[i, :n_keypoints]
            symmetry = places_symmetry(
                places, instance[:n_keypoints, :2], usable
            )
            if symmetry is not None:
                moved = renaming(places, symmetry)
                target = [
                    moved[k] if places[k] is not None else target[k]
                    for k in range(n_keypoints)
                ]
        renamed[i, target] = instance[:n_keypoints]
    return renamed[..., [0, 1, 3]].reshape(-1, 3)
