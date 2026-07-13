import tarfile
from pathlib import Path

from luxonis_ml.typing import PathType


def is_nn_archive(path: PathType) -> bool:
    """Check whether a path points to a valid NN Archive.

    A valid archive must be a tar file and contain a top-level
    ``config.json`` member.

    Args:
        path: Path to the file to check.

    Returns:
        ``True`` if the file is a valid NN Archive, otherwise ``False``.

    """
    path = Path(path)

    if not path.is_file():
        return False

    if not tarfile.is_tarfile(path):
        return False

    with tarfile.open(path, "r") as tar:
        if "config.json" not in tar.getnames():
            return False

    return True


def infer_layout(shape: list[int]) -> str:
    """Infer a layout code for a tensor shape.

    The function recognizes common image tensor layouts. For other shapes,
    it uses the first available letters starting at ``C``.

    Args:
        shape: Tensor shape to infer from.

    Returns:
        Layout code matching the number of dimensions in ``shape``.

    Raises:
        ValueError: If the shape has too many dimensions for automatic
            layout inference.

    Example:
        >>> infer_layout([1, 3, 256, 256])
        'NCHW'
        >>> infer_layout([256, 256, 3])
        'HWC'
        >>> infer_layout([1, 19, 7, 8])
        'NCDE'

    """
    layout = []
    i = 0
    if shape[0] == 1:
        layout.append("N")
        i += 1
    if len(shape) - i == 3:
        if shape[i] < shape[i + 1] and shape[i] < shape[i + 2]:
            return "".join([*layout, "C", "H", "W"])
        if shape[-1] < shape[-2] and shape[-1] < shape[-3]:
            return "".join([*layout, "H", "W", "C"])
    i = 0
    while len(layout) < len(shape):
        # Starting with "C" for more sensible defaults
        letter = chr(ord("A") + i + 2)
        if ord(letter) > ord("Z"):
            raise ValueError(
                f"Too many dimensions ({len(shape)}) for automatic layout."
            )

        if letter not in layout:
            layout.append(letter)
        i += 1
    return "".join(layout)
