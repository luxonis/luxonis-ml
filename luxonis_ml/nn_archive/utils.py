import re
import tarfile
from pathlib import Path
from typing import List, Union

from luxonis_ml.utils.filesystem import PathType


def is_nn_archive(path: PathType) -> bool:
    """Check if the given path is a valid NN archive file.

    @type path: PathType
    @param path: Path to the file to check.
    @rtype: bool
    @return: True if the file is a valid NN archive file, False otherwise.
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


def parse_layout(layout: Union[str, List[str]]) -> List[str]:
    """Validates and parses layout.

    Layout can contain any number of letters optionally followed by numbers and
    must not contain any duplicate values.
    If the letter 'N' is included, it must be the first letter.

    Examples of correct layouts:
        - 'NCHW' -> ['N', 'C', 'H', 'W']
        - 'NC' -> ['N', 'C']
        - 'NHWC' -> ['N', 'H', 'W', 'C']
        - 'C1C2HW' -> ['C1', 'C2', 'H', 'W']

    Examples of incorrect layouts:
        - 'NCHH' (duplicate letter)
        - 'CHWN' (N is not the first letter)
        - '1CHW' (number is not following a letter)

    @type layout: Union[str, List[str]]
    @param layout: Either a string or a list representation of the layout.

    @rtype: List[str]
    @return: List representation of the layout.
    """

    if isinstance(layout, list):
        layout = "".join(layout)

    layout = layout.upper()
    if not re.match(r"^([A-Z]\d*)+$", layout):
        raise ValueError(
            f"Invalid layout: {layout}. "
            "Layout must only contain letters optionally followed by numbers."
        )
    list_layout = re.findall(r"[A-Z]\d*", layout)

    if len(list_layout) != len(set(list_layout)):
        raise ValueError("Layout must not contain any duplicate values.")

    if "N" in list_layout and list_layout[0] != "N":
        raise ValueError("N (batch size) must be the first letter if included.")

    return list_layout
