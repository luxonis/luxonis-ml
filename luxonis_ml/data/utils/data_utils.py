from typing import Any, Iterable

import numpy as np


def check_arrays(values: Iterable[Any]) -> None:
    """Checks whether paths to numpy arrays are valid. This checks that th file exists
    and is readable by numpy.

    @type values: List[Any]
    @param values: A list of paths to numpy arrays.
    @rtype: NoneType
    @return: None
    """

    def _check_valid_array(path: str) -> bool:
        try:
            np.load(path)
            return True
        except Exception:
            return False

    for value in values:
        if not isinstance(value, str):
            raise Exception(
                f"Array value {value} must be a path to a numpy array (.npy)"
            )
        if not _check_valid_array(value):
            raise Exception(f"Array at path {value} is not a valid numpy array (.npy)")
