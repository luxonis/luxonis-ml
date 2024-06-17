from typing import Any, Iterable, List, Tuple
import warnings

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


def validate_text_value(
    value: str,
    classes: List[str]
) -> Tuple[str, int]:
    """Validates a text value to only contain valid classes.

    @param classes: valid character classes.
    @type classes:
    @type value: str
    @param value: text value to validate.
    @rtype: str
    @return: same input value if it's valid, raises a Warning and cleans/ignores invalid characters.
    """
    clean_value = ""
    text_value, max_len = value
    for char in text_value:
        if char in classes:
            clean_value += char
        else:
            warnings.warn(
                f"Text annotations contain invalid char ({char}): default behaviour is to exclude undefined classes, "
                f"make sure to add it to your dataset classes."
            )
    return clean_value, max_len


# def encode_text_value(
#     value: str,
#     max_len: int,
#     classes: List[str]
# ) -> [np.ndarray, int, int]:
#
#     text = value
#     text_label = np.zeros(max_len)
#     for char_idx, char in enumerate(text):
#         cls = classes.index(char)
#         text_label[char_idx] = cls
#
#     return [text_label, len(text), max_len]
