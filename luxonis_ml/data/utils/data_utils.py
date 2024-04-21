import json
from typing import Any, List, Tuple, Union

import numpy as np
import pycocotools.mask as mask_util
from typing_extensions import TypeAlias

# NOTE: This could be defined elsewhere
Number: TypeAlias = Union[int, float]
"""Alias for a number type, which can be either an integer or a float."""

ClassificationType: TypeAlias = bool
"""Alias for a classification type, which is a boolean."""

LabelType: TypeAlias = Union[str, int, float, bool]
"""Alias for a custom label type, which can be a string, integer, float, or boolean."""

BoxType: TypeAlias = Tuple[Number, Number, Number, Number]
"""Alias for a bounding box type, which is a tuple of (x, y, width, height)."""

SegmentationType: TypeAlias = Tuple[int, int, Union[List[int], bytes]]
"""Alias for a segmentation type in the RLE format, which is a tuple of (height, width,
counts).

`counts` can be encoded bytes or a raw list of integers.
"""

PolylineType: TypeAlias = List[Tuple[Number, Number]]
"""Alias for a polyline type, which is a list of (x, y) points."""

KeypointsType: TypeAlias = List[Tuple[Number, Number, int]]
"""Alias for a keypoints type, which is a list of (x, y, visibility) points.
Visibility is an integer value of the following format:

    - 0: Not visible
    - 1: Occluded
    - 2: Visible

"""

ArrayType: TypeAlias = str
"""Alias for an array type, which is a path to a numpy array (.npy)."""


def check_arrays(values: List[Any]) -> None:
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


def transform_segmentation_value(value: SegmentationType) -> str:
    """Transforms a segmentation in RLE format to the format stored by LuxonisDataset.
    The format recognized by LuxonisDataset is still RLE, but a dumped JSON string of
    height, width, and compressed counts.

    @type value: L{SegmentationType}
    @param value: The segmentation value in RLE format of (height, width, counts).
        Counts can be encoded bytes or a raw list.
    @rtype: str
    @return: A dumped string of the segmentation format recognized by LuxonisDataset.
    """

    height, width, counts = value
    if isinstance(counts, bytes):
        return json.dumps((height, width, counts.decode("utf-8")))

    rle = {"counts": counts, "size": [height, width]}
    rle = mask_util.frPyObjects(rle, height, width)
    return json.dumps((rle["size"][0], rle["size"][1], rle["counts"].decode("utf-8")))  # type: ignore
