import json
import typing
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import pycocotools.mask as mask_util
from typeguard import TypeCheckError, check_type

from .constants import ANNOTATIONS_SCHEMA as schema


def check_annotation(data: Dict) -> None:
    """Checks whether annotations match the expected format. Throws an exception if
    there is a formatting error.

    @type data: Dict
    @param data: A dictionary representing annotations, mapping annotation types to
        values.
    @rtype: NoneType
    @return: None
    """

    if len(schema.keys()) != len(data.keys()) or set(schema.keys()) != set(data.keys()):
        raise Exception(
            f"Given keys {data.keys()} do not match annotations schema {schema.keys()}"
        )

    for key in schema:
        origin = typing.get_origin(schema[key])
        if origin is not None:
            if origin == Union:
                check = False
                for typ in typing.get_args(schema[key]):
                    if isinstance(data[key], typ):
                        check = True
                if not check:
                    raise Exception(
                        f"Found type {type(data[key])} for key '{key}' but expected: {typing.get_args(schema[key])}"
                    )
            else:
                raise NotImplementedError
        elif not isinstance(data[key], schema[key]):
            raise Exception(
                f"Found type {type(data[key])} for key '{key}' but expected: {schema[key]}"
            )

    typ = data["type"]
    value = data["value"]

    if typ == "classification":
        _check_value_type(typ, value, bool)
    elif typ == "label":
        _check_value_type(typ, value, Union[str, int, float, bool])
    elif typ == "box":
        _check_value_type(
            typ,
            value,
            Tuple[
                Union[int, float],
                Union[int, float],
                Union[int, float],
                Union[int, float],
            ],
        )
    elif typ == "polyline":
        _check_value_type(
            typ,
            value,
            List[Tuple[Union[int, float], Union[int, float]]],
        )
    elif typ == "segmentation":
        _check_value_type(
            typ,
            value,
            Tuple[int, int, Union[List[int], bytes]],
        )
    elif typ == "keypoints":
        _check_value_type(
            typ,
            value,
            List[Tuple[Union[int, float], Union[int, float], int]],
        )


def check_arrays(values: List[Any]) -> None:
    """Checks whether paths to numpy arrays are valid. This checks that th file exists
    and is readable by numpy.

    @type values: List[Any]
    @param values: A list of paths to numpy arrays.
    @rtype: NoneType
    @return: None
    """

    for value in values:
        _check_array(value)


def _check_value_type(name: str, value: Any, typ: Any) -> None:
    """Checks if a value is of a given type, and raises a TypeError if not."""
    try:
        check_type(value, typ)
    except TypeCheckError as e:
        raise TypeError(f"Value {value} for key {name} is not of type {typ}") from e


def _check_array(value: Any) -> None:
    if not isinstance(value, str):
        raise Exception(f"Array value {value} must be a path to a numpy array (.npy)")
    if not _check_valid_array(value):
        raise Exception(f"Array at path {value} is not a valid numpy array (.npy)")


def _check_valid_image(path: str) -> bool:
    try:
        image = cv2.imread(path)
        return image is not None
    except Exception:
        return False


def _check_valid_array(path: str) -> bool:
    try:
        np.load(path)
        return True
    except Exception:
        return False


def transform_segmentation_value(
    value: Tuple[int, int, Union[bytes, List[int]]],
) -> str:
    """Transforms a segmentation in RLE format to the format stored by LuxonisDataset.
    The format recognized by LuxonisDataset is still RLE, but a dumped JSON string of
    height, width, and compressed counts.

    @type value: Tuple[int, int, Union[bytes, List[int]]]
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
