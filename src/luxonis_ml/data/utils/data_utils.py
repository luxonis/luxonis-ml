import numpy as np
import os
import uuid
from pathlib import Path
import cv2
import json
import pycocotools.mask as mask_util

# from luxonis_ml.data.utils.exceptions import *
import typing
from typing import Dict, List, Union, Optional, Any, Tuple
from .constants import ANNOTATIONS_SCHEMA as schema
from .constants import ANNOTATION_TYPES as atypes


def generate_hashname(filepath: str) -> str:
    """Finds the UUID generated ID for a local file"""

    # Read the contents of the file
    with open(filepath, "rb") as file:
        file_contents = file.read()

    # TODO: check for a corrupted image by handling cv2.imread

    # Generate the UUID5 based on the file contents and the NAMESPACE_URL
    file_hash_uuid = uuid.uuid5(uuid.NAMESPACE_URL, file_contents.hex())

    return str(file_hash_uuid) + os.path.splitext(filepath)[1], str(file_hash_uuid)


def check_annotation(data: Dict) -> None:
    """Throws an exception if the input data does not match the expected annotations schema"""

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

    if isinstance(data["type"], str):
        if data["type"] not in atypes:
            raise Exception(
                f"{data['type']} for key '{key}' is not a valid DataLabelType"
            )
        type_string = data["type"]

    value = data["value"]
    if type_string == "classification":
        _check_classification(value)
    elif type_string == "label":
        _check_label(value)
    elif type_string == "box":
        _check_box(value)
    elif type_string == "polyline":
        _check_polyline(value)
    elif type_string == "segmentation":
        _check_segmentation(value)
    elif type_string == "keypoints":
        _check_keypoints(value)


def check_arrays(values: List[Any]) -> None:
    """Throws an exception if a given path to an array is invalid"""

    for value in values:
        _check_array(value)


def _check_classification(value: Any) -> None:
    if not isinstance(value, bool):
        raise Exception(f"Classification {value} must be a bool (True/False)")


def _check_label(value: Any) -> None:
    if (
        not isinstance(value, str)
        and not isinstance(value, int)
        and not isinstance(value, float)
        and not isinstance(value, bool)
    ):
        raise Exception(f"Label {value} must be a string, int, or float")


def _check_box(value: Any) -> None:
    if not isinstance(value, list):
        raise Exception(f"Box {value} must be a list")
    if len(value) != 4:
        raise Exception(f"Box {value} must be of length 4")
    for pnt in value:
        if not isinstance(pnt, int) and not isinstance(pnt, float):
            raise Exception(f"Box point {pnt} must be an int or float")


def _check_polyline(value: Any) -> None:
    if not isinstance(value, list):
        raise Exception(f"Polyline must be a list")
    for coord in value:
        if not isinstance(coord, list):
            raise Exception(f"Coordinate {coord} must be a list")
        if len(coord) != 2:
            raise Exception(f"Coordinate {coord} must be length 2")
        for pnt in coord:
            if not isinstance(pnt, int) and not isinstance(pnt, float):
                raise Exception(f"Polyline point {pnt} must be an int or float")


def _check_segmentation(value: Any) -> None:
    message = f"Segmentation {value} must be an RLE representation of type Tuple[int, int List[int]] of (height, width, counts)"
    if not isinstance(value, tuple):
        raise Exception(message)
    if not isinstance(value[0], int):
        raise Exception(message)
    if not isinstance(value[1], int):
        raise Exception(message)
    if not isinstance(value[2], list):
        raise Exception(message)
    for val in value[2]:
        if not isinstance(val, int):
            raise Exception(message)


def _check_keypoints(value: Any) -> None:
    if not isinstance(value, list):
        raise Exception(f"Keypoints must be a list")
    for coord in value:
        if not isinstance(coord, list):
            raise Exception(f"Coordinate {coord} must be a list")
        if len(coord) != 3:
            raise Exception(f"Coordinate {coord} must be length 3")
        for pnt in coord:
            if not isinstance(pnt, int) and not isinstance(pnt, float):
                raise Exception(f"Keypoints point {pnt} must be an int or float")


def _check_array(value: Any) -> None:
    if not isinstance(value, str):
        raise Exception(f"Array value {value} must be a path to a numpy array (.npy)")
    if not _check_valid_array(value):
        raise Exception(f"Array at path {value} is not a valid numpy array (.npy)")


def _check_valid_image(path: str) -> bool:
    try:
        image = cv2.imread(path)
        return image is not None
    except:
        return False


def _check_valid_array(path: str) -> bool:
    try:
        np.load(path)
        return True
    except:
        return False


def transform_segmentation_value(value: Tuple[int, int, List[int]]) -> str:
    height, width, counts = value
    rle = {"counts": counts, "size": [height, width]}
    rle = mask_util.frPyObjects(rle, height, width)
    value = (rle["size"][0], rle["size"][1], rle["counts"].decode("utf-8"))
    return json.dumps(value)
