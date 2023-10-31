import numpy as np
import os
import uuid
from pathlib import Path

# from luxonis_ml.data.utils.exceptions import *
import typing
from typing import Dict, List, Union, Optional, Any
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
    else:
        _check_polyline_or_keypoints(value)


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


def _check_polyline_or_keypoints(value: Any) -> None:
    if not isinstance(value, list):
        raise Exception(f"Polyline or keypoints must be a list")
    for coord in value:
        if not isinstance(coord, list):
            raise Exception(f"Coordinate {coord} must be a list")
        if len(coord) != 2:
            raise Exception(f"Coordinate {coord} must be length 2")
        for pnt in coord:
            if not isinstance(pnt, int) and not isinstance(pnt, float):
                raise Exception(
                    f"Polyline or keypoints point {pnt} must be an int or float"
                )
