import numpy as np
import os
import uuid
import cv2
import json
import pycocotools.mask as mask_util
from typeguard import check_type, TypeCheckError

# from luxonis_ml.data.utils.exceptions import *
import typing
from typing import Dict, List, Union, Any, Tuple
from .constants import ANNOTATIONS_SCHEMA as schema


def generate_hashname(filepath: str) -> Tuple[str, str]:
    """Finds the UUID generated ID for a local file."""

    # Read the contents of the file
    with open(filepath, "rb") as file:
        file_contents = file.read()

    # TODO: check for a corrupted image by handling cv2.imread

    # Generate the UUID5 based on the file contents and the NAMESPACE_URL
    file_hash_uuid = uuid.uuid5(uuid.NAMESPACE_URL, file_contents.hex())

    return str(file_hash_uuid) + os.path.splitext(filepath)[1], str(file_hash_uuid)


def check_annotation(data: Dict) -> None:
    """Throws an exception if the input data does not match the expected annotations
    schema."""

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
        check_value_type(typ, value, bool)
    elif typ == "label":
        check_value_type(typ, value, Union[str, int, float, bool])
    elif typ == "box":
        check_value_type(
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
        check_value_type(
            typ,
            value,
            List[Tuple[Union[int, float], Union[int, float]]],
        )
    elif typ == "segmentation":
        check_value_type(
            typ,
            value,
            Tuple[int, int, Union[List[int], bytes]],
        )
    elif typ == "keypoints":
        check_value_type(
            typ,
            value,
            List[Tuple[Union[int, float], Union[int, float], int]],
        )


def check_arrays(values: List[Any]) -> None:
    """Throws an exception if a given path to an array is invalid."""

    for value in values:
        _check_array(value)


def check_value_type(name: str, value: Any, typ: Any) -> None:
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
    height, width, counts = value
    if isinstance(counts, bytes):
        return json.dumps((height, width, counts.decode("utf-8")))

    rle = {"counts": counts, "size": [height, width]}
    rle = mask_util.frPyObjects(rle, height, width)
    return json.dumps((rle["size"][0], rle["size"][1], rle["counts"].decode("utf-8")))  # type: ignore
