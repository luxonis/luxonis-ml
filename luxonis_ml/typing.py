from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import typeguard
from pydantic import BaseModel
from typing_extensions import TypeAlias, TypeGuard

# When used without installed dependencies
if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


PathType: TypeAlias = Union[str, Path]
"""A string or a `pathlib.Path` object."""

PosixPathType: TypeAlias = Union[str, PurePosixPath]
"""A string or a `pathlib.PurePosixPath` object."""


TaskType: TypeAlias = Literal[
    "classification",
    "boundingbox",
    "segmentation",
    "instance_segmentation",
    "keypoints",
    "array",
]


Labels: TypeAlias = Dict[str, "np.ndarray"]
"""Dictionary mappping task names to the annotations as C{np.ndarray}"""


LoaderOutput: TypeAlias = Tuple["np.ndarray", Labels]
"""C{LoaderOutput} is a tuple of an image as a C{np.ndarray} and a
dictionary of task group names and their annotations as
L{Annotations}."""

RGB: TypeAlias = Tuple[int, int, int]

HSV: TypeAlias = Tuple[float, float, float]

Color: TypeAlias = Union[str, int, RGB]
"""Color type alias.

Can be either a string (e.g. "red", "#FF5512"),  a tuple of RGB values,
or a single value (in which case it is interpreted as a grayscale
value).
"""

PrimitiveType: TypeAlias = Union[str, int, float, bool, None]
"""Primitive types in Python."""

# To avoid infinite recursion
if TYPE_CHECKING:  # pragma: no cover
    ParamValue: TypeAlias = Union[
        Dict[PrimitiveType, "ParamValue"],
        List["ParamValue"],
        PrimitiveType,
    ]
else:
    ParamValue: TypeAlias = Any

Params: TypeAlias = Dict[str, ParamValue]
"""A keyword dictionary of additional parameters.

Usually loaded from a YAML file.
"""

Kwargs: TypeAlias = Dict[str, Any]
"""A keyword dictionary of arbitrary parameters."""


class ConfigItem(BaseModel):
    """Configuration schema for dynamic object instantiation. Typically
    used to instantiate objects stored in registries.

    A dictionary with a name and a dictionary of parameters.

    @type name: str
    @ivar name: The name of the object this configuration applies to.
        Required.
    @type params: Dict[str, JsonDict]
    @ivar params: Additional parameters for instantiating the object.
        Not required.
    """

    name: str

    params: Params = {}


T = TypeVar("T")


def check_type(value: Any, type_: Type[T]) -> TypeGuard[T]:
    """Checks if the value has the correct type.

    @type value: Any
    @param value: The value to check.
    @type type_: Type[K]
    @param type_: The type to check against.
    @rtype: bool
    @return: C{True} if the value has the correct type, C{False}
        otherwise.
    """
    try:
        typeguard.check_type(value, type_)
    except (typeguard.TypeCheckError, TypeError):
        return False
    return True


def all_not_none(values: Iterable[Any]) -> bool:
    """Checks if none of the values in the iterable is C{None}

    @type values: Iterable[Any]
    @param values: An iterable of values
    @rtype: bool
    @return: C{True} if all values are not C{None}, C{False} otherwise
    """
    return all(v is not None for v in values)


def any_not_none(values: Iterable[Any]) -> bool:
    """Checks if at least one value in the iterable is not C{None}

    @type values: Iterable[Any]
    @param values: An iterable of values
    @rtype: bool
    @return: C{True} if at least one value is not C{None}, C{False}
        otherwise
    """
    return any(v is not None for v in values)
