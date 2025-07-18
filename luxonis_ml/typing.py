from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeGuard, TypeVar

import typeguard
from pydantic import BaseModel

# When used without installed dependencies
if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


PathType: TypeAlias = str | Path
"""A string or a `pathlib.Path` object."""

PosixPathType: TypeAlias = str | PurePosixPath
"""A string or a `pathlib.PurePosixPath` object."""


TaskType: TypeAlias = Literal[
    "classification",
    "boundingbox",
    "segmentation",
    "instance_segmentation",
    "keypoints",
    "array",
]


Labels: TypeAlias = dict[str, "np.ndarray"]
"""Dictionary mappping task names to the annotations as C{np.ndarray}"""


LoaderSingleOutput: TypeAlias = tuple["np.ndarray", Labels]
"""C{LoaderSingleOutput} is a tuple containing a single image as a
C{np.ndarray} and a dictionary of task group names and their annotations
as L{Labels}."""

LoaderMultiOutput: TypeAlias = tuple[dict[str, "np.ndarray"], Labels]
"""C{LoaderMultiOutput} is a tuple containing a dictionary mapping image
names to C{np.ndarray} and a dictionary of task group names and their
annotations as L{Labels}."""

LoaderOutput: TypeAlias = LoaderSingleOutput | LoaderMultiOutput
"""C{LoaderOutput} is a tuple containing either a single image as a
C{np.ndarray} or a dictionary mapping image names to C{np.ndarray},
along with a dictionary of task group names and their annotations as
L{Annotations}."""


RGB: TypeAlias = tuple[int, int, int]

HSV: TypeAlias = tuple[float, float, float]

Color: TypeAlias = str | int | RGB
"""Color type alias.

Can be either a string (e.g. "red", "#FF5512"),  a tuple of RGB values,
or a single value (in which case it is interpreted as a grayscale
value).
"""

PrimitiveType: TypeAlias = str | int | float | bool | None
"""Primitive types in Python."""

# To avoid infinite recursion
if TYPE_CHECKING:  # pragma: no cover
    ParamValue: TypeAlias = (
        Mapping[PrimitiveType, "ParamValue"]
        | Sequence["ParamValue"]
        | PrimitiveType
    )
else:
    ParamValue: TypeAlias = Any

Params: TypeAlias = dict[str, ParamValue]
"""A keyword dictionary of additional parameters.

Usually loaded from a YAML file.
"""

Kwargs: TypeAlias = dict[str, Any]
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


def check_type(value: Any, type_: type[T]) -> TypeGuard[T]:
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
