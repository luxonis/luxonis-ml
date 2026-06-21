from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeGuard, TypeVar

import typeguard
from pydantic import BaseModel, ConfigDict

# When used without installed dependencies
if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


PathType: TypeAlias = str | Path
"""A string or a `pathlib.Path`_ object.

.. _pathlib.Path:
    https://docs.python.org/3/library/pathlib.html#pathlib.Path
"""

PosixPathType: TypeAlias = str | PurePosixPath
"""A string or a `pathlib.PurePosixPath`_ object.

.. _pathlib.PurePosixPath:
    https://docs.python.org/3/library/pathlib.html#pathlib.PurePosixPath
"""


TaskType: TypeAlias = Literal[
    "classification",
    "boundingbox",
    "segmentation",
    "instance_segmentation",
    "keypoints",
    "array",
]


Labels: TypeAlias = dict[str, "np.ndarray"]
"""Dictionary mapping task names to annotations as ``np.ndarray`` values."""


LoaderSingleOutput: TypeAlias = tuple["np.ndarray", Labels]
"""Loader output for a single image source.

The tuple contains one image array and a label dictionary.
"""

LoaderMultiOutput: TypeAlias = tuple[dict[str, "np.ndarray"], Labels]
"""Loader output for one or more named image sources.

The first tuple item maps source names to image arrays, and the second
contains task labels.
"""

LoaderOutput: TypeAlias = LoaderSingleOutput | LoaderMultiOutput
"""Loader output containing image data and labels.

Single-source datasets return `LoaderSingleOutput`; multi-source datasets
return `LoaderMultiOutput`.
"""


RGB: TypeAlias = tuple[int, int, int]
r"""RGB color represented as a tuple of three integers
in the range :math:`\left[0, 255\right]`"""

HSV: TypeAlias = tuple[float, float, float]
"""HSV color represented as a tuple of three floats"""

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


class BaseModelExtraForbid(BaseModel):
    """Base model with extra fields forbidden.

    Attributes:
        model_config: Pydantic model configuration with ``extra`` set to
            ``"forbid"``.

    """

    model_config: ConfigDict = ConfigDict(extra="forbid")


class ConfigItem(BaseModelExtraForbid):
    """Configuration schema for dynamic object instantiation. Typically
    used to instantiate objects stored in registries.

    A dictionary with a name and a dictionary of parameters.

    Attributes:
        name: Name of the object this configuration applies to.
        params: Additional parameters for instantiating the object.

    Example:
        >>> ConfigItem(name="Resize", params={"height": 256}).name
        'Resize'
        >>> ConfigItem(name="Normalize").params
        {}

    """

    name: str

    params: Params = {}


T = TypeVar("T")


def check_type(value: Any, typ: type[T]) -> TypeGuard[T]:
    """Check whether a value has the expected type.

    Note:
        This function acts as a `type guard`_, allowing type checkers
        to narrow the type of a variable when the function returns ``True``.

    Examples:
        >>> check_type("oak", str)
        True
        >>> check_type("oak", int)
        False
        >>> check_type([1, 2, 3], list)
        True

    Args:
        value: Value to check.
        typ: Type to check against.

    Returns:
        ``True`` if ``value`` conforms to ``typ``, otherwise ``False``.

    .. _type guard:
        https://typing.python.org/en/latest/spec/narrowing.html#typeguard

    """
    try:
        typeguard.check_type(value, typ)
    except (typeguard.TypeCheckError, TypeError):
        return False
    return True


def all_not_none(values: Iterable[Any]) -> bool:
    """Check whether all values in a collection are not ``None``.

    Args:
        values: Iterable of values to check.

    Returns:
        ``True`` if all values are not ``None``, otherwise ``False``.

    Examples:
        >>> all_not_none([1, "x", 0])
        True
        >>> all_not_none([1, None, 0])
        False
        >>> all_not_none([])
        True

    """
    return all(v is not None for v in values)


def any_not_none(values: Iterable[Any]) -> bool:
    """Check whether at least one value in a collection is not ``None``.

    Args:
        values: Iterable of values to check.

    Returns:
        ``True`` if at least one value is not ``None``, otherwise ``False``.

    Examples:
        >>> any_not_none([None, "x"])
        True
        >>> any_not_none([None, None])
        False
        >>> any_not_none([])
        False

    """
    return any(v is not None for v in values)
