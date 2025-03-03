from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Dict, Literal, Mapping, Tuple, Union

from pydantic import BaseModel, JsonValue
from typing_extensions import TypeAlias

# When used without installed dependencies
with suppress(ImportError):
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

Params: TypeAlias = Mapping[str, JsonValue]
"""A JSON-like dictionary of additioanl parameters."""


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
