from pathlib import Path
from typing import Dict, Literal, Tuple, TypedDict, Union

import numpy as np
from pydantic.config import JsonDict
from typing_extensions import NotRequired, Required, TypeAlias

PathType: TypeAlias = Union[str, Path]
"""A string or a `pathlib.Path` object."""


TaskType: TypeAlias = Literal[
    "classification",
    "boundingbox",
    "segmentation",
    "instance_segmentation",
    "keypoints",
    "array",
]


Labels: TypeAlias = Dict[str, np.ndarray]
"""Dictionary mappping task names to the annotations as C{np.ndarray}"""


LoaderOutput: TypeAlias = Tuple[np.ndarray, Labels]
"""C{LoaderOutput} is a tuple of an image as a C{np.ndarray} and a
dictionary of task group names and their annotations as
L{Annotations}."""


class ConfigItem(TypedDict):
    """Configuration dictionary for dynamic object instantiation.
    Typically used to instantiate objects stored in registries.

    A dictionary with a name and a dictionary of parameters.

    @type name: str
    @ivar name: The name of the object this configuration applies to.
        Required.
    @type params: JsonDict
    @ivar params: Additional parameters for instantiating the object.
        Not required.
    """

    name: Required[str]
    params: NotRequired[JsonDict]
