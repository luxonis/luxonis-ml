from pathlib import Path
from typing import Dict, Literal, Tuple, TypedDict, Union

import numpy as np
from pydantic.config import JsonDict
from typing_extensions import TypeAlias

PathType: TypeAlias = Union[str, Path]
"""A string or a `pathlib.Path` object."""


TaskType: TypeAlias = Literal[
    "classification", "boundingbox", "segmentation", "keypoints", "array"
]


Labels: TypeAlias = Dict[str, np.ndarray]
"""Dictionary mappping task names to the annotations as C{np.ndarray}"""


LoaderOutput: TypeAlias = Tuple[np.ndarray, Labels]
"""C{LoaderOutput} is a tuple of an image as a C{np.ndarray} and a
dictionary of task group names and their annotations as
L{Annotations}."""


class ConfigItem(TypedDict):
    name: str
    params: JsonDict
