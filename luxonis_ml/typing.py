from pathlib import Path
from typing import TypedDict, Union

from pydantic.config import JsonDict
from typing_extensions import TypeAlias

PathType: TypeAlias = Union[str, Path]


class ConfigItem(TypedDict):
    name: str
    params: JsonDict
