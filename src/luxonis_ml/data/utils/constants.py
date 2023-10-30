from .enums import LabelType
from typing import List, Dict, Union

LDF_VERSION = "1.0.0"

ANNOTATIONS_SCHEMA = {
    "id": str,
    "type": Union[str, LabelType],
    "class": str,
    # "children": List[str],
    # "parent": List[str],
    "value": Union[str, list, int, float],
}

ANNOTATION_TYPES = [val.value for val in list(LabelType.__members__.values())]
