from luxonis_ml.enums import LabelType
from .enums import DataLabelType
from typing import List, Dict, Union

LDF_VERSION = "1.0.0"

ANNOTATIONS_SCHEMA = {
    "file": str,  # path to file on local disk or object storage
    "class": str,  # any string specifying what the annotation is
    "type": str,  # type of annotation/label
    "value": Union[str, list, int, float, bool],  # the actual annotation
    # "children": List[str],
    # "parent": List[str],
}

ANNOTATION_TYPES = [val.value for val in list(DataLabelType.__members__.values())]
LABEL_TYPES = [val.value for val in list(LabelType.__members__.values())]

# PARQUET_ANNOTATIONS_SCHEMA = {
#     "id": str,
#     "instance_id": str,
#     "file": str,
#     "class": str,
#     "type": Union[str, DataLabelType],
#     "value": Union[str, list, int, float],
#     # "children": List[str],
#     # "parent": List[str],
# }
