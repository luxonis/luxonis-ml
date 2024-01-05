from typing import Union

from luxonis_ml.enums import LabelType

from .enums import DataLabelType

LDF_VERSION = "1.0.0"

ANNOTATIONS_SCHEMA = {
    "file": str,  # path to file on local disk or object storage
    "class": str,  # any string specifying what the annotation is
    "type": str,  # type of annotation/label
    "value": Union[str, list, int, float, bool, tuple],  # the actual annotation
}

ANNOTATION_TYPES = [val.value for val in list(DataLabelType.__members__.values())]
LABEL_TYPES = [val.value for val in list(LabelType.__members__.values())]
