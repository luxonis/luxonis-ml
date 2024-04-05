from enum import Enum


class DataType(Enum):
    """Represents all existing data types used in i/o streams of the model."""

    INT8 = "int8"
    INT32 = "int32"
    UINT8 = "uint8"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    NV12 = "NV12"
