from enum import Enum


class DataType(Enum):
    """Represents all existing data types used in i/o streams of the model."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    UFXP8 = "ufxp8"
    UFXP16 = "ufxp16"
    UFXP32 = "ufxp32"
    UFXP64 = "ufxp64"
    FXP8 = "fxp8"
    FXP16 = "fxp16"
    FXP32 = "fxp32"
    FXP64 = "fxp64"
    BOOLEAN = "boolean"
    STRING = "string"
