from enum import Enum


class DataType(Enum):
    """Data types supported by NN Archive input and output streams.

    Attributes:
        INT4: Signed :math:`4`-bit integer.
        INT8: Signed :math:`8`-bit integer.
        INT16: Signed :math:`16`-bit integer.
        INT32: Signed :math:`32`-bit integer.
        INT64: Signed :math:`64`-bit integer.
        UINT4: Unsigned :math:`4`-bit integer.
        UINT8: Unsigned :math:`8`-bit integer.
        UINT16: Unsigned :math:`16`-bit integer.
        UINT32: Unsigned :math:`32`-bit integer.
        UINT64: Unsigned :math:`64`-bit integer.
        FLOAT16: :math:`16`-bit floating point.
        FLOAT32: :math:`32`-bit floating point.
        FLOAT64: :math:`64`-bit floating point.
        BOOLEAN: Boolean value.
        STRING: String value.

    """

    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT4 = "uint4"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOLEAN = "boolean"
    STRING = "string"
