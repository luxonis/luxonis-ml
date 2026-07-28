"""Detection matching, comparison views, and dataset-level reports."""

from .core import (
    CLASS_ERROR_COLOR,
    FN_COLOR,
    FP_COLOR,
    NONE_LABEL,
    TP_COLOR,
    ComparisonReport,
    ComparisonResult,
    Match,
    Verdict,
    compare,
    confusion_matrix_figure,
    match_detections,
)

__all__ = [
    "CLASS_ERROR_COLOR",
    "FN_COLOR",
    "FP_COLOR",
    "NONE_LABEL",
    "TP_COLOR",
    "ComparisonReport",
    "ComparisonResult",
    "Match",
    "Verdict",
    "compare",
    "confusion_matrix_figure",
    "match_detections",
]
