"""Detection matching, comparison views, and dataset-level reports.

Four layers, each usable on its own:

- `match` — pure matching and scoring; no image, no canvas.
- `render` — turns a `ComparisonResult` into a picture where colour is the verdict.
- `report` — folds many results into dataset-wide metrics.
- `figure` — draws a report's confusion matrix.
"""

from .figure import confusion_matrix_figure
from .match import (
    CLASS_ERROR_COLOR,
    FN_COLOR,
    FP_COLOR,
    NONE_LABEL,
    TP_COLOR,
    ComparisonResult,
    Match,
    Verdict,
    match_detections,
)
from .render import compare
from .report import ComparisonReport

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
