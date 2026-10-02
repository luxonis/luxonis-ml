"""Detection matching, comparison views, and dataset-level reports.

Five layers, each usable on its own:

- `match` — pure matching and scoring; no image, no canvas.
- `render` — turns a `ComparisonResult` into a picture where colour is the verdict.
- `compose` — matches and draws whole paired samples, panel and all.
- `report` — folds many results into dataset-wide metrics.
- `figure` — draws a report's confusion matrix.
"""

from .compose import ComparisonComposer
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
    "ComparisonComposer",
    "ComparisonReport",
    "ComparisonResult",
    "Match",
    "Verdict",
    "compare",
    "confusion_matrix_figure",
    "match_detections",
]
