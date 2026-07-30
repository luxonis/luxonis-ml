"""Accumulate comparison results across a whole dataset.

`ComparisonReport` folds many `ComparisonResult` objects into overall and
per-class metrics plus a confusion matrix. Like the matcher it draws nothing;
`luxonis_ml.vizlab.confusion_matrix_figure` renders what it collects.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from luxonis_ml.vizlab.annotations import BBox

from .match import (
    NONE_LABEL,
    ComparisonResult,
    Verdict,
    _fmt_class,
    _metric_row,
)

if TYPE_CHECKING:
    from luxonis_ml.ldf import Detection

    #: A matchable detection: a vizlab box or a full LDF detection tree.
    Detectionish = BBox | Detection

#: Verdict colors, tuned for the dark composite background. Themeable later.


@dataclass
class ComparisonReport:
    """Dataset-wide accumulation of comparison results across many images.

    Feed each image's result to `add`; the report tracks aggregate counts, a
    per-class breakdown, a detection confusion matrix (with a `NONE_LABEL`
    row/column for misses and false alarms), running mean IoU, and the worst
    images by error count. Counts follow COCO: a class error is one false
    positive (its predicted class) plus one false negative (its true class).
    """

    n_images: int = 0
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0
    _class: dict[str, list[int]] = field(default_factory=dict)
    _confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    _iou_sum: float = 0.0
    _n_localized: int = 0
    _worst: list[tuple[int, str]] = field(default_factory=list)

    def add(
        self, result: ComparisonResult, *, name: str | None = None
    ) -> None:
        """Accumulate one image's ``result`` (optionally tagged ``name``)."""
        self.n_images += 1
        self.n_tp += result.n_tp
        self.n_fp += result.n_fp
        self.n_fn += result.n_fn
        n_errors = 0
        for m in result.matches:
            gt = _fmt_class(m.gt) if m.gt is not None else NONE_LABEL
            pred = _fmt_class(m.pred) if m.pred is not None else NONE_LABEL
            if m.verdict is Verdict.TP:
                self._bump(gt, 0)
                self._confuse(gt, gt)
                self._iou_sum += m.iou
                self._n_localized += 1
            elif m.verdict is Verdict.FP:
                self._bump(pred, 1)
                self._confuse(NONE_LABEL, pred)
                n_errors += 1
            elif m.verdict is Verdict.FN:
                self._bump(gt, 2)
                self._confuse(gt, NONE_LABEL)
                n_errors += 1
            else:  # class error
                self._bump(pred, 1)
                self._bump(gt, 2)
                self._confuse(gt, pred)
                self._iou_sum += m.iou
                self._n_localized += 1
                n_errors += 1
        if name is not None and n_errors:
            self._worst.append((n_errors, name))

    def _bump(self, label: str, slot: int) -> None:
        self._class.setdefault(label, [0, 0, 0])[slot] += 1

    def _confuse(self, gt: str, pred: str) -> None:
        self._confusion[(gt, pred)] = self._confusion.get((gt, pred), 0) + 1

    @property
    def precision(self) -> float:
        """``TP / (TP + FP)`` over the whole dataset."""
        denom = self.n_tp + self.n_fp
        return self.n_tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """``TP / (TP + FN)`` over the whole dataset."""
        denom = self.n_tp + self.n_fn
        return self.n_tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of `precision` and `recall`."""
        p, r = self.precision, self.recall
        return 2.0 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        """Mean overlap over every localized match in the dataset."""
        return self._iou_sum / self._n_localized if self._n_localized else 0.0

    def classes(self) -> list[str]:
        """Sorted class names seen (excluding the `NONE_LABEL` background)."""
        names = {n for n in self._class if n != NONE_LABEL}
        for gt, pred in self._confusion:
            names.discard(NONE_LABEL)
            if gt != NONE_LABEL:
                names.add(gt)
            if pred != NONE_LABEL:
                names.add(pred)
        return sorted(names)

    def per_class(self) -> dict[str, dict[str, float | int]]:
        """Per-class ``{precision, recall, tp, fp, fn}`` across the dataset."""
        return {
            name: _metric_row(counts)
            for name, counts in sorted(self._class.items())
            if name != NONE_LABEL
        }

    def confusion_matrix(self) -> tuple[list[str], list[list[int]]]:
        """Return ``(labels, matrix)`` with ``matrix[gt][pred]`` counts.

        ``labels`` are the classes followed by `NONE_LABEL`; a row is the ground
        truth, a column the prediction. The ``(NONE, NONE)`` cell is always zero.
        """
        labels = [*self.classes(), NONE_LABEL]
        matrix = [
            [self._confusion.get((row, col), 0) for col in labels]
            for row in labels
        ]
        return labels, matrix

    def worst(self, n: int = 10) -> list[tuple[int, str]]:
        """Return the ``n`` images with the most errors as ``(count, name)``."""
        return sorted(self._worst, reverse=True)[:n]

    def summary(self) -> dict[str, str | int]:
        """Aggregate metrics as a display-ready mapping."""
        return {
            "images": self.n_images,
            "precision": f"{self.precision:.3f}",
            "recall": f"{self.recall:.3f}",
            "F1": f"{self.f1:.3f}",
            "mean IoU": f"{self.mean_iou:.3f}",
            "TP": self.n_tp,
            "FP": self.n_fp,
            "FN": self.n_fn,
        }
