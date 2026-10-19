# pyright: strict
# matplotlib leaves the **kwargs of its Axes methods unannotated. Strict
# mode would report each call, so this rule is off in this file.
# pyright: reportUnknownMemberType=false
"""The figures that the tracker backends draw with matplotlib.

`MLflowBackend` logs `confusion_figure` beside the JSON of a confusion
matrix. ``matplotlib`` comes with the ``mlflow`` extra.

"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure


def confusion_figure(
    matrix: npt.NDArray[Any], labels: Sequence[str], title: str
) -> Figure:
    """Draw a confusion matrix as a heatmap, with the true classes on
    the rows.

    The cells hold the counts when the matrix has at most 20 classes.
    More would not fit.

    Args:
        matrix: The counts, of shape ``(N, N)``.
        labels: The ``N`` class names.
        title: The title above the heatmap.

    Returns:
        The figure, which no ``pyplot`` state holds.

    """
    size = 2 + 0.5 * len(labels)
    figure = Figure(figsize=(size + 1, size), layout="tight")
    axes = figure.subplots()
    axes.imshow(matrix, cmap="Blues")
    ticks = range(len(labels))
    axes.set_xticks(ticks, labels, rotation=30, ha="right")
    axes.set_yticks(ticks, labels)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title(title, fontsize=9)
    if len(labels) <= 20:
        # dark cells get white text
        threshold = matrix.max() / 2
        for (i, j), count in np.ndenumerate(matrix):
            color = "white" if count > threshold else "black"
            axes.text(
                j, i, f"{count:g}", ha="center", va="center", color=color
            )
    return figure
