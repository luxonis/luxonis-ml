from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import BarContainer


def _prepare_class_data(
    task_data: List[Dict[str, Any]],
) -> Tuple[List[str], List[int]]:
    """Extracts class names and counts from task_data."""
    classes = [x["class_name"] for x in task_data]
    counts = [x["count"] for x in task_data]
    return classes, counts


def _annotate_bars(ax: plt.Axes, bars: BarContainer) -> None:
    """Adds numeric labels above bars."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_class_distribution(
    ax: plt.Axes, task_type: str, task_data: List[Dict[str, Any]]
) -> None:
    """Plots a bar chart of class distribution.

    @type ax: plt.Axes
    @param ax: The axis to plot on.
    @type task_type: str
    @param task_type: The type of task.
    @type task_data: List[Dict[str, Any]]
    @param task_data: The task data to plot.
    """
    if not task_data:
        ax.axis("off")
        ax.set_title(f"{task_type} Class Distribution (None)", fontsize=12)
        return

    classes, counts = _prepare_class_data(task_data)
    num_classes = len(classes)
    bar_width = 1 / (num_classes**0.1) if num_classes else 1
    bars = ax.bar(classes, counts, width=bar_width, color="royalblue")

    # Set plot properties
    if counts:
        ax.set_ylim(top=max(counts) * 1.15)
    ax.set_title(f"{task_type} Class Distribution", fontsize=12, pad=15)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="x", rotation=90)
    ax.margins(x=0.01)

    _annotate_bars(ax, bars)


def _prepare_heatmap_data(
    heatmap_data: Optional[List[List[float]]],
) -> np.ndarray:
    """Converts heatmap_data to a normalized NumPy array."""
    matrix = np.array(heatmap_data, dtype=np.float32)
    max_val = matrix.max()
    return matrix / max_val if max_val > 0 else matrix


def plot_heatmap(
    ax: plt.Axes,
    fig: plt.Figure,
    task_type: str,
    heatmap_data: Optional[List[List[float]]],
) -> None:
    """ " Plots a heatmap of heatmap_data.

    @type ax: plt.Axes
    @param ax: The axis to plot on.
    @type fig: plt.Figure
    @param fig: The figure to plot on.
    @type task_type: str
    @param task_type: The type of task.
    @type heatmap_data: Optional[List[List[float]]]
    @param heatmap_data: The heatmap data to plot.
    """
    if heatmap_data is None:
        ax.axis("off")
        ax.set_title(f"{task_type} Heatmap (None)", fontsize=12)
        return

    matrix = _prepare_heatmap_data(heatmap_data)
    im = ax.imshow(matrix, cmap="viridis", extent=[0, 1, 0, 1], vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Relative Annotation Density")
    ax.set_title(f"{task_type} Heatmap", fontsize=12)
