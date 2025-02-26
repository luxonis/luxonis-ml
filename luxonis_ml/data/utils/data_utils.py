from collections import defaultdict
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger

from luxonis_ml.data.utils.task_utils import task_is_metadata
from luxonis_ml.typing import RGB


def rgb_to_bool_masks(
    segmentation_mask: np.ndarray,
    class_colors: Dict[str, RGB],
    add_background_class: bool = False,
) -> Iterator[Tuple[str, np.ndarray]]:
    """Helper function to convert an RGB segmentation mask to boolean
    masks for each class.

    Example:

        >>> segmentation_mask = np.array([
        ...     [[0, 0, 0], [255, 0, 0], [0, 255, 0]],
        ...     [[0, 0, 0], [0, 255, 0], [0, 0, 255]],
        ...     ], dtype=np.uint8)
        >>> class_colors = {
        ...     "red": (255, 0, 0),
        ...     "green": (0, 255, 0),
        ...     "blue": (0, 0, 255),
        ... }
        >>> for class_name, mask in rgb_to_bool_masks(
        ...     segmentation_mask,
        ...     class_colors,
        ...     add_background_class=True,
        ... ):
        ...     print(class_name, np.array2string(mask, separator=", "))
        background [[ True, False, False],
                    [ True, False, False]]
        red        [[False,  True, False],
                    [False, False, False]]
        green      [[False, False,  True],
                    [False, True, False]]
        blue       [[False, False, False],
                    [False, False,  True]]

    @type segmentation_mask: npt.NDArray[np.uint8]
    @param segmentation_mask: An RGB segmentation mask where each pixel
        is colored according to the class it belongs to.
    @type class_colors: Dict[str, Tuple[int, int, int]]
    @param class_colors: A dictionary mapping class names to RGB colors.
    @type add_background_class: bool
    @param add_background_class: Whether to add a background class with a mask for all pixels
        that do not belong to any class. The class name will be set to "background".
        The background class will be yielded first. Default is False.
    @rtype: Iterator[Tuple[str, npt.NDArray[np.bool_]]]
    @return: An iterator of tuples where the first element is the class name and
        the second element is a boolean mask for that class.
    """
    color_to_id = {
        tuple(color): i for i, color in enumerate(class_colors.values())
    }

    lookup_table = np.zeros((256, 256, 256), dtype=np.uint8)
    for color, id in color_to_id.items():
        lookup_table[color[0], color[1], color[2]] = id + 1

    segmentation_ids = lookup_table[
        segmentation_mask[:, :, 0],
        segmentation_mask[:, :, 1],
        segmentation_mask[:, :, 2],
    ]

    if add_background_class:
        background_mask = segmentation_ids == 0
        yield "background", background_mask

    for class_name, color in class_colors.items():
        class_id = color_to_id[tuple(color)] + 1
        yield class_name, segmentation_ids == class_id


def infer_task(
    old_task: str,
    class_name: Optional[str],
    current_classes: Dict[str, Dict[str, int]],
) -> str:
    if not hasattr(infer_task, "_logged_infered_classes"):
        infer_task._logged_infered_classes = defaultdict(bool)

    def _log_once(
        cls_: Optional[str], task: str, message: str, level: str = "info"
    ):
        if not infer_task._logged_infered_classes[(cls_, task)]:
            infer_task._logged_infered_classes[(cls_, task)] = True
            getattr(logger, level)(message)

    infered_task = None

    for task, classes in current_classes.items():
        if class_name in classes:
            if infered_task is not None:
                _log_once(
                    class_name,
                    infered_task,
                    f"Class '{class_name}' is ambiguous between "
                    f"tasks '{infered_task}' and '{task}'. "
                    "Task inference failed.",
                    "warning",
                )
                infered_task = None
                break
            infered_task = task
    if infered_task is None:
        _log_once(
            class_name,
            old_task,
            f"Class '{class_name}' doesn't belong to any existing task. "
            f"Autogenerated task '{old_task}' will be used.",
            "info",
        )
    else:
        _log_once(
            class_name,
            infered_task,
            f"Class '{class_name}' infered to belong to task '{infered_task}'",
        )
        return infered_task

    return old_task


def warn_on_duplicates(df: pl.LazyFrame) -> None:
    # Warn on duplicate UUIDs
    uuid_file_pairs = df.select("uuid", "file").unique().collect()

    duplicates = (
        uuid_file_pairs.group_by("uuid")
        .agg(pl.col("file").n_unique().alias("file_count"))
        .filter(pl.col("file_count") > 1)
    )

    if not duplicates.is_empty():
        logger.warning("Found duplicate UUIDs in the dataset:")
        for row in duplicates.iter_rows():
            uuid = row[0]
            files = uuid_file_pairs.filter(pl.col("uuid") == uuid)[
                "file"
            ].to_list()
            logger.warning(f"UUID {uuid} is used in multiple files: {files}")

    # Warn on duplicate annotations
    duplicate_annotation = (
        df.group_by(
            "original_filepath",
            "task_type",
            "annotation",
        )
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .filter(pl.col("annotation") != "{}")
    ).collect()

    for (
        file_name,
        task_type,
        annotation,
        count,
    ) in duplicate_annotation.iter_rows():
        if task_type == "segmentation":
            annotation = "<binary mask>"
        if not task_is_metadata(task_type):
            logger.warning(
                f"File '{file_name}' has the same '{task_type}' annotation "
                f"'{annotation}' repeated {count} times."
            )
