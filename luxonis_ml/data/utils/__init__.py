"""Utility helpers shared by the data package.

This package collects public helper APIs used by dataset creation, parsing,
loading, exporting, validation, and visualization. The utilities are grouped by
the part of the data workflow they support:

.. list-table:: Utility groups
   :header-rows: 1

   * - Group
     - Public APIs
   * - Task keys
     - `task_is_label`, `split_task`, `get_task_name`, `get_task_type`,
       and `task_type_iterator` parse and filter ``"task_name/task_type"``
       labels.
   * - Storage and parser enums
     - `BucketStorage`, `BucketType`, `MediaType`, `ImageType`,
       `UpdateMode`, `ParserIssue`, and `ParserIssueMessage`.
   * - Dataframe and parquet helpers
     - `ParquetFileManager`, `ParquetRecord`, duplicate detection, class
       distributions, missing-annotation summaries, heatmaps, and UUID
       merging.
   * - Remote media
     - `RemoteFileDownloader` and `download_remote_file` copy supported
       remote files to local paths and validate image inputs.
   * - Visualization
     - `visualize`, color-map helpers, image concatenation, augmentation
       footers, and dataset-statistic plots.
   * - Equivalence and augmentation inspection
     - LDF equivalence checks and `AugmentationsCollector` summaries for
       configured augmentation pipelines.

The task_is_label follow the same convention as `LuxonisLoader`: labels are
addressed by ``"task_name/task_type"`` and custom labels use
``"{task_name}/labels/{key}"`` or ``"labels/{key}"`` when no task name is
present.
"""

from .augmentations_collector import AugmentationsCollector
from .data_utils import (
    find_duplicates,
    get_class_distributions,
    get_duplicates_info,
    get_heatmaps,
    get_missing_annotations,
    infer_task,
    merge_uuids,
    rgb_to_bool_masks,
    warn_on_duplicates,
)
from .enums import (
    BucketStorage,
    BucketType,
    COCOFormat,
    ImageType,
    MediaType,
    ParserIssue,
    ParserIssueMessage,
    UpdateMode,
)
from .parquet import ParquetFileManager, ParquetRecord
from .plot_utils import plot_class_distribution, plot_heatmap
from .remote_file_downloader import (
    RemoteFileDownloader,
    download_remote_file,
)
from .task_utils import (
    get_task_name,
    get_task_type,
    split_task,
    task_is_label,
    task_type_iterator,
)
from .visualizations import (
    ColorMap,
    add_augmentation_footer,
    concat_images,
    create_text_image,
    distinct_color_generator,
    visualize,
)

__all__ = [
    "AugmentationsCollector",
    "BucketStorage",
    "BucketType",
    "COCOFormat",
    "ColorMap",
    "ImageType",
    "MediaType",
    "ParquetFileManager",
    "ParquetRecord",
    "ParserIssue",
    "ParserIssueMessage",
    "RemoteFileDownloader",
    "UpdateMode",
    "add_augmentation_footer",
    "concat_images",
    "create_text_image",
    "distinct_color_generator",
    "download_remote_file",
    "find_duplicates",
    "get_class_distributions",
    "get_duplicates_info",
    "get_heatmaps",
    "get_missing_annotations",
    "get_task_name",
    "get_task_type",
    "infer_task",
    "merge_uuids",
    "plot_class_distribution",
    "plot_heatmap",
    "rgb_to_bool_masks",
    "split_task",
    "task_is_label",
    "task_type_iterator",
    "visualize",
    "warn_on_duplicates",
]
