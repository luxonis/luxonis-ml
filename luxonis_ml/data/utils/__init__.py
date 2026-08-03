"""Utility helpers shared by the data package.

This package collects public helper APIs used by dataset creation, parsing,
loading, exporting, validation, and visualization. The utilities are grouped by
the part of the data workflow they support:

.. list-table:: Utility groups
   :header-rows: 1

   * - Group
     - Public APIs
   * - Task keys
     - `task_is_metadata`, `split_task`, `get_task_name`, `get_task_type`,
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
     - Label and image visualization (including dataset-health charts) is owned
       by `luxonis_ml.vizlab`.
   * - Equivalence
     - LDF equivalence checks.

The task-key helpers follow the same convention as `LuxonisLoader`: labels are
addressed by ``"task_name/task_type"`` and metadata labels use
``"task_name/metadata/key"`` or ``"metadata/key"`` when no task name is
present.
"""

from .data_utils import (
    find_duplicates,
    get_class_distributions,
    get_class_heatmaps,
    get_duplicates_info,
    get_heatmap_statistics,
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
from .remote_file_downloader import (
    RemoteFileDownloader,
    download_remote_file,
)
from .task_utils import (
    get_task_name,
    get_task_type,
    split_task,
    task_is_metadata,
    task_type_iterator,
)

__all__ = [
    "BucketStorage",
    "BucketType",
    "COCOFormat",
    "ImageType",
    "MediaType",
    "ParquetFileManager",
    "ParquetRecord",
    "ParserIssue",
    "ParserIssueMessage",
    "RemoteFileDownloader",
    "UpdateMode",
    "download_remote_file",
    "find_duplicates",
    "get_class_distributions",
    "get_class_heatmaps",
    "get_duplicates_info",
    "get_heatmap_statistics",
    "get_heatmaps",
    "get_missing_annotations",
    "get_task_name",
    "get_task_type",
    "infer_task",
    "merge_uuids",
    "rgb_to_bool_masks",
    "split_task",
    "task_is_metadata",
    "task_type_iterator",
    "warn_on_duplicates",
]
