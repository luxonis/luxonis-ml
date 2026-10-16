"""Migration of datasets that an older LDF version wrote."""

import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Final, Literal, cast, overload

import polars as pl
from loguru import logger
from semver.version import Version
from typing_extensions import TypedDict

from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.data.utils.parquet import DEFAULT_METADATA
from luxonis_ml.ldf import KeypointMetadata, KeypointSpec

from .metadata import Metadata

LDF_1_0_0_TASKS: Final[set[str]] = {
    "classification",
    "segmentation",
    "boundingbox",
    "keypoints",
    "array",
}

LDF_1_0_0_TASK_TYPES: Final[dict[str, str]] = {
    "BBoxAnnotation": "boundingbox",
    "ClassificationAnnotation": "classification",
    "PolylineSegmentationAnnotation": "segmentation",
    "RLESegmentationAnnotation": "segmentation",
    "MaskSegmentationAnnotation": "segmentation",
    "KeypointAnnotation": "keypoints",
    "ArrayAnnotation": "array",
}


class LDF_1_0_0_Skeleton(TypedDict):
    """Keypoint skeleton as LDF ``1.0.0`` stored it.

    LDF 2.x stores the same layout. `migrate_metadata` turns it into a
    `luxonis_ml.ldf.KeypointMetadata`.

    Attributes:
        labels: Keypoint names in index order.
        edges: Keypoint graph edges as :math:`0`-based index pairs.

    """

    labels: list[str]
    edges: list[tuple[int, int]]


class LDF_1_0_0_MetadataDict(TypedDict):
    """Metadata dictionary used by LDF ``1.0.0``.

    Attributes:
        source: Source metadata dictionary.
        ldf_version: LDF version string.
        classes: Class names keyed by old task names.
        tasks: Task types keyed by old task names.
        skeletons: Keypoint skeletons keyed by task name.
        categorical_encodings: Categorical metadata encodings.
        metadata_types: Metadata value types.

    """

    source: dict[str, Any]
    ldf_version: str
    classes: dict[str, list[str]]
    tasks: dict[str, list[str]]
    skeletons: dict[str, LDF_1_0_0_Skeleton]
    categorical_encodings: dict[str, dict[str, int]]
    metadata_types: dict[str, Literal["float", "int", "str", "Category"]]


@overload
def migrate_dataframe(df: pl.LazyFrame) -> pl.LazyFrame: ...


@overload
def migrate_dataframe(df: pl.DataFrame) -> pl.DataFrame: ...


def migrate_dataframe(
    df: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:  # pragma: no cover
    """Migrate LDF ``1.0.0`` annotation rows to the current columns.

    Args:
        df: Annotation rows in the LDF ``1.0.0`` layout.

    Returns:
        The rows with the current column names, task names and task
        types, as the same frame type as ``df``.

    """
    return (
        df.rename({"class": "class_name"})
        .with_columns(
            pl.when(pl.col("task").is_in(LDF_1_0_0_TASKS))
            .then(pl.lit("detection"))
            .otherwise(pl.col("task"))
            .alias("task_name")
        )
        .with_columns(
            pl.when(pl.col("type") == "BBoxAnnotation")
            .then(pl.lit("boundingbox"))
            .when(pl.col("type") == "ClassificationAnnotation")
            .then(pl.lit("classification"))
            .when(
                pl.col("type").is_in(
                    [
                        "PolylineSegmentationAnnotation",
                        "RLESegmentationAnnotation",
                        "MaskSegmentationAnnotation",
                    ]
                )
            )
            .then(pl.lit("segmentation"))
            .when(pl.col("type") == "KeypointAnnotation")
            .then(pl.lit("keypoints"))
            .when(pl.col("type") == "ArrayAnnotation")
            .then(pl.lit("array"))
            .otherwise(pl.col("type"))
            .alias("task_type")
        )
        .with_columns(pl.lit("image").alias("source_name"))
        .with_columns(pl.lit(DEFAULT_METADATA).alias("sample_metadata"))
        .select(
            [
                "file",
                "source_name",
                "task_name",
                "class_name",
                "instance_id",
                "task_type",
                "annotation",
                "sample_metadata",
                "uuid",
            ]
        )
    )


def migrate_metadata(
    metadata: dict[str, Any],
    df: pl.LazyFrame | None,
    version: Version,
) -> Metadata:
    """Migrate stored metadata to the schema this version reads.

    LDF 2.x and ``1.0.0`` store the keypoints of a task as a ``skeletons``
    entry with a list of labels and index edges. It becomes the
    `KeypointMetadata` of the task. LDF 2.x metadata gets the current
    version stamp, and `migrate_keypoint_rows` keys its keypoint rows by
    name. LDF ``1.0.0`` metadata keeps its stamp, and `migrate_dataframe`
    converts its annotation rows on every load.

    Args:
        metadata: Metadata dictionary as it was stored.
        df: Optional annotation dataframe. LDF ``1.0.0`` takes the task
            names of a non-default dataset from it, and LDF 2.x takes the
            keypoint count of each task.
        version: LDF version the metadata was written by.

    Returns:
        Migrated metadata model.

    Raises:
        ValueError: If task inference requires annotation rows but
            ``df`` is ``None``.

    """
    skeletons = metadata.pop("skeletons", {})
    if version.major == 2:
        # The 2.x schema is current apart from the skeletons. On the old
        # stamp, every load would run `migrate_dataframe` on these rows.
        widths = {} if df is None else _keypoint_row_widths(df)
        return Metadata.model_validate(
            {
                **metadata,
                "ldf_version": str(LDF_VERSION),
                "keypoint_metadata": _migrate_skeletons(skeletons, widths),
            }
        )
    migrated = _migrate_ldf_1_0_0(cast(LDF_1_0_0_MetadataDict, metadata), df)
    return migrated.model_copy(
        update={"keypoint_metadata": _migrate_skeletons(skeletons, {})}
    )


def migrate_keypoint_rows(
    df: pl.LazyFrame, keypoint_metadata: Mapping[str, KeypointMetadata]
) -> pl.DataFrame | None:
    """Key the positional keypoint rows of LDF 2.x by name.

    LDF 2.x stores the keypoints of a row as a list in the order of the
    skeleton labels. LDF 3.0 stores them by name.

    Args:
        df: The annotation rows. The check for list rows reads only the
            task type and the annotation of each row.
        keypoint_metadata: The migrated keypoint metadata of each task.

    Returns:
        The migrated rows, or ``None`` if no keypoint row is a list.

    """
    # A shard without annotations stores the column as nulls.
    annotation = pl.col("annotation").cast(pl.String)
    positional = (
        pl.col("task_type") == "keypoints"
    ) & annotation.str.contains(r'^\{"keypoints":\s*\[')
    if not df.select(positional.any()).collect().item():
        return None

    def named(row: dict[str, Any]) -> str | None:
        if row["task_type"] != "keypoints" or row["annotation"] is None:
            return row["annotation"]
        keypoints = json.loads(row["annotation"])["keypoints"]
        if not isinstance(keypoints, list):
            return row["annotation"]
        labels = keypoint_metadata[row["task_name"]].labels
        return json.dumps(
            {"keypoints": dict(zip(labels, keypoints, strict=False))}
        )

    return df.with_columns(
        pl.struct("task_name", "task_type", annotation)
        .map_elements(named, return_dtype=pl.String)
        .alias("annotation")
    ).collect()


def _migrate_skeletons(
    skeletons: Mapping[str, LDF_1_0_0_Skeleton], widths: Mapping[str, int]
) -> dict[str, KeypointMetadata]:
    """Turn stored skeletons into the keypoint metadata of each task.

    A row with more keypoints than the labels names the others by
    position. Repeated names give every keypoint of the task its position
    as its name, and an edge out of range is dropped. Both log a warning.
    """
    migrated = {}
    for task in dict.fromkeys([*skeletons, *widths]):
        skeleton = skeletons.get(task, {})
        labels = skeleton.get("labels", [])
        n_keypoints = max(len(labels), widths.get(task, 0))
        names = labels + [str(i) for i in range(len(labels), n_keypoints)]
        if len(set(names)) < len(names):
            logger.warning(
                f"The skeleton of task '{task}' repeats keypoint names. The "
                f"keypoints get the names 0 to {n_keypoints - 1}."
            )
            names = [str(i) for i in range(n_keypoints)]
        connections: dict[str, list[str]] = {name: [] for name in names}
        for a, b in skeleton.get("edges", []):
            if not (0 <= a < n_keypoints and 0 <= b < n_keypoints):
                logger.warning(
                    f"The skeleton of task '{task}' has the edge ({a}, {b}) "
                    f"for {n_keypoints} keypoints. The edge is dropped."
                )
                continue
            connections[names[a]].append(names[b])
        # A skeleton without names that no keypoint row uses describes no
        # keypoint task. A task with keypoint rows keeps its entry, also
        # when each row is empty.
        if names or task in widths:
            migrated[task] = KeypointMetadata(
                keypoints={
                    name: KeypointSpec(connected_to=targets)
                    for name, targets in connections.items()
                }
            )
    return migrated


def _keypoint_row_widths(df: pl.LazyFrame) -> dict[str, int]:
    """Return the keypoint count of the widest keypoint row of each task."""
    rows = (
        df.filter(pl.col("task_type") == "keypoints")
        # A shard without annotations stores the column as nulls.
        .select("task_name", pl.col("annotation").cast(pl.String))
        .drop_nulls("annotation")
        .collect()
    )
    widths: dict[str, int] = {}
    for task, annotation in rows.iter_rows():
        n_keypoints = len(json.loads(annotation)["keypoints"])
        widths[task] = max(widths.get(task, 0), n_keypoints)
    return widths


def _migrate_ldf_1_0_0(
    metadata: LDF_1_0_0_MetadataDict, df: pl.LazyFrame | None
) -> Metadata:  # pragma: no cover
    """Migrate LDF ``1.0.0`` metadata to the current schema."""
    new_metadata = {}
    old_classes = metadata["classes"]
    if set(old_classes.keys()) <= LDF_1_0_0_TASKS:
        old_class_names = next(iter(old_classes.values()))
        new_metadata["classes"] = {
            "detection": {
                class_name: i for i, class_name in enumerate(old_class_names)
            }
        }
        new_metadata["tasks"] = {"detection": list(old_classes.keys())}
    else:
        if df is None:
            raise ValueError("Cannot migrate when the dataset is empty")
        tasks_df = df.select(["task", "type"]).unique().collect()
        new_classes = defaultdict(dict)
        tasks = defaultdict(list)
        for task_name, task_type in tasks_df.iter_rows():
            new_task_name = task_name
            if task_name in LDF_1_0_0_TASKS:
                new_task_name = "detection"
            tasks[new_task_name].append(LDF_1_0_0_TASK_TYPES[task_type])
            old_class_names = old_classes.get(task_name, [])
            new_classes[new_task_name].update(
                {class_name: i for i, class_name in enumerate(old_class_names)}
            )
        new_metadata["classes"] = dict(new_classes)
        new_metadata["tasks"] = dict(tasks)

    metadata.update(new_metadata)
    return Metadata(**metadata)  # type: ignore
