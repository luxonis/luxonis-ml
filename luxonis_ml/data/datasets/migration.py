from collections import defaultdict
from typing import Any, Final, Literal, TypeVar

import polars as pl
from semver.version import Version
from typing_extensions import TypedDict

from .metadata import Metadata, Skeletons

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


class _LDF_1_0_0_MetadataDict(TypedDict):
    source: dict[str, Any]
    ldf_version: str
    classes: dict[str, list[str]]
    tasks: dict[str, list[str]]
    skeletons: dict[str, Skeletons]
    categorical_encodings: dict[str, dict[str, int]]
    metadata_types: dict[str, Literal["float", "int", "str", "Category"]]


class _LDF_2_0_0_MetadataDict(TypedDict):
    source: dict[str, Any]
    ldf_version: str
    classes: dict[str, list[str]]
    tasks: dict[str, list[str]]
    skeletons: dict[str, Skeletons]
    categorical_encodings: dict[str, dict[str, int]]
    metadata_types: dict[str, Literal["float", "int", "str", "Category"]]
    parent_dataset: str | None


T = TypeVar("T", pl.LazyFrame, pl.DataFrame)
TValue = TypeVar("TValue")


def _rename_metadata_key(key: str) -> str:
    if key.startswith("metadata/"):
        return f"labels/{key.removeprefix('metadata/')}"
    return key.replace("/metadata/", "/labels/")


def _rename_metadata_keys(mapping: dict[str, TValue]) -> dict[str, TValue]:
    return {_rename_metadata_key(key): value for key, value in mapping.items()}


def _frame_columns(df: pl.LazyFrame | pl.DataFrame) -> list[str]:
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema().names()
    return df.columns


def _add_metadata_column(df: T) -> T:
    if "metadata" in _frame_columns(df):
        return df
    return df.with_columns(pl.lit(None).alias("metadata"))


def _select_current_columns(df: T) -> T:
    columns = [
        "file",
        "source_name",
        "task_name",
        "metadata",
        "class_name",
        "instance_id",
        "task_type",
        "annotation",
        "uuid",
        "group_id",
    ]
    existing = set(_frame_columns(df))
    return df.select([column for column in columns if column in existing])


def migrate_dataframe(df: T, version: Version) -> T:  # pragma: no cover
    if version < Version(2):
        df = (
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
            .select(
                [
                    "file",
                    "source_name",
                    "task_name",
                    "class_name",
                    "instance_id",
                    "task_type",
                    "annotation",
                    "uuid",
                ]
            )
            .with_columns(pl.col("uuid").alias("group_id"))
        )
    if version < Version(3):
        df = _add_metadata_column(df)
        df = df.with_columns(
            pl.col("task_type")
            .str.replace("^metadata/", "labels/")
            .alias("task_type")
        )
        df = _select_current_columns(df)
    return df


def migrate_metadata(
    metadata: _LDF_1_0_0_MetadataDict | _LDF_2_0_0_MetadataDict,
    version: Version,
    df: pl.LazyFrame | None,
) -> Metadata:  # pragma: no cover
    """Migrate outdated metadata to the current schema.

    Args:
        metadata: Metadata dictionary in the LDF :math:`1.0.0` layout.
        version: LDF version of the input metadata.
        df: Optional annotation dataframe used to infer task names for
            non-default datasets.

    Returns:
        Migrated metadata model.

    Raises:
        ValueError: If task inference requires annotation rows but
            ``df`` is ``None``.

    """
    new_metadata = {}
    old_classes = metadata["classes"]
    if version < Version(2):
        if set(old_classes.keys()) <= LDF_1_0_0_TASKS:
            old_class_names = next(iter(old_classes.values()))
            new_metadata["classes"] = {
                "detection": {
                    class_name: i
                    for i, class_name in enumerate(old_class_names)
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
                    {
                        class_name: i
                        for i, class_name in enumerate(old_class_names)
                    }
                )
            new_metadata["classes"] = dict(new_classes)
            new_metadata["tasks"] = dict(tasks)

    metadata.update(new_metadata)
    if version < Version(3):
        if "metadata_types" in metadata:
            metadata["label_types"] = _rename_metadata_keys(
                metadata.pop("metadata_types")
            )
        elif "label_types" in metadata:
            metadata["label_types"] = _rename_metadata_keys(
                metadata["label_types"]
            )
        metadata["categorical_encodings"] = _rename_metadata_keys(
            metadata.get("categorical_encodings", {})
        )
        metadata["tasks"] = {
            task_name: [
                _rename_metadata_key(task_type) for task_type in task_types
            ]
            for task_name, task_types in metadata.get("tasks", {}).items()
        }
    return Metadata(**metadata)  # type: ignore
