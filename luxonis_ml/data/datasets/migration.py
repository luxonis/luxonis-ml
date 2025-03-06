from collections import defaultdict
from typing import (
    Any,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Set,
    Union,
    overload,
)

import polars as pl
from typing_extensions import TypedDict

from .metadata import Metadata, Skeletons

LDF_1_0_0_TASKS: Final[Set[str]] = {
    "classification",
    "segmentation",
    "boundingbox",
    "keypoints",
    "array",
}

LDF_1_0_0_TASK_TYPES: Final[Dict[str, str]] = {
    "BBoxAnnotation": "boundingbox",
    "ClassificationAnnotation": "classification",
    "PolylineSegmentationAnnotation": "segmentation",
    "RLESegmentationAnnotation": "segmentation",
    "MaskSegmentationAnnotation": "segmentation",
    "KeypointAnnotation": "keypoints",
    "ArrayAnnotation": "array",
}


class LDF_1_0_0_MetadataDict(TypedDict):
    source: Dict[str, Any]
    ldf_version: str
    classes: Dict[str, List[str]]
    tasks: Dict[str, List[str]]
    skeletons: Dict[str, Skeletons]
    categorical_encodings: Dict[str, Dict[str, int]]
    metadata_types: Dict[str, Literal["float", "int", "str", "Category"]]


@overload
def migrate_dataframe(df: pl.LazyFrame) -> pl.LazyFrame: ...


@overload
def migrate_dataframe(df: pl.DataFrame) -> pl.DataFrame: ...


def migrate_dataframe(
    df: Union[pl.LazyFrame, pl.DataFrame],
) -> Union[pl.LazyFrame, pl.DataFrame]:  # pragma: no cover
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
    )


def migrate_metadata(
    metadata: LDF_1_0_0_MetadataDict, df: Optional[pl.LazyFrame]
) -> Metadata:  # pragma: no cover
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
