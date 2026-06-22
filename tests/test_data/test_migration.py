import polars as pl
from semver.version import Version

from luxonis_ml.data.datasets.migration import (
    migrate_dataframe,
    migrate_metadata,
)


def test_migrate_v2_dataframe_adds_record_metadata_and_renames_labels():
    df = pl.DataFrame(
        {
            "file": ["image.jpg"],
            "source_name": ["image"],
            "task_name": [""],
            "class_name": ["car"],
            "instance_id": [0],
            "task_type": ["metadata/color"],
            "annotation": ['"red"'],
            "uuid": ["uuid-0"],
            "group_id": ["uuid-0"],
        }
    )

    migrated = migrate_dataframe(df, Version.parse("2.0.0"))

    assert migrated.columns == [
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
    assert migrated["metadata"].to_list() == [None]
    assert migrated["task_type"].to_list() == ["labels/color"]


def test_migrate_v2_metadata_renames_custom_label_keys():
    metadata = {
        "source": None,
        "ldf_version": "2.0.0",
        "classes": {"": {"car": 0}},
        "tasks": {
            "": ["classification", "metadata/color"],
            "/license_plate": ["metadata/text"],
        },
        "skeletons": {},
        "categorical_encodings": {
            "/metadata/color": {"red": 0},
            "/license_plate/metadata/text": {"ABC123": 0},
        },
        "metadata_types": {
            "/metadata/color": "Category",
            "/license_plate/metadata/text": "str",
        },
        "parent_dataset": None,
    }

    migrated = migrate_metadata(metadata, Version.parse("2.0.0"), None)

    assert migrated.tasks == {
        "": ["classification", "labels/color"],
        "/license_plate": ["labels/text"],
    }
    assert migrated.categorical_encodings == {
        "/labels/color": {"red": 0},
        "/license_plate/labels/text": {"ABC123": 0},
    }
    assert migrated.label_types == {
        "/labels/color": "Category",
        "/license_plate/labels/text": "str",
    }
