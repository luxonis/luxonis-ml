"""A record survives a trip through the arrays a loader returns."""

from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.ldf import Category, DatasetRecord, DatasetSchema
from luxonis_ml.ldf.conversion import labels_to_record

IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)


def roundtrip(
    record: DatasetRecord, schema: DatasetSchema, **kwargs
) -> DatasetRecord:
    """Convert a record to loader arrays and back."""
    return record.to_loader_output(schema, **kwargs).to_ldf()


def test_boxes_keep_their_class_and_coordinates():
    schema = DatasetSchema(
        tasks={"vehicles": ["boundingbox", "classification"]},
        classes={"vehicles": {"car": 0, "truck": 1}},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "vehicles": [
                    {
                        "class": "truck",
                        "boundingbox": {
                            "x": 0.5,
                            "y": 0.5,
                            "w": 0.2,
                            "h": 0.2,
                        },
                    },
                    {
                        "class": "car",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.2,
                            "h": 0.2,
                        },
                    },
                ]
            },
        }
    )

    rebuilt = roundtrip(record, schema)

    detections = rebuilt.annotation["vehicles"]
    assert [detection.class_name for detection in detections] == [
        "truck",
        "car",
    ]
    assert (
        detections[0].boundingbox
        == record.annotation["vehicles"][0].boundingbox
    )


def test_class_only_labels_survive_beside_boxes():
    schema = DatasetSchema(
        tasks={"vehicles": ["boundingbox", "classification"]},
        classes={"vehicles": {"car": 0, "truck": 1}},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "vehicles": [
                    {
                        "class": "car",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.2,
                            "h": 0.2,
                        },
                    },
                    {"class": "truck"},
                ]
            },
        }
    )
    sample = record.to_loader_output(schema)

    rebuilt = sample.to_ldf()

    assert [
        (detection.class_name, detection.boundingbox is not None)
        for detection in rebuilt.annotation["vehicles"]
    ] == [("car", True), ("truck", False)]
    assert np.array_equal(
        rebuilt.to_loader_output(schema).labels["vehicles/classification"],
        sample.labels["vehicles/classification"],
    )


def test_keypoints_take_the_class_of_their_instance():
    """A keypoint row carries no class, so its box has to supply one."""
    schema = DatasetSchema(
        tasks={"pose": ["boundingbox", "keypoints", "classification"]},
        classes={"pose": {"person": 0}},
        n_keypoints={"pose": 2},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "pose": [
                    {
                        "class": "person",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.5,
                            "h": 0.5,
                        },
                        "keypoints": {
                            "keypoints": [(0.2, 0.2, 2), (0.3, 0.3, 1)]
                        },
                    }
                ]
            },
        }
    )

    (detection,) = roundtrip(record, schema).annotation["pose"]

    assert detection.class_name == "person"
    assert detection.keypoints is not None
    assert list(detection.keypoints.keypoints.values()) == [
        (0.2, 0.2, 2),
        (0.3, 0.3, 1),
    ]


def test_semantic_masks_keep_one_detection_per_class():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:4] = 1
    schema = DatasetSchema(
        tasks={"scene": ["segmentation", "classification"]},
        classes={"scene": {"road": 0, "sky": 1}},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "scene": [{"class": "road", "segmentation": {"mask": mask}}]
            },
        }
    )

    (detection,) = roundtrip(record, schema).annotation["scene"]

    assert detection.class_name == "road"
    assert detection.segmentation is not None
    assert np.array_equal(detection.segmentation.to_numpy(), mask)


def test_instance_masks_pair_with_their_box():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    schema = DatasetSchema(
        tasks={
            "cars": ["boundingbox", "instance_segmentation", "classification"]
        },
        classes={"cars": {"car": 0}},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "cars": [
                    {
                        "class": "car",
                        "boundingbox": {
                            "x": 0.2,
                            "y": 0.2,
                            "w": 0.3,
                            "h": 0.3,
                        },
                        "instance_segmentation": {"mask": mask},
                    }
                ]
            },
        }
    )

    (detection,) = roundtrip(record, schema).annotation["cars"]

    assert detection.class_name == "car"
    assert detection.boundingbox is not None
    assert detection.instance_segmentation is not None
    assert np.array_equal(detection.instance_segmentation.to_numpy(), mask)


def test_arrays_keep_their_data(tempdir: Path):
    array_path = tempdir / "embedding.npy"
    np.save(array_path, np.array([1.0, 2.0, 3.0]))
    schema = DatasetSchema(
        tasks={"embeddings": ["array", "classification"]},
        classes={"embeddings": {"vector": 0}},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "embeddings": [
                    {"class": "vector", "array": {"data": array_path}}
                ]
            },
        }
    )

    (detection,) = roundtrip(record, schema).annotation["embeddings"]

    assert detection.class_name == "vector"
    assert detection.array is not None
    assert detection.array.to_numpy().tolist() == [1.0, 2.0, 3.0]


def test_metadata_stays_with_its_own_instance():
    schema = DatasetSchema(
        tasks={
            "vehicles": [
                "boundingbox",
                "classification",
                "metadata/color",
                "metadata/wheels",
            ]
        },
        classes={"vehicles": {"car": 0}},
        categorical_encodings={
            "vehicles/metadata/color": {"red": 0, "blue": 1}
        },
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "vehicles": [
                    {
                        "class": "car",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.2,
                            "h": 0.2,
                        },
                        "metadata": {"color": Category("blue"), "wheels": 4},
                    },
                    {
                        "class": "car",
                        "boundingbox": {
                            "x": 0.5,
                            "y": 0.5,
                            "w": 0.2,
                            "h": 0.2,
                        },
                        "metadata": {"color": Category("red"), "wheels": 6},
                    },
                ]
            },
        }
    )

    detections = roundtrip(record, schema).annotation["vehicles"]

    assert detections[0].metadata == {"color": Category("blue"), "wheels": 4}
    assert detections[1].metadata == {"color": Category("red"), "wheels": 6}


def test_a_true_negative_stays_empty():
    schema = DatasetSchema(
        tasks={"vehicles": ["boundingbox", "classification"]},
        classes={"vehicles": {"car": 0}},
    )
    record = DatasetRecord.model_validate({"media": IMAGE})

    rebuilt = roundtrip(record, schema)

    assert rebuilt.annotation == {"vehicles": []}


def test_a_sub_detection_comes_back_nested():
    schema = DatasetSchema(
        tasks={
            "driver": ["boundingbox", "classification"],
            "driver/face": ["boundingbox", "classification"],
        },
        classes={
            "driver": {"person": 0},
            "driver/face": {"face": 0},
        },
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "driver": [
                    {
                        "class": "person",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.5,
                            "h": 0.5,
                        },
                        "sub_detections": {
                            "face": {
                                "class": "face",
                                "boundingbox": {
                                    "x": 0.2,
                                    "y": 0.2,
                                    "w": 0.1,
                                    "h": 0.1,
                                },
                            }
                        },
                    }
                ]
            },
        }
    )

    rebuilt = roundtrip(record, schema)

    assert set(rebuilt.annotation) == {"driver"}
    (driver,) = rebuilt.annotation["driver"]
    assert driver.class_name == "person"
    face = driver.sub_detections["face"]
    assert face.class_name == "face"
    assert face.boundingbox is not None
    assert face.boundingbox.x == pytest.approx(0.2)


def test_two_levels_of_sub_detections_come_back_nested():
    schema = DatasetSchema(
        tasks={
            "vehicle": ["boundingbox", "classification"],
            "vehicle/plate": ["boundingbox", "classification"],
            "vehicle/plate/text": ["classification"],
        },
        classes={
            "vehicle": {"car": 0},
            "vehicle/plate": {"plate": 0},
            "vehicle/plate/text": {"CO": 0},
        },
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "vehicle": [
                    {
                        "class": "car",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.5,
                            "h": 0.5,
                        },
                        "sub_detections": {
                            "plate": {
                                "class": "plate",
                                "boundingbox": {
                                    "x": 0.2,
                                    "y": 0.2,
                                    "w": 0.1,
                                    "h": 0.1,
                                },
                                "sub_detections": {"text": {"class": "CO"}},
                            }
                        },
                    }
                ]
            },
        }
    )

    rebuilt = roundtrip(record, schema)

    (vehicle,) = rebuilt.annotation["vehicle"]
    plate = vehicle.sub_detections["plate"]
    assert plate.class_name == "plate"
    assert plate.sub_detections["text"].class_name == "CO"


def test_every_parent_keeps_its_own_sub_detection():
    """Parents and children pair by row index, so the order has to hold."""
    schema = DatasetSchema(
        tasks={
            "driver": ["boundingbox", "classification"],
            "driver/face": ["boundingbox", "classification"],
        },
        classes={
            "driver": {"person": 0},
            "driver/face": {"happy": 0, "sad": 1},
        },
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "driver": [
                    {
                        "class": "person",
                        "instance_id": index,
                        "boundingbox": {
                            "x": 0.1 * index,
                            "y": 0.1,
                            "w": 0.1,
                            "h": 0.1,
                        },
                        "sub_detections": {
                            "face": {
                                "class": mood,
                                "instance_id": index,
                                "boundingbox": {
                                    "x": 0.1 * index,
                                    "y": 0.5,
                                    "w": 0.1,
                                    "h": 0.1,
                                },
                            }
                        },
                    }
                    for index, mood in enumerate(["happy", "sad", "happy"])
                ]
            },
        }
    )

    drivers = roundtrip(record, schema).annotation["driver"]

    assert [
        driver.sub_detections["face"].class_name for driver in drivers
    ] == ["happy", "sad", "happy"]
    assert [
        driver.boundingbox.x  # type: ignore[union-attr]
        for driver in drivers
    ] == pytest.approx([0.0, 0.1, 0.2])


def test_the_schema_travels_with_the_sample():
    schema = DatasetSchema(
        tasks={"vehicles": ["classification"]},
        classes={"vehicles": {"car": 0}},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {"vehicles": [{"class": "car"}]},
            "sample_metadata": {"camera": "left"},
        }
    )

    sample = record.to_loader_output(schema)

    assert sample.metadata["camera"] == "left"
    # The record rebuilds without being handed the schema again.
    assert sample.to_ldf().sample_metadata == {"camera": "left"}


def test_rebuilding_without_a_schema_is_refused():
    schema = DatasetSchema(classes={"": {"car": 0}})
    record = DatasetRecord.model_validate(
        {"media": IMAGE, "annotation": {"": [{"class": "car"}]}}
    )
    sample = record.to_loader_output(schema)
    sample.metadata.clear()

    with pytest.raises(ValueError, match="without the dataset schema"):
        sample.to_ldf()


def test_sub_detections_follow_their_parent_out_of_order():
    """The parents sort by instance ID; their faces must move with them.

    A sub-detection carries no instance ID of its own, so sorting the faces
    on their own leaves them in the order they were written while the
    parents move.
    """
    schema = DatasetSchema(
        tasks={
            "driver": ["boundingbox", "classification"],
            "driver/face": ["boundingbox", "classification"],
        },
        classes={
            "driver": {"person": 0},
            "driver/face": {"happy": 0, "sad": 1},
        },
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "driver": [
                    {
                        "class": "person",
                        "instance_id": 1,
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.1,
                            "h": 0.1,
                        },
                        "sub_detections": {
                            "face": {
                                "class": "sad",
                                "boundingbox": {
                                    "x": 0.15,
                                    "y": 0.2,
                                    "w": 0.05,
                                    "h": 0.05,
                                },
                            }
                        },
                    },
                    {
                        "class": "person",
                        "instance_id": 0,
                        "boundingbox": {
                            "x": 0.6,
                            "y": 0.1,
                            "w": 0.1,
                            "h": 0.1,
                        },
                        "sub_detections": {
                            "face": {
                                "class": "happy",
                                "boundingbox": {
                                    "x": 0.65,
                                    "y": 0.2,
                                    "w": 0.05,
                                    "h": 0.05,
                                },
                            }
                        },
                    },
                ]
            },
        }
    )

    drivers = roundtrip(record, schema).annotation["driver"]

    faces = {
        round(driver.boundingbox.x, 3): (  # type: ignore[union-attr]
            driver.sub_detections["face"].class_name,
            round(driver.sub_detections["face"].boundingbox.x, 3),  # type: ignore[union-attr]
        )
        for driver in drivers
    }
    assert faces == {0.1: ("sad", 0.15), 0.6: ("happy", 0.65)}


def test_a_class_the_schema_does_not_define_is_refused():
    """An array numbers its classes, so the schema has to know them all."""
    schema = DatasetSchema(
        tasks={"vehicles": ["boundingbox", "classification"]},
        classes={"vehicles": {"car": 0}},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "vehicles": [
                    {
                        "class": "van",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.1,
                            "h": 0.1,
                        },
                    }
                ]
            },
        }
    )

    with pytest.raises(ValueError, match="does not define the class 'van'"):
        record.to_loader_output(schema)


def test_a_class_id_the_schema_does_not_know_rebuilds_without_a_name():
    """A rebuild must not invent a name for an ID the schema lacks."""
    schema = DatasetSchema(
        tasks={"v": ["boundingbox", "classification"]},
        classes={"v": {"car": 0}},
    )

    record = labels_to_record(
        {"v/boundingbox": np.array([[7, 0.1, 0.2, 0.3, 0.4]])}, schema
    )

    (detection,) = record.annotation["v"]
    assert detection.class_name is None
    assert detection.boundingbox is not None
    assert detection.boundingbox.x == pytest.approx(0.1)


def test_a_sub_task_without_a_parent_keeps_its_own_name():
    """Nothing may be dropped when the parent task has no detections."""
    schema = DatasetSchema(
        tasks={"driver/face": ["boundingbox", "classification"]},
        classes={"driver/face": {"happy": 0}},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "driver/face": [
                    {
                        "class": "happy",
                        "boundingbox": {
                            "x": 0.3,
                            "y": 0.1,
                            "w": 0.1,
                            "h": 0.1,
                        },
                    }
                ]
            },
        }
    )

    rebuilt = roundtrip(record, schema)

    assert set(rebuilt.annotation) == {"driver/face"}
    (face,) = rebuilt.annotation["driver/face"]
    assert face.boundingbox is not None
    assert face.boundingbox.x == pytest.approx(0.3)


def test_children_are_not_nested_when_a_parent_is_missing():
    schema = DatasetSchema(
        tasks={
            "driver": ["boundingbox"],
            "driver/face": ["boundingbox"],
        },
        classes={"driver": {"person": 0}, "driver/face": {"happy": 0}},
    )

    record = labels_to_record(
        {
            "driver/boundingbox": np.array([[0, 0.1, 0.1, 0.1, 0.1]]),
            "driver/face/boundingbox": np.array(
                [[0, 0.2, 0.2, 0.1, 0.1], [0, 0.6, 0.6, 0.1, 0.1]]
            ),
        },
        schema,
    )

    assert record.annotation["driver"][0].sub_detections == {}
    assert len(record.annotation["driver/face"]) == 2


def test_sparse_children_are_not_attached_by_compacted_index():
    schema = DatasetSchema(
        tasks={
            "driver": ["boundingbox"],
            "driver/face": ["boundingbox"],
        },
        classes={"driver": {"person": 0}, "driver/face": {"face": 0}},
    )

    record = labels_to_record(
        {
            "driver/boundingbox": np.array(
                [
                    [0, 0.1, 0.1, 0.1, 0.1],
                    [0, 0.3, 0.1, 0.1, 0.1],
                    [0, 0.6, 0.1, 0.1, 0.1],
                ]
            ),
            "driver/face/boundingbox": np.array(
                [[0, 0.1, 0.2, 0.1, 0.1], [0, 0.6, 0.2, 0.1, 0.1]]
            ),
        },
        schema,
    )

    assert all(
        not driver.sub_detections for driver in record.annotation["driver"]
    )
    assert [
        face.boundingbox.x  # type: ignore[union-attr]
        for face in record.annotation["driver/face"]
    ] == pytest.approx([0.1, 0.6])


def test_a_named_task_is_not_nested_under_the_default_task():
    """The default task is named ``""``, and every name ends with it.

    A rebuild that splits a task name on its last ``"/"`` reads ``"vehicles"``
    as the ``"vehicles"`` sub-detection of the default task, and the named
    task disappears.
    """
    schema = DatasetSchema(
        tasks={"": ["classification"], "vehicles": ["classification"]},
        classes={"": {"scene": 0}, "vehicles": {"car": 0}},
    )
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "": [{"class": "scene"}],
                "vehicles": [{"class": "car"}],
            },
        }
    )

    rebuilt = roundtrip(record, schema)

    assert set(rebuilt.annotation) == {"", "vehicles"}
    assert rebuilt.annotation["vehicles"][0].class_name == "car"
    assert rebuilt.annotation[""][0].sub_detections == {}
