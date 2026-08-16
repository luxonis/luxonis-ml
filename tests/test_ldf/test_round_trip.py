"""A record survives a trip through the arrays a loader returns."""

from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.ldf import Category, DatasetRecord, DatasetSchema

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
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={
            "vehicles": [
                {
                    "class": "truck",
                    "boundingbox": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},
                },
                {
                    "class": "car",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                },
            ]
        },
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


def test_keypoints_take_the_class_of_their_instance():
    """A keypoint row carries no class, so its box has to supply one."""
    schema = DatasetSchema(
        tasks={"pose": ["boundingbox", "keypoints", "classification"]},
        classes={"pose": {"person": 0}},
        n_keypoints={"pose": 2},
    )
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={
            "pose": [
                {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
                    "keypoints": {"keypoints": [(0.2, 0.2, 2), (0.3, 0.3, 1)]},
                }
            ]
        },
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
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={
            "scene": [{"class": "road", "segmentation": {"mask": mask}}]
        },
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
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={
            "cars": [
                {
                    "class": "car",
                    "boundingbox": {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3},
                    "instance_segmentation": {"mask": mask},
                }
            ]
        },
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
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={
            "embeddings": [{"class": "vector", "array": {"data": array_path}}]
        },
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
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={
            "vehicles": [
                {
                    "class": "car",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                    "metadata": {"color": Category("blue"), "wheels": 4},
                },
                {
                    "class": "car",
                    "boundingbox": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},
                    "metadata": {"color": Category("red"), "wheels": 6},
                },
            ]
        },
    )

    detections = roundtrip(record, schema).annotation["vehicles"]

    assert detections[0].metadata == {"color": Category("blue"), "wheels": 4}
    assert detections[1].metadata == {"color": Category("red"), "wheels": 6}


def test_a_true_negative_stays_empty():
    schema = DatasetSchema(
        tasks={"vehicles": ["boundingbox", "classification"]},
        classes={"vehicles": {"car": 0}},
    )
    record = DatasetRecord(media=IMAGE)  # type: ignore[call-arg]

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
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={
            "driver": [
                {
                    "class": "person",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
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
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={
            "vehicle": [
                {
                    "class": "car",
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
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
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={
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
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={"vehicles": [{"class": "car"}]},
        sample_metadata={"camera": "left"},
    )

    sample = record.to_loader_output(schema)

    assert sample.metadata["camera"] == "left"
    # The record rebuilds without being handed the schema again.
    assert sample.to_ldf().sample_metadata == {"camera": "left"}


def test_rebuilding_without_a_schema_is_refused():
    schema = DatasetSchema(classes={"": {"car": 0}})
    record = DatasetRecord(
        media=IMAGE,  # type: ignore[call-arg]
        annotation={"": [{"class": "car"}]},
    )
    sample = record.to_loader_output(schema)
    sample.metadata.clear()

    with pytest.raises(ValueError, match="without the dataset schema"):
        sample.to_ldf()
