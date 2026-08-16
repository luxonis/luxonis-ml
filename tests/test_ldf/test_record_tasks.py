"""A record groups its detections by task name."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from luxonis_ml.ldf import DatasetRecord, Detection

CAR = {
    "class": "car",
    "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
}
TRUCK = {
    "class": "truck",
    "boundingbox": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},
}


@pytest.fixture
def image(tempdir: Path) -> Path:
    path = tempdir / "image.png"
    path.touch()
    return path


def rows(record: DatasetRecord) -> list[tuple[str, str | None, str | None]]:
    """Return the task name, task type and class of each parquet row."""
    return [
        (row["task_name"], row["task_type"], row["class_name"])
        for row in record.to_parquet_rows()
    ]


def test_a_bare_detection_lands_in_the_default_task(image: Path):
    record = DatasetRecord(media=image, annotation=CAR)  # type: ignore[call-arg]

    assert set(record.annotation) == {""}
    assert record.annotation[""][0].class_name == "car"


def test_a_detection_and_a_one_item_list_agree(image: Path):
    single = DatasetRecord(media=image, annotation=CAR)  # type: ignore[call-arg]
    listed = DatasetRecord(media=image, annotation=[CAR])  # type: ignore[call-arg]

    assert list(single.to_parquet_rows()) == list(listed.to_parquet_rows())


def test_detections_are_grouped_by_task(image: Path):
    record = DatasetRecord(
        media=image,  # type: ignore[call-arg]
        annotation={"vehicles": [CAR, TRUCK], "weather": [{"class": "rain"}]},
    )

    assert rows(record) == [
        ("vehicles", "boundingbox", "car"),
        ("vehicles", "classification", "car"),
        ("vehicles", "boundingbox", "truck"),
        ("vehicles", "classification", "truck"),
        ("weather", "classification", "rain"),
    ]


def test_the_deprecated_task_name_becomes_the_mapping_key(image: Path):
    record = DatasetRecord(
        media=image,  # type: ignore[call-arg]
        task_name="vehicles",  # type: ignore[call-arg]
        annotation=[CAR],
    )

    assert set(record.annotation) == {"vehicles"}
    assert rows(record) == [
        ("vehicles", "boundingbox", "car"),
        ("vehicles", "classification", "car"),
    ]


def test_a_task_name_beside_a_mapping_has_to_match(image: Path):
    with pytest.raises(ValidationError, match="does not match the tasks"):
        DatasetRecord(
            media=image,  # type: ignore[call-arg]
            task_name="weather",  # type: ignore[call-arg]
            annotation={"vehicles": [CAR]},
        )


def test_a_task_name_without_detections_declares_the_task(image: Path):
    record = DatasetRecord(  # type: ignore[call-arg]
        media=image,  # type: ignore[call-arg]
        task_name="vehicles",  # type: ignore[call-arg]
    )

    assert record.annotation == {"vehicles": []}
    assert rows(record) == [("vehicles", None, None)]


def test_a_record_without_an_annotation_declares_nothing(image: Path):
    record = DatasetRecord(media=image)  # type: ignore[call-arg]

    assert record.annotation == {}
    assert rows(record) == [("", None, None)]


def test_task_names_may_name_a_sub_detection(image: Path):
    record = DatasetRecord(
        media=image,  # type: ignore[call-arg]
        annotation={"driver/face": [{"class": "face"}]},
    )

    assert rows(record) == [("driver/face", "classification", "face")]


def test_an_empty_part_of_a_task_name_is_rejected(image: Path):
    with pytest.raises(ValidationError, match="empty part"):
        DatasetRecord(
            media=image,  # type: ignore[call-arg]
            annotation={"driver//face": [{"class": "face"}]},
        )


def test_the_task_name_property_needs_a_single_task(image: Path):
    record = DatasetRecord(
        media=image,  # type: ignore[call-arg]
        annotation={"vehicles": [CAR], "weather": [{"class": "rain"}]},
    )

    with pytest.raises(ValueError, match="no single task name"):
        _ = record.task_name


def test_every_secondary_source_gets_exactly_one_empty_row(tempdir: Path):
    """A sub-detection must not add a row to the other sources.

    The rows were emitted once per detection level, so a nested detection
    repeated the empty row of every secondary source.
    """
    rgb = tempdir / "rgb.png"
    depth = tempdir / "depth.png"
    for path in (rgb, depth):
        path.touch()

    record = DatasetRecord(  # type: ignore[call-arg]
        media={"rgb": rgb, "depth": depth},  # type: ignore[call-arg]
        annotation={
            "driver": [
                Detection(
                    class_name="person",
                    sub_detections={"face": Detection(class_name="face")},
                )
            ]
        },
    )

    emitted = [
        (row["source_name"], row["task_name"], row["task_type"])
        for row in record.to_parquet_rows()
    ]

    # The main source is the first file in path order, so the other one
    # carries the single empty row.
    empty = [row for row in emitted if row[2] is None]
    assert len(empty) == 1
    assert empty[0][1] == "driver"


def test_detections_keep_their_own_sub_detections(image: Path):
    record = DatasetRecord(
        media=image,  # type: ignore[call-arg]
        annotation={
            "driver": [
                Detection(
                    class_name="person",
                    sub_detections={"face": Detection(class_name="face")},
                )
            ]
        },
    )

    assert rows(record) == [
        ("driver", "classification", "person"),
        ("driver/face", "classification", "face"),
    ]
