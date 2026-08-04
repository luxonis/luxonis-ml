"""Tests for `DatasetRecord` accepting a list of detections."""

from pathlib import Path

import pytest

from luxonis_ml.ldf import DatasetRecord, Detection


@pytest.fixture
def image(tmp_path: Path) -> str:
    path = tmp_path / "a_image.jpg"
    path.write_bytes(b"x")
    return str(path)


@pytest.fixture
def depth(tmp_path: Path) -> str:
    path = tmp_path / "b_depth.png"
    path.write_bytes(b"x")
    return str(path)


def _car(**kwargs) -> Detection:
    return Detection(
        class_name="car",
        instance_id=0,
        boundingbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},  # type: ignore[arg-type]
        **kwargs,
    )


def _person() -> Detection:
    return Detection(
        class_name="person",
        instance_id=1,
        boundingbox={"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},  # type: ignore[arg-type]
        keypoints={"keypoints": [(0.5, 0.5, 2)]},  # type: ignore[arg-type]
    )


def test_single_detection_is_stored_as_a_list(image: str):
    car = _car()
    record = DatasetRecord(file=image, annotation=car)  # type: ignore[call-arg]

    assert record.annotation == [car]


def test_single_detection_equals_singleton_list(image: str):
    """A single `Detection` and a length-1 list produce identical rows."""
    single = DatasetRecord(file=image, annotation=_car(), task_name="t")  # type: ignore[call-arg]
    listed = DatasetRecord(file=image, annotation=[_car()], task_name="t")  # type: ignore[call-arg]

    assert list(single.to_parquet_rows()) == list(listed.to_parquet_rows())


def test_list_flattens_all_detections(image: str):
    record = DatasetRecord(
        file=image,  # type: ignore[call-arg]
        annotation=[_car(), _person()],
        task_name="t",
    )

    rows = record.to_parquet_rows()
    assert {(row["class_name"], row["task_type"]) for row in rows} == {
        ("car", "boundingbox"),
        ("car", "classification"),
        ("person", "boundingbox"),
        ("person", "keypoints"),
        ("person", "classification"),
    }


def test_secondary_source_gets_one_null_row_per_source(image: str, depth: str):
    """Each secondary source gets one null row, not one per detection."""
    record = DatasetRecord(
        files={"image": image, "depth": depth},  # type: ignore[dict-item]
        annotation=[_car(), _person()],
        task_name="t",
    )

    # `image` sorts before `depth`, so `image` is the main source.
    null_rows = [
        row for row in record.to_parquet_rows() if row["task_type"] is None
    ]

    assert len(null_rows) == 1
    assert null_rows[0]["source_name"] == "depth"


def test_sub_detections_do_not_repeat_secondary_source_rows(
    image: str, depth: str
):
    car = _car(
        sub_detections={
            "occupant": Detection(
                class_name="person",
                boundingbox={"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1},  # type: ignore[arg-type]
            )
        }
    )
    record = DatasetRecord(
        files={"image": image, "depth": depth},  # type: ignore[dict-item]
        annotation=[car, _person()],
        task_name="t",
    )

    rows = list(record.to_parquet_rows())
    null_rows = [row for row in rows if row["task_type"] is None]
    nested = [row for row in rows if row["task_name"] == "t/occupant"]

    assert len(null_rows) == 1
    assert null_rows[0]["source_name"] == "depth"
    assert {row["source_name"] for row in nested} == {"image"}
