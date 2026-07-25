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


def _car() -> Detection:
    return Detection(
        class_name="car",
        instance_id=0,
        boundingbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},  # type: ignore
    )


def _person() -> Detection:
    return Detection(
        class_name="person",
        instance_id=1,
        boundingbox={"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},  # type: ignore
        keypoints={"keypoints": [(0.5, 0.5, 2)]},  # type: ignore
    )


def test_single_detection_equals_singleton_list(image: str):
    """A single `Detection` and a length-1 list produce identical rows."""
    single = DatasetRecord(file=image, annotation=_car(), task_name="t")  # type: ignore
    listed = DatasetRecord(file=image, annotation=[_car()], task_name="t")  # type: ignore
    assert list(single.to_parquet_rows()) == list(listed.to_parquet_rows())


def test_list_flattens_all_detections(image: str):
    record = DatasetRecord(  # type: ignore
        file=image,  # type: ignore
        annotation=[_car(), _person()],
        task_name="t",
    )
    rows = list(record.to_parquet_rows())
    pairs = {(r["class_name"], r["task_type"]) for r in rows}
    assert ("car", "boundingbox") in pairs
    assert ("person", "boundingbox") in pairs
    assert ("person", "keypoints") in pairs


def test_secondary_source_gets_one_null_row_per_source(image: str, depth: str):
    """Each secondary source gets exactly one null row, not one per detection."""
    record = DatasetRecord(
        files={"image": image, "depth": depth},  # type: ignore
        annotation=[_car(), _person()],
        task_name="t",
    )
    rows = list(record.to_parquet_rows())
    # `image` sorts before `depth`, so `image` is the main source.
    null_rows = [r for r in rows if r["task_type"] is None]
    assert len(null_rows) == 1
    assert null_rows[0]["source_name"] == "depth"
    assert not any(r["source_name"] == "image" for r in null_rows)


def test_annotations_helper_normalizes(image: str):
    assert DatasetRecord(file=image)._annotations() == []  # type: ignore
    car = _car()
    assert DatasetRecord(file=image, annotation=car)._annotations() == [car]  # type: ignore
    assert DatasetRecord(file=image, annotation=[car])._annotations() == [car]  # type: ignore
