"""Media and array annotations take a path or the data itself."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from luxonis_ml.ldf import ArrayAnnotation, DatasetRecord


@pytest.fixture
def image(tempdir: Path) -> Path:
    path = tempdir / "image.png"
    path.touch()
    return path


@pytest.fixture
def array(tempdir: Path) -> Path:
    path = tempdir / "array.npy"
    np.save(path, np.zeros(4))
    return path


def test_media_takes_one_file(image: Path):
    record = DatasetRecord(media=image)  # type: ignore[call-arg]

    assert record.files == {"image": image.absolute()}


def test_media_takes_a_mapping_of_sources(tempdir: Path):
    rgb = tempdir / "rgb.png"
    depth = tempdir / "depth.png"
    for path in (rgb, depth):
        path.touch()

    record = DatasetRecord(media={"rgb": rgb, "depth": depth})  # type: ignore[call-arg]

    assert set(record.files) == {"rgb", "depth"}


def test_the_deprecated_names_still_work(image: Path):
    from_file = DatasetRecord(file=image)  # type: ignore[call-arg]
    from_files = DatasetRecord(files={"image": image})

    assert from_file.files == from_files.files == {"image": image.absolute()}


def test_two_media_names_are_rejected(image: Path):
    with pytest.raises(ValidationError, match="not both"):
        DatasetRecord(media=image, file=image)  # type: ignore[call-arg]


def test_a_record_may_hold_the_image_itself():
    record = DatasetRecord(media=np.zeros((4, 4, 3), dtype=np.uint8))  # type: ignore[call-arg]

    assert isinstance(record.file, np.ndarray)


def test_an_in_memory_image_cannot_be_stored():
    record = DatasetRecord(media={"rgb": np.zeros((4, 4, 3), dtype=np.uint8)})  # type: ignore[call-arg]

    with pytest.raises(NotImplementedError, match="in-memory image"):
        _ = record.file_paths
    with pytest.raises(NotImplementedError, match="in-memory image"):
        list(record.to_parquet_rows())


def test_an_array_annotation_takes_a_path(array: Path):
    annotation = ArrayAnnotation(data=array)  # type: ignore[call-arg]

    assert annotation.to_numpy().tolist() == [0, 0, 0, 0]
    # Parsed rather than compared as text: JSON escapes the backslashes of a
    # Windows path, so the raw string differs per platform.
    assert json.loads(annotation.model_dump_json()) == {"path": str(array)}


def test_an_array_annotation_takes_the_data_itself():
    annotation = ArrayAnnotation(data=np.ones(3))  # type: ignore[call-arg]

    assert annotation.to_numpy().tolist() == [1, 1, 1]


def test_an_in_memory_array_cannot_be_serialized():
    annotation = ArrayAnnotation(data=np.ones(3))  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="Cannot serialize"):
        annotation.model_dump_json()


def test_the_deprecated_array_path_still_works(array: Path):
    assert ArrayAnnotation(path=array).to_numpy().tolist() == [0, 0, 0, 0]


def test_both_array_names_are_rejected(array: Path):
    with pytest.raises(ValidationError, match="not both"):
        ArrayAnnotation(path=array, data=array)  # type: ignore[call-arg]


def test_a_malformed_media_value_names_the_source():
    with pytest.raises(ValidationError) as error:
        DatasetRecord(media={"image": 5})  # type: ignore[call-arg]

    assert error.value.errors()[0]["loc"] == ("files", "image")
