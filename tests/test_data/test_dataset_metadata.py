import pytest

from luxonis_ml.data import Metadata
from luxonis_ml.data.datasets.source import LuxonisSource
from luxonis_ml.ldf import KeypointMetadata


@pytest.fixture
def basic_metadata() -> Metadata:
    return Metadata(
        source=None,
        ldf_version="2.0.0",
        classes={"task1": {"class1": 0, "class2": 1}},
        tasks={"task1": ["subtask1", "subtask2"]},
        keypoint_metadata={
            "task1": KeypointMetadata.model_validate(
                {
                    "keypoints": {
                        "head": {"sigma": 0.05},
                        "tail": {"connected_to": ["head"]},
                    }
                }
            )
        },
        categorical_encodings={"cat1": {"a": 0, "b": 1}},
        metadata_types={"field1": "str", "field2": "int"},
    )


def test_merge_with_different_versions():
    """Test that merging metadata with different LDF versions raises
    ValueError.
    """
    metadata1 = Metadata(
        ldf_version="2.0.0",
        source=None,
        classes={},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={},
    )
    metadata2 = Metadata(
        ldf_version="2.0",
        source=None,
        classes={},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={},
    )

    with pytest.raises(
        ValueError, match="Cannot merge metadata with different LDF versions"
    ):
        metadata1.merge_with(metadata2)


def test_merge_classes(basic_metadata: Metadata):
    """Test merging of classes dictionaries."""
    other_metadata = Metadata(
        source=None,
        ldf_version="2.0.0",
        classes={"task1": {"class2": 1, "class3": 2}, "task2": {"classA": 0}},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={},
    )

    merged = basic_metadata.merge_with(other_metadata)

    assert "task1" in merged.classes
    assert "task2" in merged.classes
    assert len(merged.classes["task1"]) == 3
    assert all(
        c in merged.classes["task1"] for c in ["class1", "class2", "class3"]
    )
    assert merged.classes["task2"] == {"classA": 0}


def test_merge_tasks(basic_metadata: Metadata):
    """Test merging of tasks dictionaries."""
    other_metadata = Metadata(
        source=None,
        ldf_version="2.0.0",
        classes={},
        tasks={"task1": ["subtask2", "subtask3"], "task2": ["other_subtask"]},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={},
    )

    merged = basic_metadata.merge_with(other_metadata)

    assert set(merged.tasks["task1"]) == {"subtask1", "subtask2", "subtask3"}
    assert merged.tasks["task2"] == ["other_subtask"]


def test_merge_categorical_encodings(basic_metadata: Metadata):
    """Test merging of categorical encodings."""
    other_metadata = Metadata(
        source=None,
        ldf_version="2.0.0",
        classes={},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={"cat1": {"b": 1, "c": 2}, "cat2": {"x": 0}},
        metadata_types={},
    )

    merged = basic_metadata.merge_with(other_metadata)

    assert "cat1" in merged.categorical_encodings
    assert "cat2" in merged.categorical_encodings
    assert merged.categorical_encodings["cat1"]["c"] == 2
    assert merged.categorical_encodings["cat2"] == {"x": 0}


def test_merge_sources():
    """Test merging of sources."""
    source1 = LuxonisSource()
    source2 = LuxonisSource()

    metadata1 = Metadata(
        source=source1,
        ldf_version="2.0.0",
        classes={},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={},
    )

    metadata2 = Metadata(
        source=source2,
        ldf_version="2.0.0",
        classes={},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={},
    )

    merged = metadata1.merge_with(metadata2)
    assert merged.source is not None


def test_merge_with_none_sources():
    """Test merging when one or both sources are None."""
    source = LuxonisSource()

    metadata1 = Metadata(
        source=None,
        ldf_version="2.0.0",
        classes={},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={},
    )

    metadata2 = Metadata(
        source=source,
        ldf_version="2.0.0",
        classes={},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={},
    )

    merged = metadata1.merge_with(metadata2)
    assert merged.source == source

    merged = metadata2.merge_with(metadata1)
    assert merged.source == source


def test_merge_metadata_types(basic_metadata: Metadata):
    """Test merging of metadata types."""
    other_metadata = Metadata(
        source=None,
        ldf_version="2.0.0",
        classes={},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={"field2": "int", "field3": "float"},
    )

    merged = basic_metadata.merge_with(other_metadata)

    assert merged.metadata_types["field1"] == "str"
    assert merged.metadata_types["field2"] == "int"
    assert merged.metadata_types["field3"] == "float"


def test_merge_adds_the_new_keypoint_names_after_the_own(
    basic_metadata: Metadata,
):
    other = Metadata(
        source=None,
        ldf_version="2.0.0",
        keypoint_metadata={
            "task1": KeypointMetadata.model_validate(
                {
                    "keypoints": {
                        "fin": {},
                        "tail": {
                            "sigma": 0.07,
                            "connected_to": ["fin"],
                        },
                    }
                }
            )
        },
    )

    merged = basic_metadata.merge_with(other).keypoint_metadata["task1"]

    assert merged.labels == ["head", "tail", "fin"]
    assert merged.edges == [(0, 1), (1, 2)]
    sigmas = {name: spec.sigma for name, spec in merged.keypoints.items()}
    assert sigmas == {"head": 0.05, "tail": 0.07, "fin": None}


def test_merge_rejects_two_sigmas_for_one_keypoint(basic_metadata: Metadata):
    other = Metadata(
        source=None,
        ldf_version="2.0.0",
        keypoint_metadata={
            "task1": KeypointMetadata.model_validate(
                {"keypoints": {"head": {"sigma": 0.08}}}
            )
        },
    )

    with pytest.raises(ValueError, match=r"Task 'task1'.*'head'"):
        basic_metadata.merge_with(other)
