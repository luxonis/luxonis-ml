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
            "task1": KeypointMetadata(labels=["head", "tail"], edges=[(0, 1)])
        },
        categorical_encodings={"cat1": {"a": 0, "b": 1}},
        metadata_types={"field1": "str", "field2": "int"},
    )


def test_merge_with_different_versions():
    """Test that merging metadata with different major LDF versions
    raises ValueError.
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
        ldf_version="1.0.0",
        source=None,
        classes={},
        tasks={},
        keypoint_metadata={},
        categorical_encodings={},
        metadata_types={},
    )

    with pytest.raises(
        ValueError,
        match="Cannot merge metadata with different major LDF versions",
    ):
        metadata1.merge_with(metadata2)


@pytest.mark.parametrize(
    ("version", "other_version"),
    [("2.1.0", "2.2.0"), ("2.2.0", "2.1"), ("2.2", "2.2.0")],
)
def test_merge_keeps_the_newer_minor_version(version: str, other_version: str):
    """Datasets of one major LDF version merge.

    The merged metadata holds content of both datasets, so it gets the
    newer version.
    """
    metadata = Metadata(source=None, ldf_version=version)
    other = Metadata(source=None, ldf_version=other_version)

    assert metadata.merge_with(other).ldf_version == "2.2.0"


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


def test_merge_keeps_keypoint_fields_the_other_dataset_omits(
    basic_metadata: Metadata,
):
    """Test that a merge fills the empty keypoint fields.

    A plainer dataset that merges in must not drop the edges, flip pairs
    and sigmas of the target. The fields hold indices into the labels, so
    they only carry over while both datasets list the same labels in the
    same order.
    """
    other_metadata = Metadata(
        source=None,
        keypoint_metadata={
            "task1": KeypointMetadata(
                labels=["head", "tail"], sigmas=[0.05, 0.05]
            )
        },
    )

    merged = basic_metadata.merge_with(other_metadata)

    assert merged.keypoint_metadata["task1"] == KeypointMetadata(
        labels=["head", "tail"], edges=[(0, 1)], sigmas=[0.05, 0.05]
    )


@pytest.mark.parametrize("unnamed_side", ["mine", "theirs"])
def test_merge_keeps_the_labels_that_only_one_dataset_has(unnamed_side: str):
    """An empty list of labels is not a different list of labels.

    An older luxonis-ml could store edges without labels, so a merge with
    such a dataset must keep the names of the other one.
    """
    named = KeypointMetadata(labels=["head", "tail"], sigmas=[0.05, 0.05])
    unnamed = KeypointMetadata(edges=[(0, 1)])
    mine, theirs = (
        (unnamed, named) if unnamed_side == "mine" else (named, unnamed)
    )

    merged = Metadata(
        source=None, keypoint_metadata={"task1": mine}
    ).merge_with(Metadata(source=None, keypoint_metadata={"task1": theirs}))

    assert merged.keypoint_metadata["task1"] == KeypointMetadata(
        labels=["head", "tail"], edges=[(0, 1)], sigmas=[0.05, 0.05]
    )


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
