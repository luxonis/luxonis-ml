"""Focused tests for typed dataset-inspection sample filters."""

from pathlib import Path

import pytest

from luxonis_ml.data.utils.inspection import (
    InspectionAnnotationType,
    InspectionQuery,
    MetadataPredicate,
    SampleFilterConfig,
    _metadata_equal,
)
from luxonis_ml.ldf import BBoxAnnotation, DatasetRecord, Detection
from luxonis_ml.typing import Params


def _records() -> dict[str, DatasetRecord]:
    child = Detection(class_name="plate", metadata={"text": "ABC-123"})
    detection = Detection(
        class_name="car",
        instance_id=7,
        boundingbox=BBoxAnnotation(x=0.1, y=0.2, w=0.3, h=0.4),
        metadata={"confidence": 0.86, "quality": "approved"},
        sub_detections={"plate": child},
    )
    return {
        "objects": DatasetRecord.model_construct(
            files={"image": Path("frame.jpg")},
            annotation=[detection],
            task_name="objects",
        )
    }


def test_sample_filter_config_validates_and_builds_query() -> None:
    config = SampleFilterConfig(
        task_name=["objects", "objects"],
        class_name=["car"],
        annotation_type=["boundingbox"],
        metadata_filter=[("camera.side", "left")],
        min_confidence=0.8,
        min_instances=1,
        max_instances=2,
        search="  FRAME_0042  ",
    )

    config.validate(
        available_tasks=["objects"],
        available_classes=["car", "person"],
    )

    assert config.task_filter == frozenset({"objects"})
    assert config.query().matches(
        _records(),
        {
            "filenames": {"image": "frame_0042.jpg"},
            "camera": {"side": "left"},
        },
    )


@pytest.mark.parametrize(
    "query",
    [
        InspectionQuery(class_names=frozenset({"car"})),
        InspectionQuery(annotation_types=frozenset({"boundingbox"})),
        InspectionQuery(annotation_types=frozenset({"metadata"})),
        InspectionQuery(min_confidence=0.8),
        InspectionQuery(min_instances=1, max_instances=1),
        InspectionQuery(
            metadata=(MetadataPredicate.from_pair("camera.side", "left"),)
        ),
        InspectionQuery(
            metadata=(MetadataPredicate.from_pair("quality", "approved"),)
        ),
        InspectionQuery(
            class_names=frozenset({"person"}),
            class_name_mode="exclude",
        ),
        InspectionQuery(search="abc-123"),
        InspectionQuery(search="FRAME_0042"),
    ],
)
def test_query_matches_supported_sample_filters(
    query: InspectionQuery,
) -> None:
    metadata: Params = {
        "filenames": {"image": "frame_0042.jpg"},
        "camera": {"side": "left"},
    }
    assert query.matches(_records(), metadata)


@pytest.mark.parametrize(
    "query",
    [
        InspectionQuery(class_names=frozenset({"person"})),
        InspectionQuery(annotation_types=frozenset({"segmentation"})),
        InspectionQuery(min_confidence=0.9),
        InspectionQuery(min_instances=2),
        InspectionQuery(max_instances=0),
        InspectionQuery(
            metadata=(MetadataPredicate.from_pair("camera.side", "right"),)
        ),
        InspectionQuery(
            class_names=frozenset({"car"}),
            class_name_mode="exclude",
        ),
        InspectionQuery(search="missing"),
        InspectionQuery(unlabeled_only=True),
    ],
)
def test_query_rejects_nonmatching_samples(query: InspectionQuery) -> None:
    metadata: Params = {
        "filenames": {"image": "frame_0042.jpg"},
        "camera": {"side": "left"},
    }
    assert not query.matches(_records(), metadata)


def test_unlabeled_query_accepts_an_empty_sample() -> None:
    assert InspectionQuery(unlabeled_only=True).matches({}, {})


def test_array_filter_uses_loader_only_annotation_type() -> None:
    array_type: frozenset[InspectionAnnotationType] = frozenset({"array"})
    assert InspectionQuery(annotation_types=array_type).matches(
        {}, {}, extra_annotation_types=array_type
    )
    assert not InspectionQuery(unlabeled_only=True).matches(
        {},
        {},
        extra_annotation_types=array_type,
    )


def test_repeated_metadata_predicates_are_conjunctive() -> None:
    query = InspectionQuery(
        metadata=(
            MetadataPredicate.from_pair("camera.side", "left"),
            MetadataPredicate.from_pair("quality", "rejected"),
        )
    )
    assert not query.matches(_records(), {"camera": {"side": "left"}})


@pytest.mark.parametrize("path", ["", ".", " . "])
def test_metadata_predicate_rejects_empty_path(path: str) -> None:
    with pytest.raises(ValueError, match="paths cannot be empty"):
        MetadataPredicate.from_pair(path, "value")


def test_query_rejects_invalid_filter_combinations() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        InspectionQuery(min_confidence=1.1)
    with pytest.raises(ValueError, match="--min-instances"):
        InspectionQuery(min_instances=-1)
    with pytest.raises(ValueError, match="--max-instances"):
        InspectionQuery(max_instances=-1)
    with pytest.raises(ValueError, match="cannot be greater"):
        InspectionQuery(min_instances=2, max_instances=1)
    with pytest.raises(ValueError, match="--unlabeled-only"):
        InspectionQuery(
            class_names=frozenset({"car"}),
            unlabeled_only=True,
        )
    assert InspectionQuery(
        class_names=frozenset({"car"}),
        class_name_mode="exclude",
        unlabeled_only=True,
    ).matches({}, {})


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (0.5, "0.50"),  # trailing zero: not in the float's repr
        (0.5, "0.5"),
        (1.0, "1"),  # stored as a float, typed as an int
        (1, "1.0"),  # and the other way around
        (12, "12"),
    ],
)
def test_numeric_metadata_filters_compare_numerically(
    stored: float, expected: str
) -> None:
    """Numbers match by value, not by their Python repr.

    Detection metadata decodes to `float`, so a `str()` comparison made
    ``--metadata-filter score 0.50`` silently match nothing.
    """
    assert _metadata_equal(stored, expected)


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (0.5, "0.6"),
        (0.5, "not-a-number"),
        (1, "2"),
    ],
)
def test_numeric_metadata_filters_still_reject_other_values(
    stored: float, expected: str
) -> None:
    assert not _metadata_equal(stored, expected)


def test_bool_metadata_still_matches_by_name_not_by_number() -> None:
    # `bool` is an `int` subclass; it must keep comparing as "true"/"false".
    assert _metadata_equal(True, "true")
    assert _metadata_equal(False, "FALSE")
    assert not _metadata_equal(True, "1")


def test_string_metadata_still_compares_case_insensitively() -> None:
    assert _metadata_equal("Approved", "approved")
    assert not _metadata_equal("approved", "rejected")
