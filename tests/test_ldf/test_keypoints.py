"""Tests for the label-keyed keypoint annotation and its metadata.

Code far outside of this module depends on three things: the shape of the
stored payload, the final order of the keypoints, and the way a name
becomes an index. These tests pin all three in one place.
"""

import json
from typing import Any

import numpy as np
import pydantic
import pytest

from luxonis_ml.ldf import (
    Keypoint,
    KeypointAnnotation,
    KeypointMetadata,
    load_annotation,
)


def payload(
    annotation: KeypointAnnotation,
    keypoint_metadata: KeypointMetadata | None = None,
) -> dict[str, Any]:
    return json.loads(annotation.to_parquet_json(keypoint_metadata))


def test_a_keypoint_is_a_triplet():
    """A `Keypoint` is a named tuple, so a holder does not have to care.

    `to_numpy`, the ``(N, 3K)`` loader output, and every consumer that
    unpacks or indexes a keypoint rely on the tuple.
    """
    keypoint = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 1)]}
    ).keypoints["0"]

    assert isinstance(keypoint, tuple)
    assert keypoint == (0.1, 0.2, 1)
    assert (keypoint.x, keypoint.y, keypoint.visibility) == (0.1, 0.2, 1)


def test_a_keypoint_defaults_to_visible():
    assert Keypoint(0.1, 0.2) == (0.1, 0.2, 2)
    assert KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2)]}
    ).keypoints["0"] == (0.1, 0.2, 2)


def test_keypoint_constraints_survive_the_named_tuple():
    """The field constraints have to keep applying through it.

    Coordinates are clipped into range before the bounds are ever reached,
    so visibility is where the constraint is observable.
    """
    with pytest.raises(pydantic.ValidationError, match="literal_error"):
        KeypointAnnotation.model_validate({"keypoints": {"a": (0.5, 0.5, 7)}})

    # A `Keypoint` built out of range is still clipped, not rejected.
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": {"a": Keypoint(1.5, 0.5, 2)}}
    )
    assert annotation.keypoints["a"] == (1.0, 0.5, 2)


def test_keypoints_accept_named_fields():
    """A named tuple also validates from a mapping of its fields."""
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": {"nose": {"x": 0.1, "y": 0.2, "visibility": 1}}}
    )

    assert annotation.keypoints["nose"] == (0.1, 0.2, 1)


@pytest.mark.parametrize(
    ("keypoint", "match"),
    [
        pytest.param({"y": 0.2, "visibility": 1}, "x", id="no-x"),
        pytest.param({"x": 0.1, "visibility": 1}, "y", id="no-y"),
        pytest.param({"x": 0.1, "y": 0.2, "vis": 1}, "vis", id="typo"),
    ],
)
def test_an_incomplete_named_keypoint_is_an_error(
    keypoint: dict[str, Any], match: str
):
    """A missing or misspelled field must not shift the other values.

    The result would look valid, so nothing downstream could catch it.
    Pydantic names the field that is wrong.
    """
    with pytest.raises(pydantic.ValidationError, match=match):
        KeypointAnnotation.model_validate({"keypoints": {"nose": keypoint}})


@pytest.mark.parametrize(
    "keypoints",
    [
        pytest.param([(0.1, 0.2, 2), (0.3, 0.4, 1)], id="list"),
        pytest.param(
            {"nose": (0.1, 0.2, 2), "left_eye": (0.3, 0.4, 1)}, id="dict"
        ),
        pytest.param(
            {"nose": [0.1, 0.2, 2], "left_eye": [0.3, 0.4, 1]}, id="lists"
        ),
    ],
)
def test_keypoints_are_stored_as_flat_triplets(keypoints: Any):
    """Other modules parse this payload outside of pydantic.

    `coco_exporter` unpacks each entry into three values.
    `ldf_equivalence` enumerates it. A mapping would give
    `ldf_equivalence` garbage, and it would not raise.
    """
    annotation = KeypointAnnotation.model_validate({"keypoints": keypoints})

    assert payload(annotation) == {"keypoints": [[0.1, 0.2, 2], [0.3, 0.4, 1]]}


def test_task_fields_are_not_stored_per_annotation():
    """They describe the task, so `add` hoists them into the dataset."""
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "nose": {"x": 0.1, "y": 0.2, "sigma": 0.1},
                "left_eye": {
                    "x": 0.3,
                    "y": 0.4,
                    "visibility": 1,
                    "sigma": 0.2,
                    "connected_to": "nose",
                },
            }
        }
    )

    assert payload(annotation) == {"keypoints": [[0.1, 0.2, 2], [0.3, 0.4, 1]]}


def test_a_json_round_trip_keeps_the_task_fields():
    """A dump gives the task fields with each keypoint again."""
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "nose": {"x": 0.1, "y": 0.2, "sigma": 0.1},
                "left_eye": {
                    "x": 0.3,
                    "y": 0.4,
                    "visibility": 1,
                    "sigma": 0.2,
                    "connected_to": "nose",
                },
            }
        }
    )

    restored = KeypointAnnotation.model_validate_json(
        annotation.model_dump_json()
    )

    assert restored == annotation
    assert restored.declared_metadata() == KeypointMetadata(
        labels=["nose", "left_eye"], edges=[(0, 1)], sigmas=[0.1, 0.2]
    )


def test_combine_to_numpy_gives_one_flat_row_per_annotation():
    """The loader output contract, and the augmentation stride of three."""
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 1)]}
    )

    np.testing.assert_array_equal(
        KeypointAnnotation.combine_to_numpy([annotation, annotation]),
        [[0.1, 0.2, 2, 0.3, 0.4, 1]] * 2,
    )


def test_combining_different_sized_annotations_is_an_error():
    one = KeypointAnnotation.model_validate({"keypoints": [(0.1, 0.2, 2)]})
    two = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 1)]}
    )

    with pytest.raises(ValueError, match="different numbers of keypoints"):
        KeypointAnnotation.combine_to_numpy([one, two])


def test_declaration_order_does_not_affect_stored_order():
    """Column position is keypoint identity, so it cannot follow dict order.

    Two records naming the same keypoints in a different order have to end
    up in the same columns; otherwise the ``(N, 3K)`` array silently mixes
    up which keypoint is which.
    """
    keypoint_metadata = KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"]
    )
    forwards = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "nose": (0.1, 0.1, 2),
                "left_eye": (0.2, 0.2, 2),
                "right_eye": (0.3, 0.3, 2),
            }
        }
    )
    backwards = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "right_eye": (0.3, 0.3, 2),
                "nose": (0.1, 0.1, 2),
                "left_eye": (0.2, 0.2, 2),
            }
        }
    )

    assert payload(forwards, keypoint_metadata=keypoint_metadata) == payload(
        backwards, keypoint_metadata=keypoint_metadata
    )


def test_omitted_keypoints_are_padded():
    """Annotating only the keypoints that are there is the point of names."""
    keypoint_metadata = KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"]
    )
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": {"left_eye": (0.2, 0.2, 2)}}
    )

    assert payload(annotation, keypoint_metadata=keypoint_metadata) == {
        "keypoints": [[0.0, 0.0, 0], [0.2, 0.2, 2], [0.0, 0.0, 0]]
    }


def test_unnamed_records_align_with_named_ones():
    """A task can mix the two, e.g. after a native export writes the
    skeleton onto only the first record of each task.
    """
    keypoint_metadata = KeypointMetadata(labels=["nose", "left_eye"])
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.1, 2), (0.2, 0.2, 2)]}
    )

    assert payload(annotation, keypoint_metadata=keypoint_metadata) == {
        "keypoints": [[0.1, 0.1, 2], [0.2, 0.2, 2]]
    }


def test_an_unknown_keypoint_is_an_error():
    """A typo must not quietly become a missing keypoint."""
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": {"noze": (0.1, 0.2, 2)}}
    )

    with pytest.raises(ValueError, match="noze"):
        annotation.to_parquet_json(KeypointMetadata(labels=["nose"]))


def test_positional_names_are_not_a_declaration():
    """Otherwise every unnamed record would clash with real names."""
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 1)]}
    )

    assert list(annotation.keypoints) == ["0", "1"]
    assert annotation.declared_metadata() is None


def test_connections_become_edges_by_index():
    """Stored as indices: the consumers all index positionally."""
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "nose": {
                    "x": 0.1,
                    "y": 0.1,
                    "connected_to": ["left_eye", "right_eye"],
                },
                "left_eye": (0.2, 0.2, 2),
                "right_eye": {"x": 0.3, "y": 0.3, "connected_to": "nose"},
            },
        }
    )

    declared = annotation.declared_metadata()
    assert declared is not None
    assert declared.edges == [(0, 1), (0, 2)]


def test_a_state_gives_the_visibility():
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "a": {"x": 0.1, "y": 0.1, "state": "visible"},
                "b": {"x": 0.2, "y": 0.2, "state": "occluded"},
                "c": {"x": 0.3, "y": 0.3, "state": "unlabeled"},
            }
        }
    )

    assert [k.visibility for k in annotation.keypoints.values()] == [2, 1, 0]


def test_the_task_fields_are_given_with_each_keypoint():
    """A record has no lists of edges or sigmas beside its keypoints."""
    with pytest.raises(pydantic.ValidationError, match="sigmas"):
        KeypointAnnotation.model_validate(
            {"keypoints": {"a": (0.1, 0.1, 2)}, "sigmas": [0.1]}
        )


def test_names_and_indices_can_be_mixed():
    keypoint_metadata = KeypointMetadata.model_validate(
        {"labels": ["nose", "left_eye"], "edges": [("nose", 1)]}
    )

    assert keypoint_metadata.edges == [(0, 1)]


def test_referring_by_name_without_labels_is_an_error():
    with pytest.raises(pydantic.ValidationError, match="require"):
        KeypointMetadata.model_validate({"edges": [("nose", "left_eye")]})


def test_an_unknown_name_in_an_edge_is_an_error():
    with pytest.raises(pydantic.ValidationError, match="not part of the task"):
        KeypointMetadata.model_validate(
            {"labels": ["nose"], "edges": [("nose", "left_eye")]}
        )


def test_edges_are_sorted():
    """`set_keypoint_metadata` sorts them, and the hoist agrees."""
    assert KeypointMetadata(edges=[(2, 3), (0, 1)]).edges == [(0, 1), (2, 3)]


def test_an_edge_has_no_direction():
    """No consumer reads the direction, so the lower index comes first.

    An edge named from the other end must not make two records disagree.
    """
    forwards = KeypointMetadata.model_validate(
        {"labels": ["nose", "left_eye"], "edges": [("nose", "left_eye")]}
    )
    backwards = KeypointMetadata.model_validate(
        {"labels": ["nose", "left_eye"], "edges": [("left_eye", "nose")]}
    )

    assert backwards.edges == [(0, 1)]
    assert forwards.merge_with(backwards).edges == [(0, 1)]


@pytest.mark.parametrize(
    ("b", "match"),
    [
        pytest.param({"x": 0.2, "y": 0.2}, "Missing: b", id="some-sigmas"),
        pytest.param(
            {"x": 0.2, "y": 0.2, "sigma": 0.1, "connected_to": "c"},
            "not part of the task",
            id="unknown-connection",
        ),
        pytest.param(
            {"x": 0.2, "y": 0.2, "sigma": 0.1, "state": "hidden"},
            "unknown state 'hidden'",
            id="unknown-state",
        ),
        pytest.param(
            {"x": 0.2, "y": 0.2, "sigma": 0.1, "state": "visible"}
            | {"visibility": 2},
            "both a state and a visibility",
            id="state-and-visibility",
        ),
    ],
)
def test_the_task_fields_must_fit_the_keypoints(b: dict[str, Any], match: str):
    with pytest.raises(pydantic.ValidationError, match=match):
        KeypointAnnotation.model_validate(
            {"keypoints": {"a": {"x": 0.1, "y": 0.1, "sigma": 0.1}, "b": b}}
        )


def test_duplicate_names_are_rejected():
    """A name is the key of a keypoint, so a duplicate drops one.

    An annotation cannot hit this, because its names are dict keys. A
    dataset from an older luxonis-ml can, and `Metadata` must still open
    it. The check thus runs where a name keys a keypoint, and not on
    construction.
    """
    keypoint_metadata = KeypointMetadata(labels=["a", "a", "b"])
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": {"a": (0.1, 0.2, 2), "b": (0.3, 0.4, 1)}}
    )

    assert not keypoint_metadata.has_names
    assert keypoint_metadata.repeated_labels == ["a"]
    with pytest.raises(ValueError, match="Duplicate keypoint names: a"):
        annotation.to_parquet_json(keypoint_metadata)


task_labels = pytest.mark.parametrize(
    "labels",
    [
        pytest.param(["a", "b", "c"], id="unique"),
        # A list needs only the count, so old repeated names still work.
        pytest.param(["a", "a", "b"], id="repeated"),
    ],
)


@task_labels
def test_a_short_list_holds_the_leading_keypoints(labels: list[str]):
    short_list = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 1)]}
    )

    assert payload(short_list, KeypointMetadata(labels=labels)) == {
        "keypoints": [[0.1, 0.2, 2], [0.3, 0.4, 1], [0.0, 0.0, 0]]
    }


@task_labels
def test_a_list_longer_than_the_task_is_an_error(labels: list[str]):
    long_list = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2)] * 4}
    )

    with pytest.raises(ValueError, match="4 keypoints"):
        long_list.to_parquet_json(KeypointMetadata(labels=labels))


def test_an_edge_cannot_refer_to_a_repeated_name():
    """A repeated name does not tell which keypoint an edge joins."""
    with pytest.raises(pydantic.ValidationError, match="names point repeat"):
        KeypointMetadata.model_validate(
            {"labels": ["point", "point", "tip"], "edges": [("point", "tip")]}
        )


def test_a_short_stored_row_is_read_back_padded():
    """A row written before the task had names can be short.

    It holds the leading keypoints, as a short list does on the write
    path. Every row of a task must load to the same width.
    """
    annotation = load_annotation(
        "keypoints",
        {"keypoints": [[0.1, 0.2, 2], [0.3, 0.4, 1]]},
        n_keypoints=3,
    )
    assert isinstance(annotation, KeypointAnnotation)

    assert list(annotation.keypoints.items()) == [
        ("0", (0.1, 0.2, 2)),
        ("1", (0.3, 0.4, 1)),
        ("2", (0.0, 0.0, 0)),
    ]


def test_every_field_is_serialized():
    """The stored keypoint metadata is a plain dump of the four fields."""
    dumped = json.loads(
        KeypointMetadata(
            labels=["left_a", "right_a"],
            edges=[(0, 1)],
            sigmas=[0.1, 0.2],
        ).model_dump_json()
    )

    assert dumped == {
        "labels": ["left_a", "right_a"],
        "edges": [[0, 1]],
        "sigmas": [0.1, 0.2],
    }


def test_legacy_metadata_still_loads():
    """Datasets written before sigmas existed."""
    keypoint_metadata = KeypointMetadata.model_validate(
        {"labels": ["a", "b"], "edges": [[0, 1]]}
    )

    assert keypoint_metadata.sigmas == []


def test_negative_edges_are_still_accepted():
    """`Metadata` is revalidated on every open.

    An older luxonis-ml stored the edges without a check, so a new check
    would make existing datasets impossible to open. `visualizations`
    guards against out-of-range indices when drawing.
    """
    assert KeypointMetadata(edges=[(-1, 3)]).edges == [(-1, 3)]


def test_merging_fills_in_the_gaps():
    merged = KeypointMetadata(labels=["a", "b"]).merge_with(
        KeypointMetadata(sigmas=[0.1, 0.2])
    )

    assert merged.labels == ["a", "b"]
    assert merged.sigmas == [0.1, 0.2]


def test_merging_the_same_names_in_a_different_order_agrees():
    """Only the set of names matters; the first declaration sets the order."""
    merged = KeypointMetadata(labels=["a", "b"]).merge_with(
        KeypointMetadata(labels=["b", "a"])
    )

    assert merged.labels == ["a", "b"]


def test_merging_moves_the_other_records_indices_into_the_merged_order():
    """`edges` and `sigmas` index the labels of one record.

    The merged labels come from the record that declared first, so the
    indices of the other record move to that order.
    """
    first = KeypointMetadata(labels=["nose", "left_eye", "right_eye"])
    second = KeypointMetadata.model_validate(
        {
            "labels": ["right_eye", "left_eye", "nose"],
            "edges": [("nose", "left_eye")],
            "sigmas": [0.035, 0.025, 0.026],
        }
    )

    merged = first.merge_with(second, "task 'pose'")

    assert merged.labels == ["nose", "left_eye", "right_eye"]
    assert merged.edges == [(0, 1)]
    assert merged.sigmas == [0.026, 0.025, 0.035]


def test_merging_compares_the_task_fields_by_name():
    """Two records can give the same edge and sigmas in another key order.

    Their indices differ, but the two declarations agree.
    """
    first = KeypointMetadata.model_validate(
        {
            "labels": ["nose", "left_eye", "right_eye"],
            "edges": [("nose", "left_eye")],
            "sigmas": [0.026, 0.025, 0.035],
        }
    )
    second = KeypointMetadata.model_validate(
        {
            "labels": ["right_eye", "left_eye", "nose"],
            "edges": [("nose", "left_eye")],
            "sigmas": [0.035, 0.025, 0.026],
        }
    )

    merged = first.merge_with(second, "task 'pose'")

    assert merged == first


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"edges": [(0, 1)]}, id="edges"),
        pytest.param({"sigmas": [0.026, 0.025, 0.035]}, id="sigmas"),
    ],
)
def test_merging_finds_a_conflict_behind_equal_indices(
    fields: dict[str, Any],
):
    """Equal indices can name different keypoints in another key order."""
    first = KeypointMetadata.model_validate(
        {"labels": ["nose", "left_eye", "right_eye"], **fields}
    )
    second = KeypointMetadata.model_validate(
        {"labels": ["right_eye", "nose", "left_eye"], **fields}
    )

    with pytest.raises(ValueError, match="Conflicting keypoint metadata"):
        first.merge_with(second, "task 'pose'")


def test_reindexing_to_more_names_moves_the_pairs():
    """A record can name a subset of the keypoints of a task.

    The indices of the record point into its own names. In the names of
    the task, the same keypoints have other indices.
    """
    subset = KeypointMetadata.model_validate(
        {
            "labels": ["right_eye", "left_eye"],
            "edges": [("left_eye", "right_eye")],
        }
    )

    assert subset.reindexed_to(
        ["nose", "left_eye", "right_eye"]
    ) == KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"],
        edges=[(1, 2)],
    )


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        pytest.param({"labels": ["nose", "chin"]}, "chin", id="unknown"),
        pytest.param(
            {"labels": ["nose"], "sigmas": [0.1]}, "1 of the 3", id="sigmas"
        ),
    ],
)
def test_reindexing_rejects_what_it_cannot_move(
    fields: dict[str, Any], match: str
):
    """No index exists for an unknown name, and no sigma for a new one."""
    keypoint_metadata = KeypointMetadata.model_validate(fields)

    with pytest.raises(ValueError, match=match):
        keypoint_metadata.reindexed_to(["nose", "left_eye", "right_eye"])


def test_the_conflict_names_the_task_and_the_fields():
    with pytest.raises(ValueError, match="Conflicting") as info:
        KeypointMetadata(labels=["a", "b"]).merge_with(
            KeypointMetadata(labels=["a", "c"]), "task 'pose'"
        )

    message = str(info.value)
    assert "task 'pose'" in message
    assert "labels" in message
    # Two records annotating different subsets land here, so the message
    # has to point at the fix rather than just at the disagreement.
    assert "name the full set" in message


def test_a_conflict_of_indices_names_the_keypoint_order():
    """`LuxonisDataset.add` moves each record to the stored names first.

    The conflict can thus show indices that no record gives, so the
    message names the keypoint order.
    """
    first = KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"], edges=[(1, 2)]
    )
    second = KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"], edges=[(0, 2)]
    )

    with pytest.raises(ValueError, match="nose, left_eye, right_eye"):
        first.merge_with(second, "task 'pose'")
