"""Tests for the label-keyed keypoint annotation and its metadata.

Code far outside of this module depends on three things: the shape of the
stored payload, the final order of the keypoints, and the way a name
becomes an index. These tests pin all three in one place.
"""

import json
from typing import Any

import pydantic
import pytest

from luxonis_ml.ldf import (
    Keypoint,
    KeypointAnnotation,
    KeypointMetadata,
    load_annotation,
)

COCO_LABELS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
COCO_FLIP_PAIRS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]


def payload(
    annotation: KeypointAnnotation,
    keypoint_metadata: KeypointMetadata | None = None,
) -> dict[str, Any]:
    return json.loads(annotation.to_parquet_json(keypoint_metadata))


def test_a_keypoint_is_still_a_triplet():
    """A `Keypoint` is a named tuple, so a holder does not have to care.

    `to_numpy`, the ``(N, 3K)`` loader output, and every consumer that
    unpacks or indexes a keypoint rely on the tuple.
    """
    keypoint = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 1)]}
    ).keypoints["0"]

    assert keypoint == (0.1, 0.2, 1)
    assert tuple(keypoint) == (0.1, 0.2, 1)
    assert len(keypoint) == 3
    assert keypoint[0] == 0.1
    x, y, visibility = keypoint
    assert (x, y, visibility) == (0.1, 0.2, 1)
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
    """A gap in the mapping used to shift the later values left.

    ``{"y": 0.2, "visibility": 1}`` became ``(0.2, 1.0, 2)``, and a typo
    such as ``vis`` was dropped without a word. Both give coordinates that
    look valid, so nothing downstream can catch them. The mapping has to
    reach pydantic, which names the field that is wrong.
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
    """Six modules parse this payload outside of pydantic.

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
            "keypoints": {"nose": (0.1, 0.2, 2), "left_eye": (0.3, 0.4, 1)},
            "edges": [("nose", "left_eye")],
            "sigmas": [0.1, 0.2],
        }
    )

    stored = annotation.to_parquet_json()

    assert json.loads(stored) == {"keypoints": [[0.1, 0.2, 2], [0.3, 0.4, 1]]}
    assert "edges" not in stored
    assert "sigmas" not in stored
    assert "nose" not in stored


def test_naming_the_keypoints_does_not_grow_the_payload():
    """A named annotation must cost no more on disk than a bare one."""
    named = KeypointAnnotation.model_validate(
        {
            "keypoints": dict.fromkeys(COCO_LABELS, (0.1, 0.2, 2)),
            "edges": [(0, 1)],
            "sigmas": [0.05] * 17,
        }
    )
    bare = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2)] * 17}
    )

    assert named.to_parquet_json() == bare.to_parquet_json()


def test_to_numpy_shapes_are_unchanged():
    """The loader output contract, and the augmentation stride of three."""
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 1)]}
    )

    assert annotation.to_numpy().shape == (6,)
    assert KeypointAnnotation.combine_to_numpy(
        [annotation, annotation, annotation]
    ).shape == (3, 6)


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


def test_edges_and_flip_pairs_accept_names():
    """Stored as indices: the consumers all index positionally."""
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "nose": (0.1, 0.1, 2),
                "left_eye": (0.2, 0.2, 2),
                "right_eye": (0.3, 0.3, 2),
            },
            "edges": [("nose", "left_eye"), ("nose", "right_eye")],
            "flip_pairs": [("left_eye", "right_eye")],
        }
    )

    assert annotation.edges == [(0, 1), (0, 2)]
    assert annotation.flip_pairs == [(1, 2)]


def test_names_and_indices_can_be_mixed():
    keypoint_metadata = KeypointMetadata.model_validate(
        {"labels": ["nose", "left_eye"], "edges": [("nose", 1)]}
    )

    assert keypoint_metadata.edges == [(0, 1)]


def test_referring_by_name_without_labels_is_an_error():
    with pytest.raises(pydantic.ValidationError, match="require"):
        KeypointMetadata.model_validate({"edges": [("nose", "left_eye")]})


def test_an_unknown_name_in_an_edge_is_an_error():
    with pytest.raises(pydantic.ValidationError, match="Unknown keypoint"):
        KeypointMetadata.model_validate(
            {"labels": ["nose"], "edges": [("nose", "left_eye")]}
        )


def test_edges_are_sorted():
    """`set_keypoint_metadata` sorts them, and the hoist agrees."""
    assert KeypointMetadata(edges=[(2, 3), (0, 1)]).edges == [(0, 1), (2, 3)]


def test_an_edge_has_no_direction():
    """Only the edge list was sorted, never the two ends of an edge.

    An edge named from the other end thus gave an unequal record, and
    `merge_with` aborted the whole `add` over an edge set that agrees.
    No consumer reads the direction, so the ends order like a flip pair.
    """
    forwards = KeypointMetadata.model_validate(
        {"labels": ["nose", "left_eye"], "edges": [("nose", "left_eye")]}
    )
    backwards = KeypointMetadata.model_validate(
        {"labels": ["nose", "left_eye"], "edges": [("left_eye", "nose")]}
    )

    assert backwards.edges == [(0, 1)]
    assert forwards.merge_with(backwards).edges == [(0, 1)]


def test_flip_pairs_are_normalized():
    assert KeypointMetadata(flip_pairs=[(4, 3), (2, 1)]).flip_pairs == [
        (1, 2),
        (3, 4),
    ]


@pytest.mark.parametrize(
    ("flip_pairs", "match"),
    [
        ([(1, 1)], "onto itself"),
        ([(1, 2), (2, 3)], "disjoint"),
    ],
)
def test_invalid_flip_pairs_are_rejected(
    flip_pairs: list[tuple[int, int]], match: str
):
    with pytest.raises(pydantic.ValidationError, match=match):
        KeypointMetadata(flip_pairs=flip_pairs)


@pytest.mark.parametrize(
    ("n_keypoints", "fields", "match"),
    [
        pytest.param(1, {"sigmas": [0.1, 0.2]}, "2 sigmas", id="sigmas"),
        pytest.param(1, {"edges": [(0, 5)]}, "keypoint 5", id="edge-range"),
        pytest.param(
            1, {"flip_pairs": [(0, 5)]}, "keypoint 5", id="flip-range"
        ),
    ],
)
def test_the_task_fields_must_match_the_keypoints(
    n_keypoints: int, fields: dict[str, Any], match: str
):
    with pytest.raises(pydantic.ValidationError, match=match):
        KeypointAnnotation.model_validate(
            {"keypoints": [(0.1, 0.2, 2)] * n_keypoints, **fields}
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


def test_repeated_names_still_align_a_record_without_names():
    """A record without names needs only the number of keypoints.

    The alignment rejected repeated names for such a record too. `add`
    thus left a task from an older luxonis-ml out of the alignment, and
    nothing checked the number of keypoints of its records.
    """
    keypoint_metadata = KeypointMetadata(labels=["a", "a", "b"])
    short_list = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 1)]}
    )
    long_list = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2)] * 4}
    )

    assert payload(short_list, keypoint_metadata=keypoint_metadata) == {
        "keypoints": [[0.1, 0.2, 2], [0.3, 0.4, 1], [0.0, 0.0, 0]]
    }
    with pytest.raises(ValueError, match="4 keypoints"):
        long_list.to_parquet_json(keypoint_metadata)


def test_an_edge_cannot_refer_to_a_repeated_name():
    """A repeated name does not tell which keypoint an edge joins.

    The lookup took the last keypoint with that name, and nothing warned.
    """
    with pytest.raises(pydantic.ValidationError, match="names point repeat"):
        KeypointMetadata.model_validate(
            {"labels": ["point", "point", "tip"], "edges": [("point", "tip")]}
        )


def test_an_annotation_may_hold_fewer_keypoints_than_the_task():
    """A sparse mapping is padded when the payload is written.

    In a list the position is the identity, so a short list holds the
    leading keypoints. Its keys ``"0"``, ``"1"``, ... were read as unknown
    names, and the record failed. A task can hold records with fewer
    keypoints, so a native export of such a task did not import.
    """
    keypoint_metadata = KeypointMetadata(labels=["a", "b", "c"])
    sparse = KeypointAnnotation.model_validate(
        {"keypoints": {"b": (0.1, 0.2, 2)}}
    )

    assert payload(sparse, keypoint_metadata=keypoint_metadata) == {
        "keypoints": [[0.0, 0.0, 0], [0.1, 0.2, 2], [0.0, 0.0, 0]]
    }

    short_list = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 1)]}
    )

    assert payload(short_list, keypoint_metadata=keypoint_metadata) == {
        "keypoints": [[0.1, 0.2, 2], [0.3, 0.4, 1], [0.0, 0.0, 0]]
    }


def test_a_list_longer_than_the_task_is_an_error():
    """No label is left for the keypoints past the end of the task."""
    long_list = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2)] * 4}
    )

    with pytest.raises(ValueError, match="4 keypoints"):
        long_list.to_parquet_json(KeypointMetadata(labels=["a", "b", "c"]))


def test_a_short_stored_row_is_read_back_padded():
    """A row written before the task had names can be short.

    Such a row holds the leading keypoints, as a short list does on the
    write path. The loader read it under positional keys instead, so one
    sample of the task had a narrower keypoint array than the others.
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


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        pytest.param(COCO_LABELS, COCO_FLIP_PAIRS, id="coco"),
        pytest.param(["l_wrist", "r_wrist"], [(0, 1)], id="single-letter"),
        pytest.param(["wrist_left", "wrist_right"], [(0, 1)], id="suffix"),
        pytest.param(["wrist_l", "wrist_r"], [(0, 1)], id="suffix-letter"),
        pytest.param(
            ["LEFT_EYE", "Right_Eye"], [(0, 1)], id="case-insensitive"
        ),
        pytest.param(["left-eye", "right eye"], [(0, 1)], id="separators"),
        pytest.param(
            ["left_eye", "right_eye", "l_ear", "r_ear"],
            [(0, 1), (2, 3)],
            id="mixed-conventions",
        ),
        pytest.param(["nose", "throat"], [], id="no-markers"),
        pytest.param(["left_eye", "nose"], [], id="partner-missing"),
        pytest.param(
            ["bright_spot", "left_x", "right_x"],
            [(1, 2)],
            id="no-substring-match",
        ),
        pytest.param(["left", "right"], [], id="marker-only"),
        pytest.param([str(i) for i in range(5)], [], id="positional"),
        pytest.param(
            ["left_eye", "right_eye", "eye_left"], [], id="ambiguous"
        ),
    ],
)
def test_infer_flip_pairs(labels: list[str], expected: list[tuple[int, int]]):
    """A wrong pair mirrors the wrong keypoints without ever failing, so
    matching stays narrow and refuses anything ambiguous.
    """
    assert KeypointMetadata.infer_flip_pairs(labels) == expected


def test_inferred_flip_pairs_are_valid_metadata():
    flip_pairs = KeypointMetadata.infer_flip_pairs(COCO_LABELS)

    keypoint_metadata = KeypointMetadata(
        labels=COCO_LABELS, flip_pairs=flip_pairs
    )

    assert keypoint_metadata.flip_pairs == flip_pairs


def test_inference_is_not_applied_by_the_model():
    """Only a write path infers them, never validation.

    See `LuxonisDataset._fill_in_flip_pairs` for why.
    """
    assert KeypointMetadata(labels=COCO_LABELS).flip_pairs == []


def test_every_field_is_serialized():
    """The stored keypoint metadata is a plain dump of the four fields."""
    dumped = json.loads(
        KeypointMetadata(
            labels=["left_a", "right_a"],
            edges=[(0, 1)],
            flip_pairs=[(0, 1)],
            sigmas=[0.1, 0.2],
        ).model_dump_json()
    )

    assert dumped == {
        "labels": ["left_a", "right_a"],
        "edges": [[0, 1]],
        "flip_pairs": [[0, 1]],
        "sigmas": [0.1, 0.2],
    }


def test_legacy_metadata_still_loads():
    """Datasets written before flip pairs and sigmas existed."""
    keypoint_metadata = KeypointMetadata.model_validate(
        {"labels": ["a", "b"], "edges": [[0, 1]]}
    )

    assert keypoint_metadata.flip_pairs == []
    assert keypoint_metadata.sigmas == []


def test_negative_edges_are_still_accepted():
    """`Metadata` is revalidated on every open, so tightening a field that
    predates this change would make existing datasets impossible to open.
    `visualizations` guards against out-of-range indices when drawing.
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
    """`edges`, `flip_pairs` and `sigmas` index the labels of one record.

    The merged labels come from the record that declared first, so the
    other record's indices belong to another order. The merge used to
    copy them unchanged, so `add` silently stored an edge between the
    wrong keypoints and gave the wrong keypoint each sigma.
    """
    first = KeypointMetadata(labels=["nose", "left_eye", "right_eye"])
    second = KeypointMetadata.model_validate(
        {
            "labels": ["right_eye", "left_eye", "nose"],
            "edges": [("nose", "left_eye")],
            "flip_pairs": [("left_eye", "right_eye")],
            "sigmas": [0.035, 0.025, 0.026],
        }
    )

    merged = first.merge_with(second, "task 'pose'")

    assert merged.labels == ["nose", "left_eye", "right_eye"]
    assert merged.edges == [(0, 1)]
    assert merged.flip_pairs == [(1, 2)]
    assert merged.sigmas == [0.026, 0.025, 0.035]


def test_merging_compares_the_task_fields_by_name():
    """The merge compared the indices before it moved them.

    Two records that give the same edge and the same sigmas in another
    key order hold different indices. `add` thus aborted with a conflict
    between two declarations that agree.
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
    fields: dict[str, list[tuple[int, int]] | list[float]],
):
    """Equal indices can name different keypoints in another key order.

    The merge saw no conflict there. It kept the first record, and the
    edge ``right_eye``-``nose`` or the sigmas of the second record were
    lost without a warning.
    """
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
    the task, the same keypoints have other indices. The move looked up
    each name of the task among the names of the record, so a name that
    only the task has raised ``'nose' is not in list``.
    """
    subset = KeypointMetadata.model_validate(
        {
            "labels": ["right_eye", "left_eye"],
            "edges": [("left_eye", "right_eye")],
            "flip_pairs": [("left_eye", "right_eye")],
        }
    )

    assert subset.reindexed_to(
        ["nose", "left_eye", "right_eye"]
    ) == KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"],
        edges=[(1, 2)],
        flip_pairs=[(1, 2)],
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
    fields: dict[str, list[str] | list[float]], match: str
):
    """No index exists for an unknown name, and no sigma for a new one."""
    keypoint_metadata = KeypointMetadata.model_validate(fields)

    with pytest.raises(ValueError, match=match):
        keypoint_metadata.reindexed_to(["nose", "left_eye", "right_eye"])


def test_merging_disagreeing_metadata_is_an_error():
    with pytest.raises(ValueError, match="Conflicting keypoint metadata"):
        KeypointMetadata(labels=["a", "b"]).merge_with(
            KeypointMetadata(labels=["a", "c"]), "task 'pose'"
        )


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

    The conflict can thus show indices that no record gives. The message
    did not tell which keypoints the indices refer to.
    """
    first = KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"], edges=[(1, 2)]
    )
    second = KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"], edges=[(0, 2)]
    )

    with pytest.raises(ValueError, match="nose, left_eye, right_eye"):
        first.merge_with(second, "task 'pose'")
