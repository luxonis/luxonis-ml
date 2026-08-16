"""Tests for the label-keyed keypoint annotation and its skeleton.

The stored payload shape, the order keypoints end up in, and the way names
are resolved to indices are all relied upon well outside of this module, so
they are pinned here rather than left implicit at each of those sites.
"""

import json
from typing import Any

import numpy as np
import pydantic
import pytest

from luxonis_ml.ldf import (
    Keypoint,
    KeypointAnnotation,
    Skeleton,
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
    annotation: KeypointAnnotation, skeleton: Skeleton | None = None
) -> dict[str, Any]:
    return json.loads(annotation.to_parquet_json(skeleton))


def test_a_keypoint_is_still_a_triplet():
    """It is a named tuple so that nothing holding one has to care.

    Being a real tuple is what keeps `to_numpy`, the ``(N, 3K)`` loader
    output, and every consumer that unpacks or indexes a keypoint working.
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
    """Six modules parse this payload without going through pydantic.

    `coco_exporter` unpacks each entry into exactly three values and
    `ldf_equivalence` enumerates it, the latter silently producing garbage
    rather than raising if it were ever handed a mapping.
    """
    annotation = KeypointAnnotation.model_validate({"keypoints": keypoints})

    assert payload(annotation) == {"keypoints": [[0.1, 0.2, 2], [0.3, 0.4, 1]]}


def test_names_and_skeleton_are_not_stored_per_annotation():
    """They describe the task, so `add` hoists them into the metadata."""
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {"nose": (0.1, 0.2, 2), "left_eye": (0.3, 0.4, 1)},
            "skeleton": {
                "edges": [("nose", "left_eye")],
                "sigmas": [0.1, 0.2],
            },
        }
    )

    stored = annotation.to_parquet_json()

    assert json.loads(stored) == {"keypoints": [[0.1, 0.2, 2], [0.3, 0.4, 1]]}
    assert "skeleton" not in stored
    assert "nose" not in stored


def test_naming_the_keypoints_does_not_grow_the_payload():
    """A named annotation must cost no more on disk than a bare one."""
    named = KeypointAnnotation.model_validate(
        {
            "keypoints": dict.fromkeys(COCO_LABELS, (0.1, 0.2, 2)),
            "skeleton": {"edges": [(0, 1)], "sigmas": [0.05] * 17},
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
    skeleton = Skeleton(labels=["nose", "left_eye", "right_eye"])
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

    assert payload(forwards, skeleton=skeleton) == payload(
        backwards, skeleton=skeleton
    )


def test_omitted_keypoints_are_padded():
    """Annotating only the keypoints that are there is the point of names."""
    skeleton = Skeleton(labels=["nose", "left_eye", "right_eye"])
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": {"left_eye": (0.2, 0.2, 2)}}
    )

    assert payload(annotation, skeleton=skeleton) == {
        "keypoints": [[0.0, 0.0, 0], [0.2, 0.2, 2], [0.0, 0.0, 0]]
    }


def test_unnamed_records_align_with_named_ones():
    """A task can mix the two, e.g. after a native export writes the
    skeleton onto only the first record of each task.
    """
    skeleton = Skeleton(labels=["nose", "left_eye"])
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.1, 2), (0.2, 0.2, 2)]}
    )

    assert payload(annotation, skeleton=skeleton) == {
        "keypoints": [[0.1, 0.1, 2], [0.2, 0.2, 2]]
    }


def test_an_unknown_keypoint_is_an_error():
    """A typo must not quietly become a missing keypoint."""
    with pytest.raises(pydantic.ValidationError, match="noze"):
        KeypointAnnotation.model_validate(
            {
                "keypoints": {"noze": (0.1, 0.2, 2)},
                "skeleton": {"labels": ["nose"]},
            }
        )


def test_a_declared_skeleton_orders_and_pads_on_its_own():
    """No dataset needed: the annotation already knows the full set."""
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {"right_eye": (0.3, 0.3, 2)},
            "skeleton": {"labels": ["nose", "left_eye", "right_eye"]},
        }
    )

    assert list(annotation.keypoints) == ["nose", "left_eye", "right_eye"]
    assert annotation.keypoints["nose"] == (0.0, 0.0, 0)


def test_stored_keypoints_are_read_back_under_their_names():
    """The loader supplies the task's names; without them they are positional."""
    data = json.loads(
        KeypointAnnotation.model_validate(
            {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 1)]}
        ).to_parquet_json()
    )

    named = load_annotation(
        "keypoints", data, keypoint_labels=["nose", "left_eye"]
    )
    positional = load_annotation("keypoints", data)

    assert list(named.keypoints) == ["nose", "left_eye"]  # type: ignore[attr-defined]
    assert list(positional.keypoints) == ["0", "1"]  # type: ignore[attr-defined]
    assert np.allclose(named.to_numpy(), positional.to_numpy())  # type: ignore[attr-defined]


def test_positional_names_are_not_a_declaration():
    """Otherwise every unnamed record would clash with real names."""
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 1)]}
    )

    assert list(annotation.keypoints) == ["0", "1"]
    assert annotation.declared_skeleton() is None


def test_edges_and_flip_pairs_accept_names():
    """Stored as indices: the consumers all index positionally."""
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "nose": (0.1, 0.1, 2),
                "left_eye": (0.2, 0.2, 2),
                "right_eye": (0.3, 0.3, 2),
            },
            "skeleton": {
                "edges": [("nose", "left_eye"), ("nose", "right_eye")],
                "flip_pairs": [("left_eye", "right_eye")],
            },
        }
    )

    assert annotation.skeleton is not None
    assert annotation.skeleton.edges == [(0, 1), (0, 2)]
    assert annotation.skeleton.flip_pairs == [(1, 2)]


def test_names_and_indices_can_be_mixed():
    skeleton = Skeleton.model_validate(
        {"labels": ["nose", "left_eye"], "edges": [("nose", 1)]}
    )

    assert skeleton.edges == [(0, 1)]


def test_referring_by_name_without_labels_is_an_error():
    with pytest.raises(pydantic.ValidationError, match="require"):
        Skeleton.model_validate({"edges": [("nose", "left_eye")]})


def test_an_unknown_name_in_an_edge_is_an_error():
    with pytest.raises(pydantic.ValidationError, match="Unknown keypoint"):
        Skeleton.model_validate(
            {"labels": ["nose"], "edges": [("nose", "left_eye")]}
        )


def test_edges_are_sorted():
    """`set_skeletons` has always sorted them; the hoist must agree."""
    assert Skeleton(edges=[(2, 3), (0, 1)]).edges == [(0, 1), (2, 3)]


def test_flip_pairs_are_normalized():
    assert Skeleton(flip_pairs=[(4, 3), (2, 1)]).flip_pairs == [(1, 2), (3, 4)]


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
        Skeleton(flip_pairs=flip_pairs)


@pytest.mark.parametrize(
    ("n_keypoints", "skeleton", "match"),
    [
        pytest.param(1, {"sigmas": [0.1, 0.2]}, "2 sigmas", id="sigmas"),
        pytest.param(
            2, {"labels": ["a", "b"], "sigmas": [0.1]}, "1 sigmas", id="named"
        ),
        pytest.param(1, {"edges": [(0, 5)]}, "keypoint 5", id="edge-range"),
        pytest.param(
            1, {"flip_pairs": [(0, 5)]}, "keypoint 5", id="flip-range"
        ),
        pytest.param(2, {"labels": ["a", "a"]}, "Duplicate", id="duplicates"),
    ],
)
def test_a_skeleton_must_match_the_keypoints(
    n_keypoints: int, skeleton: dict[str, Any], match: str
):
    with pytest.raises(pydantic.ValidationError, match=match):
        KeypointAnnotation.model_validate(
            {
                "keypoints": [(0.1, 0.2, 2)] * n_keypoints,
                "skeleton": skeleton,
            }
        )


def test_a_positional_list_is_named_by_its_own_skeleton():
    """Writing the keypoints in skeleton order is the obvious thing to do."""
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": [(0.1, 0.1, 2), (0.2, 0.2, 2), (0.3, 0.3, 1)],
            "skeleton": {"labels": ["nose", "left_eye", "right_eye"]},
        }
    )

    assert annotation.keypoints["left_eye"] == (0.2, 0.2, 2)


def test_naming_more_keypoints_than_are_present_is_allowed():
    """That is what makes annotating only some of them possible.

    Only the mapping form can be sparse: in a list, position *is* the
    identity, so a short list has no way to say which keypoints it holds.
    """
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {"b": (0.1, 0.2, 2)},
            "skeleton": {"labels": ["a", "b", "c"]},
        }
    )

    assert len(annotation.keypoints) == 3
    assert annotation.keypoints["a"] == (0.0, 0.0, 0)

    with pytest.raises(pydantic.ValidationError, match="not part of the"):
        KeypointAnnotation.model_validate(
            {
                "keypoints": [(0.1, 0.2, 2)],
                "skeleton": {"labels": ["a", "b", "c"]},
            }
        )


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
    assert Skeleton.infer_flip_pairs(labels) == expected


def test_inferred_flip_pairs_are_a_valid_skeleton():
    flip_pairs = Skeleton.infer_flip_pairs(COCO_LABELS)

    skeleton = Skeleton(labels=COCO_LABELS, flip_pairs=flip_pairs)

    assert skeleton.flip_pairs == flip_pairs


def test_inference_is_not_applied_by_the_model():
    """It belongs to the write paths only.

    `Metadata` is revalidated every time a dataset is opened, so inferring
    here would materialize flip pairs for datasets that never asked for
    them, and a skeleton carrying them cannot be read by older versions.
    """
    assert Skeleton(labels=COCO_LABELS).flip_pairs == []


def test_unused_fields_are_left_out_of_the_serialized_skeleton():
    """Older versions reject a stored skeleton with unknown keys, so a
    dataset only becomes unreadable to them if it truly uses the fields.
    """
    assert json.loads(Skeleton(labels=["a"], edges=[]).model_dump_json()) == {
        "labels": ["a"],
        "edges": [],
    }


def test_used_fields_are_serialized():
    dumped = json.loads(
        Skeleton(
            labels=["left_a", "right_a"],
            flip_pairs=[(0, 1)],
            sigmas=[0.1, 0.2],
        ).model_dump_json()
    )

    assert dumped["flip_pairs"] == [[0, 1]]
    assert dumped["sigmas"] == [0.1, 0.2]


def test_a_legacy_skeleton_still_loads():
    """Datasets written before flip pairs and sigmas existed."""
    skeleton = Skeleton.model_validate(
        {"labels": ["a", "b"], "edges": [[0, 1]]}
    )

    assert skeleton.flip_pairs == []
    assert skeleton.sigmas == []


def test_negative_edges_are_still_accepted():
    """`Metadata` is revalidated on every open, so tightening a field that
    predates this change would make existing datasets impossible to open.
    `visualizations` guards against out-of-range indices when drawing.
    """
    assert Skeleton(edges=[(-1, 3)]).edges == [(-1, 3)]


def test_merging_fills_in_the_gaps():
    merged = Skeleton(labels=["a", "b"]).merge_with(
        Skeleton(sigmas=[0.1, 0.2])
    )

    assert merged.labels == ["a", "b"]
    assert merged.sigmas == [0.1, 0.2]


def test_merging_the_same_names_in_a_different_order_agrees():
    """Only the set of names matters; the first declaration sets the order."""
    merged = Skeleton(labels=["a", "b"]).merge_with(
        Skeleton(labels=["b", "a"])
    )

    assert merged.labels == ["a", "b"]


def test_merging_disagreeing_skeletons_is_an_error():
    with pytest.raises(ValueError, match="Conflicting keypoint skeletons"):
        Skeleton(labels=["a", "b"]).merge_with(
            Skeleton(labels=["a", "c"]), "task 'pose'"
        )


def test_the_conflict_names_the_task_and_the_fields():
    with pytest.raises(ValueError, match="Conflicting") as info:
        Skeleton(labels=["a", "b"]).merge_with(
            Skeleton(labels=["a", "c"]), "task 'pose'"
        )

    message = str(info.value)
    assert "task 'pose'" in message
    assert "labels" in message
    # Two records annotating different subsets land here, so the message
    # has to point at the fix rather than just at the disagreement.
    assert "name the full set" in message
