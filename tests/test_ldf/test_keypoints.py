import json
from decimal import Decimal

import numpy as np
import pydantic
import pytest

from luxonis_ml.ldf import (
    Detection,
    Keypoint,
    KeypointAnnotation,
    KeypointMetadata,
    KeypointState,
    load_annotation,
)

NOSE_AND_EYE = {
    "keypoints": {
        "nose": {
            "x": 0.5,
            "y": 0.3,
            "sigma": 0.026,
        },
        "left_eye": {
            "x": 0.4,
            "y": 0.2,
            "state": "occluded",
            "sigma": 0.025,
            "connected_to": "nose",
        },
    }
}


@pytest.mark.parametrize("state", ["occluded", 1, KeypointState.OCCLUDED])
def test_the_state_is_a_word_or_a_number(state: str | int):
    keypoint = Keypoint.model_validate({"x": 0.1, "y": 0.2, "state": state})

    assert keypoint.state is KeypointState.OCCLUDED


def test_a_tuple_without_a_state_is_visible():
    assert Keypoint.model_validate((0.1, 0.2)).state is KeypointState.VISIBLE


@pytest.mark.parametrize("x", ["1.5", Decimal("1.5")], ids=["str", "decimal"])
def test_a_number_that_is_not_a_float_is_clipped(x: object):
    assert Keypoint.model_validate((x, 0.2)).x == 1.0


@pytest.mark.parametrize(
    "value",
    [
        {"x": 0.1, "y": 0.2, "state": "hidden"},
        {"x": 0.1, "y": 0.2, "state": 3},
        {"x": 0.1, "y": 0.2, "visibility": 2},
        (0.1,),
        (0.1, 0.2, 2, 0.5),
    ],
    ids=["unknown-word", "unknown-number", "visibility-key", "one", "four"],
)
def test_a_keypoint_rejects_what_it_does_not_define(value: object):
    with pytest.raises(pydantic.ValidationError):
        Keypoint.model_validate(value)


def test_a_list_is_keyed_by_position():
    annotation = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2), (0.3, 0.4, 0)]}
    )

    assert list(annotation.keypoints) == ["0", "1"]


@pytest.mark.parametrize(
    "keypoints",
    [
        np.array([[0.1, 0.2, 2], [0.3, 0.4, 1]]),
        list(np.array([[0.1, 0.2, 2], [0.3, 0.4, 1]])),
    ],
    ids=["array", "list_of_rows"],
)
def test_a_numpy_array_is_keyed_by_position(keypoints: object):
    annotation = KeypointAnnotation.model_validate({"keypoints": keypoints})

    assert json.loads(annotation.to_parquet_json()) == {
        "keypoints": {"0": [0.1, 0.2, 2], "1": [0.3, 0.4, 1]}
    }


def test_a_connection_needs_both_keypoints_in_the_annotation():
    with pytest.raises(pydantic.ValidationError, match="nose"):
        KeypointAnnotation.model_validate(
            {
                "keypoints": {
                    "left_eye": {
                        "x": 0.4,
                        "y": 0.2,
                        "connected_to": "nose",
                    },
                }
            }
        )


def test_to_numpy_follows_the_labels():
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "left_eye": (0.4, 0.2, "occluded"),
                "nose": (0.5, 0.3),
            }
        }
    )

    row = annotation.to_numpy(["nose", "right_eye", "left_eye"])

    assert row.tolist() == [0.5, 0.3, 2, 0, 0, 0, 0.4, 0.2, 1]


def test_combine_to_numpy_gives_each_name_one_column():
    """The columns follow the names in the order they first occur."""
    nose = KeypointAnnotation.model_validate(
        {"keypoints": {"nose": (0.5, 0.3)}}
    )
    eye = KeypointAnnotation.model_validate({"keypoints": {"eye": (0.4, 0.2)}})

    rows = KeypointAnnotation.combine_to_numpy([nose, eye])

    assert rows.tolist() == [[0.5, 0.3, 2, 0, 0, 0], [0, 0, 0, 0.4, 0.2, 2]]


def test_the_stored_payload_holds_only_the_coordinates():
    annotation = KeypointAnnotation.model_validate(NOSE_AND_EYE)

    payload = json.loads(annotation.to_parquet_json())
    stored = load_annotation("keypoints", payload)
    assert isinstance(stored, KeypointAnnotation)

    assert payload == {
        "keypoints": {"nose": [0.5, 0.3, 2], "left_eye": [0.4, 0.2, 1]}
    }
    assert np.array_equal(stored.to_numpy(), annotation.to_numpy())


def test_metadata_takes_the_task_fields_of_each_keypoint():
    annotation = KeypointAnnotation.model_validate(NOSE_AND_EYE)

    metadata = KeypointMetadata.from_annotation(annotation)

    assert metadata.labels == ["nose", "left_eye"]
    assert metadata.edges == [(0, 1)]
    assert metadata.sigmas == [0.026, 0.025]


def test_sigmas_are_empty_unless_each_keypoint_has_one():
    metadata = KeypointMetadata.model_validate(
        {"keypoints": {"nose": {"sigma": 0.026}, "left_eye": {}}}
    )

    assert metadata.sigmas == []


def test_an_edge_has_no_direction():
    metadata = KeypointMetadata.model_validate(
        {
            "keypoints": {
                "nose": {"connected_to": ["left_eye"]},
                "left_eye": {"connected_to": ["nose"]},
            }
        }
    )

    assert metadata.edges == [(0, 1)]


def test_metadata_rejects_a_connection_to_an_unknown_keypoint():
    with pytest.raises(pydantic.ValidationError, match="nose"):
        KeypointMetadata.model_validate(
            {"keypoints": {"left_eye": {"connected_to": ["nose"]}}}
        )


def test_merge_appends_new_names_and_keeps_both_sides():
    mine = KeypointMetadata.model_validate(
        {"keypoints": {"nose": {}, "left_eye": {"connected_to": ["nose"]}}}
    )
    theirs = KeypointMetadata.model_validate(
        {
            "keypoints": {
                "right_eye": {},
                "nose": {
                    "sigma": 0.026,
                    "connected_to": ["right_eye"],
                },
            }
        }
    )

    merged = mine.merge(theirs)

    assert merged.labels == ["nose", "left_eye", "right_eye"]
    assert merged.edges == [(0, 1), (0, 2)]
    assert merged.keypoints["nose"].sigma == 0.026


def test_merge_rejects_two_sigmas_for_one_keypoint():
    mine = KeypointMetadata.model_validate(
        {"keypoints": {"nose": {"sigma": 0.026}}}
    )
    theirs = KeypointMetadata.model_validate(
        {"keypoints": {"nose": {"sigma": 0.1}}}
    )

    with pytest.raises(ValueError, match="nose"):
        mine.merge(theirs)


def test_box_relative_keypoints_keep_their_task_fields():
    detection = Detection.model_validate(
        {
            "scale_to_boxes": True,
            "boundingbox": {"x": 0.2, "y": 0.2, "w": 0.5, "h": 0.5},
            "keypoints": {
                "keypoints": {
                    "nose": {"x": 0.5, "y": 0.5},
                    "left_eye": {
                        "x": 0.0,
                        "y": 0.0,
                        "state": "occluded",
                        "sigma": 0.025,
                        "connected_to": "nose",
                    },
                }
            },
        }
    )

    assert detection.keypoints is not None
    eye = detection.keypoints.keypoints["left_eye"]
    assert (eye.x, eye.y, eye.state) == (0.2, 0.2, KeypointState.OCCLUDED)
    assert (eye.sigma, eye.connected_to) == (0.025, ["nose"])
