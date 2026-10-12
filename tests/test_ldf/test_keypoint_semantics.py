"""A keypoint can say what it means: its state, sigma, edges and partner.

The record form keeps everything about one keypoint in one place. The
annotation folds it into the lists of the task, which `LuxonisDataset.add`
moves into the `KeypointMetadata` of the task.
"""

from typing import Any

import pydantic
import pytest

from luxonis_ml.ldf import KeypointAnnotation, KeypointMetadata
from luxonis_ml.utils.logging import log_once

TILE = {
    "top_left": {"x": 0.2, "y": 0.2, "picture": "top_left"},
    "top_right": {"x": 0.8, "y": 0.2, "picture": "top_right"},
    "bottom_right": {"x": 0.8, "y": 0.8, "picture": "bottom_right"},
    "bottom_left": {"x": 0.2, "y": 0.8, "picture": "bottom_left"},
}


def test_each_keypoint_can_say_what_it_means():
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "nose": {"x": 0.5, "y": 0.3, "sigma": 0.026},
                "left_eye": {
                    "x": 0.45,
                    "y": 0.25,
                    "state": "occluded",
                    "sigma": 0.025,
                    "connected_to": "nose",
                    "mirror": "right_eye",
                },
                "right_eye": {
                    "x": 0.55,
                    "y": 0.25,
                    "sigma": 0.025,
                    "connected_to": ["nose", "left_eye"],
                },
            }
        }
    )

    assert annotation.to_numpy().tolist() == [
        *(0.5, 0.3, 2.0),
        *(0.45, 0.25, 1.0),
        *(0.55, 0.25, 2.0),
    ]
    assert annotation.declared_metadata() == KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"],
        edges=[(0, 1), (0, 2), (1, 2)],
        flip_pairs=[(1, 2)],
        sigmas=[0.026, 0.025, 0.025],
    )


def test_a_partner_named_on_both_sides_is_one_pair():
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {
                "left": {"x": 0.4, "y": 0.5, "mirror": "right"},
                "right": {"x": 0.6, "y": 0.5, "mirror": "left"},
            }
        }
    )

    assert annotation.flip_pairs == [(0, 1)]


def test_the_corners_of_an_object_without_a_front_name_places():
    annotation = KeypointAnnotation.model_validate({"keypoints": TILE})

    metadata = annotation.declared_metadata()

    assert metadata is not None
    assert metadata.picture == [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    ]
    assert metadata.flip_pairs == []


def test_a_list_of_task_fields_still_works():
    annotation = KeypointAnnotation.model_validate(
        {
            "keypoints": {"a": (0.4, 0.5), "b": (0.6, 0.5)},
            "flip_pairs": [("a", "b")],
            "sigmas": [0.1, 0.1],
        }
    )

    assert (annotation.flip_pairs, annotation.sigmas) == ([(0, 1)], [0.1, 0.1])


@pytest.mark.parametrize(
    ("keypoints", "extra", "match"),
    [
        pytest.param(
            {"a": {"x": 0.1, "y": 0.1, "state": "hidden"}},
            {},
            "unknown state 'hidden'",
            id="unknown-state",
        ),
        pytest.param(
            {"a": {"x": 0.1, "y": 0.1, "state": "visible", "visibility": 2}},
            {},
            "both a state and a visibility",
            id="state-and-visibility",
        ),
        pytest.param(
            {"a": {"x": 0.1, "y": 0.1, "sigma": 0.1}, "b": (0.2, 0.2)},
            {},
            "Missing: b",
            id="some-sigmas",
        ),
        pytest.param(
            {"a": {"x": 0.1, "y": 0.1, "sigma": 0.1}},
            {"sigmas": [0.1]},
            "either with each keypoint or as a list",
            id="sigmas-twice",
        ),
        pytest.param(
            {"a": {"x": 0.1, "y": 0.1, "mirror": "c"}, "b": (0.2, 0.2)},
            {},
            "not part of the task",
            id="unknown-partner",
        ),
        pytest.param(
            {"a": {"x": 0.1, "y": 0.1, "mirror": "a"}},
            {},
            "onto itself",
            id="own-partner",
        ),
        pytest.param(
            {
                "a": {"x": 0.1, "y": 0.1, "mirror": "b"},
                "b": {"x": 0.2, "y": 0.1, "mirror": "c"},
                "c": (0.3, 0.1),
            },
            {},
            "both flip pairs",
            id="two-partners",
        ),
        pytest.param(
            {
                "a": {"x": 0.1, "y": 0.1, "picture": "top_left"},
                "b": {"x": 0.9, "y": 0.1, "picture": "top_right"},
            },
            {},
            "Missing: bottom_left, bottom_right",
            id="half-the-corners",
        ),
        pytest.param(
            {"a": {"x": 0.1, "y": 0.1, "picture": "center", "mirror": "b"}}
            | {"b": (0.2, 0.2)},
            {},
            "both a mirror partner and a place",
            id="partner-and-place",
        ),
        pytest.param(
            {
                "a": {"x": 0.5, "y": 0.5, "picture": "center"},
                "b": {"x": 0.5, "y": 0.5, "picture": "center"},
            },
            {},
            "More than one keypoint has the place center",
            id="place-twice",
        ),
    ],
)
def test_a_wrong_meaning_is_an_error(
    keypoints: dict[str, Any], extra: dict[str, Any], match: str
):
    with pytest.raises(pydantic.ValidationError, match=match):
        KeypointAnnotation.model_validate({"keypoints": keypoints, **extra})


def test_corners_that_are_not_at_their_places_give_a_warning(
    warnings_log: list[str], monkeypatch: pytest.MonkeyPatch
):
    """The record says ``top_left`` for the corner at the top right."""
    # The warning is logged once per process.
    monkeypatch.setattr(log_once, "_cache", set(), raising=False)
    mirrored = {
        name: {**keypoint, "x": 1 - keypoint["x"]}
        for name, keypoint in TILE.items()
    }

    KeypointAnnotation.model_validate({"keypoints": mirrored})

    assert any("not at those places" in message for message in warnings_log)


def test_new_names_move_the_places():
    metadata = KeypointMetadata.model_validate(
        {
            "labels": list(TILE),
            "picture": [keypoint["picture"] for keypoint in TILE.values()],
        }
    )

    moved = metadata.reindexed_to(["bottom_left", *list(TILE)[:3], "note"])

    assert moved.picture == [
        "bottom_left",
        "top_left",
        "top_right",
        "bottom_right",
        None,
    ]
