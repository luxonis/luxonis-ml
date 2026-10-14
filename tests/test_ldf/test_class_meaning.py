"""The ``class`` of a record can say what turns and mirrors make of it."""

import numpy as np
import pytest
from pydantic import ValidationError

from luxonis_ml.ldf import ClassMeaning, ClassMetadata, Detection, Direction
from luxonis_ml.ldf.class_meaning import nearest_symmetry, turn

ARROWS: dict[Direction, str] = {
    "up": "arrow_up",
    "right": "arrow_right",
    "down": "arrow_down",
    "left": "arrow_left",
}


def test_a_class_mapping_splits_into_the_name_and_the_meaning():
    detection = Detection.model_validate(
        {
            "class": {
                "name": "left_hand",
                "mirror": "right_hand",
            },
        }
    )

    assert detection.class_name == "left_hand"
    assert detection.class_meaning == ClassMeaning(mirror="right_hand")
    assert "class_meaning" not in detection.model_dump()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"mirror": "right_hand"}, "name"),
        ({"name": "hand", "mirror": "hand"}, "itself"),
        ({"name": "arrow_x", "direction": ARROWS}, "arrow_x"),
        ({"name": "a", "mirror": "b", "direction": ARROWS}, "one of"),
        ({"name": "a", "upright": False}, "upright"),
    ],
)
def test_a_class_mapping_that_cannot_hold_is_rejected(value, message):
    with pytest.raises(ValidationError, match=message):
        Detection.model_validate({"class": value})


def test_a_direction_set_needs_all_four_directions():
    with pytest.raises(ValidationError, match="cardinal"):
        ClassMeaning(direction={"up": "arrow_up", "left": "arrow_left"})


def test_a_pair_swaps_after_each_mirror_and_stays_after_each_turn():
    table = ClassMetadata(mirror_pairs=[("left", "right")]).table()

    assert table["left"]["hflip"] == "right"
    assert table["left"]["transpose"] == "right"
    assert table["left"]["rotate_90"] == "left"
    assert table["right"]["vflip"] == "left"


def test_an_upright_pair_allows_only_the_horizontal_mirror():
    metadata = ClassMetadata().with_meanings(
        {
            "b": ClassMeaning(upright=True, mirror="d"),
        }
    )

    table = metadata.table()
    assert metadata.upright == ["b", "d"]
    assert table["b"]["hflip"] == "d"
    assert table["d"]["hflip"] == "b"
    assert table["b"]["rotate_180"] is None
    assert table["b"]["vflip"] is None


def test_a_direction_set_turns_with_the_image():
    table = ClassMetadata(direction_sets=[ARROWS]).table()

    assert table["arrow_up"]["rotate_90"] == "arrow_left"
    assert table["arrow_up"]["rotate_270"] == "arrow_right"
    assert table["arrow_left"]["hflip"] == "arrow_right"
    assert table["arrow_up"]["transpose"] == "arrow_left"


def test_a_symmetry_table_fills_in_the_classes_it_names():
    metadata = ClassMetadata().with_meanings(
        {
            "b": ClassMeaning(
                symmetry={
                    "hflip": "d",
                    "vflip": "p",
                    "rotate_180": "q",
                }
            ),
        }
    )

    table = metadata.table()
    assert metadata.classes() == {"b", "d", "p", "q"}
    assert table["q"]["hflip"] == "p"
    assert table["d"]["rotate_90"] is None


def test_a_symmetry_table_that_contradicts_itself_is_rejected():
    with pytest.raises(ValueError, match="b"):
        ClassMetadata(
            symmetry={
                "b": {
                    "hflip": "d",
                    "vflip": "d",
                    "rotate_180": "d",
                },
            }
        ).table()


def test_a_new_meaning_replaces_the_relations_of_its_classes():
    metadata = ClassMetadata(mirror_pairs=[("left", "right")])

    metadata = metadata.with_meanings(
        {
            "right": ClassMeaning(mirror="other"),
        }
    )
    assert metadata.mirror_pairs == [("right", "other")]

    metadata = metadata.with_meanings(
        {
            "other": None,
        }
    )
    assert metadata.classes() == set()


def test_the_meanings_rebuild_the_same_metadata():
    metadata = ClassMetadata(
        mirror_pairs=[("left", "right"), ("b", "d")],
        upright=["b", "d", "digit_6"],
        direction_sets=[ARROWS],
    )

    assert ClassMetadata().with_meanings(metadata.meanings()) == metadata


def test_a_class_in_two_relations_is_rejected():
    with pytest.raises(ValidationError, match="left"):
        ClassMetadata(
            mirror_pairs=[("left", "right")],
            direction_sets=[{**ARROWS, "up": "left"}],
        )


@pytest.mark.parametrize(
    ("matrix", "symmetry"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], "identity"),
        ([[-0.9, 0.1], [0.0, 1.2]], "hflip"),
        ([[0.0, 1.0], [-1.0, 0.0]], "rotate_90"),
        ([[0.9, -0.4], [0.4, 0.9]], "identity"),
        ([[0.4, 0.9], [-0.9, 0.4]], "rotate_90"),
        ([[3.0, 0.0], [0.0, 0.5]], None),
        ([[1.0, 0.0], [0.0, 0.0]], None),
    ],
)
def test_nearest_symmetry_rounds_a_measured_map(matrix, symmetry):
    assert nearest_symmetry(np.array(matrix)) == symmetry


def test_turn_moves_a_direction_like_the_image():
    assert turn("up", "rotate_90") == "left"
    assert turn("up_right", "hflip") == "up_left"
    assert turn("right", "transpose") == "down"
