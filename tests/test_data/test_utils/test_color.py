import pytest

from luxonis_ml.data.utils.color import resolve_color


def test_resolve_color_int_grayscale():
    assert resolve_color(0) == (0, 0, 0)
    assert resolve_color(128) == (128, 128, 128)
    assert resolve_color(255) == (255, 255, 255)


def test_resolve_color_tuple_passthrough():
    assert resolve_color((1, 2, 3)) == (1, 2, 3)


def test_resolve_color_string_name():
    # matplotlib color names resolve to float RGB in [0, 1].
    assert resolve_color("black") == (0.0, 0.0, 0.0)
    assert resolve_color("white") == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("value", [300, -1])
def test_resolve_color_out_of_range(value: int):
    with pytest.raises(ValueError, match="out of range"):
        resolve_color(value)


def test_resolve_color_tuple_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        resolve_color((1, 2, 300))
