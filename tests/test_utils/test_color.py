"""The shared Color primitive parses every color-like input shape."""

import pytest

from luxonis_ml.utils.color import Color


def test_parse_hex_with_and_without_hash():
    assert Color.parse("#4c8dff") == Color(76, 141, 255)
    assert Color.parse("4c8dff") == Color(76, 141, 255)
    assert Color.parse("#abc") == Color(170, 187, 204)


def test_parse_named_basics_without_pillow():
    # The common names resolve from the built-in table (no Pillow needed).
    assert Color.parse("black").rgb == (0, 0, 0)
    assert Color.parse("white").rgb == (255, 255, 255)
    assert Color.parse("Gray").rgb == (128, 128, 128)
    assert Color.parse("grey").rgb == (128, 128, 128)


def test_parse_named_extended_via_pillow():
    pytest.importorskip("PIL")
    assert Color.parse("royalblue").rgb == (65, 105, 225)


def test_parse_grayscale_int():
    assert Color.parse(0).rgb == (0, 0, 0)
    assert Color.parse(128).rgb == (128, 128, 128)
    assert Color.parse(255).rgb == (255, 255, 255)


def test_parse_tuples():
    assert Color.parse((1, 2, 3)) == Color(1, 2, 3, 255)
    assert Color.parse((1, 2, 3, 4)) == Color(1, 2, 3, 4)


def test_out_of_range_channels_are_clamped():
    # Color clamps rather than raising (its consistent policy).
    assert Color.parse(300).rgb == (255, 255, 255)
    assert Color.parse(-5).rgb == (0, 0, 0)
    assert Color.parse((1, 2, 300)).rgb == (1, 2, 255)


def test_rgb_and_rgba_properties():
    assert Color(1, 2, 3, 4).rgb == (1, 2, 3)
    assert Color(1, 2, 3, 4).rgba == (1, 2, 3, 4)


def test_parse_color_passthrough():
    color = Color(10, 20, 30)
    assert Color.parse(color) is color


def test_invalid_hex_reports_as_hex():
    with pytest.raises(ValueError, match="invalid hex color"):
        Color.parse("#12")


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="invalid color name"):
        Color.parse("not_a_real_color")


def test_bool_is_rejected():
    with pytest.raises(ValueError, match="cannot parse color"):
        Color.parse(True)
