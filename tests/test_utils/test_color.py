"""The shared Color primitive parses every color-like input shape."""

import importlib.util
import sys
import time
from threading import Thread

import numpy as np
import pytest

from luxonis_ml.utils.color import Color, Palette
from luxonis_ml.utils.color import gradient as gradient_module
from luxonis_ml.utils.color.gradient import (
    GRADIENTS,
    Gradient,
    resolve_gradient,
)


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


def test_chrome_for_picks_by_background_lightness():
    """A light background resolves the white/purple chrome; a dark one the navy."""
    from luxonis_ml.utils.color import brand

    light = brand.chrome_for(brand.LIGHT_BACKGROUND)
    dark = brand.chrome_for(brand.BACKGROUND)
    assert light is brand.LIGHT_CHROME
    assert dark is brand.DARK_CHROME
    # Light chrome: white card, brand-purple body text, deeper purple titles,
    # and a hairline border.
    assert light.card_bg.r > 240
    assert light.card_bg.g > 240
    assert light.card_text is brand.PURPLE
    assert light.card_title is brand.PURPLE_TITLE
    assert light.border is not None
    # Dark chrome: navy card, light-purple (periwinkle) text — on-brand, not
    # plain white — and no border (it relies on its shadow instead).
    assert dark.card_text is brand.PERIWINKLE
    assert dark.card_text.b > dark.card_text.r  # blue-purple, not white
    assert dark.border is None


def test_light_title_is_deeper_than_body_purple():
    """The light-mode heading outranks the regular-purple body text."""
    from luxonis_ml.utils.color import brand

    # Same blue-purple family, but the title is clearly darker than the body.
    assert brand.PURPLE_TITLE.b > brand.PURPLE_TITLE.r
    assert sum(brand.PURPLE_TITLE.rgb) < sum(brand.PURPLE.rgb)


def test_dark_title_is_lighter_than_body_on_dark():
    """The dark-mode heading is a brighter lavender than the periwinkle body."""
    from luxonis_ml.utils.color import brand

    dark = brand.chrome_for(brand.BACKGROUND)
    # Same purple family, but the title is clearly lighter (stronger on dark).
    assert dark.card_title.b > dark.card_title.r
    assert sum(dark.card_title.rgb) > sum(dark.card_text.rgb)


def test_parse_rejects_a_tuple_that_is_not_rgb_or_rgba():
    with pytest.raises(ValueError, match="3 or 4 elements"):
        Color.parse((1, 2))  # type: ignore[arg-type]


def test_saturate_moves_toward_and_away_from_gray():
    muted = Color(140, 120, 120)
    assert muted.saturate(1.0).hls[2] == pytest.approx(1.0)
    assert muted.saturate(-1.0).hls[2] == pytest.approx(0.0)
    assert muted.saturate(0.5).a == muted.a  # alpha is preserved


def test_shift_hue_is_a_full_circle_at_one_turn():
    color = Color(200, 60, 40)
    assert color.shift_hue(1.0).rgb == color.rgb
    # Half a turn lands on the opposing hue, so the channels really move.
    assert color.shift_hue(0.5).rgb != color.rgb


def test_palette_at_does_not_register_the_class():
    palette = Palette()
    first = palette.at(0)
    assert len(palette) == 0  # peeking must not consume a slot
    assert palette.color_for("car") == first
    assert len(palette) == 1


def test_pinned_class_keeps_its_color_and_consumes_no_slot():
    palette = Palette().pin("car", "#ff0000")
    assert palette.color_for("car") == Color(255, 0, 0)
    # The pin sits outside the sequence, so the next class still gets slot 0.
    assert palette.color_for("person") == palette.at(0)


def test_with_colors_returns_a_specialized_copy():
    shared = Palette(classes=["car"])
    car = shared.color_for("car")
    special = shared.with_colors({"person": "#00ff00"})

    assert special.color_for("person") == Color(0, 255, 0)
    assert special.color_for("car") == car  # inherits what was assigned
    # The original is untouched, so a shared palette survives specialization.
    assert shared.color_for("person") != Color(0, 255, 0)


def test_two_threads_registering_at_once_get_different_colors():
    """Concurrent first uses must not land on the same generator slot."""

    def slow_generator(index: int) -> Color:
        # Widen the read-then-write window so the race is hit every run.
        time.sleep(0.05)
        return Color(index, index, index)

    palette = Palette(generator=slow_generator)
    assigned: dict[str, Color] = {}

    def register(key: str) -> None:
        assigned[key] = palette.color_for(key)

    threads = [Thread(target=register, args=(key,)) for key in ("car", "bus")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert assigned["car"] != assigned["bus"]
    assert {c.r for c in assigned.values()} == {0, 1}


def test_gradient_needs_at_least_two_colors():
    with pytest.raises(ValueError, match="at least two colors"):
        Gradient.from_colors(["red"])


def test_gradient_positions_must_match_the_colors():
    with pytest.raises(ValueError, match="one entry per color"):
        Gradient.from_colors(["black", "white"], positions=[0.0, 0.5, 1.0])


def test_gradient_honors_explicit_stop_positions():
    # White arrives at 0.25, so the midpoint is already past it.
    gradient = Gradient.from_colors(["black", "white"], positions=[0.0, 0.25])
    assert gradient.color_at(0.25) == Color(255, 255, 255)
    assert gradient.color_at(0.5) == Color(255, 255, 255)
    assert gradient.color_at(0.125).rgb == (128, 128, 128)


def test_colorize_maps_a_field_to_rgb_and_clamps():
    field = np.array([[0.0, 1.0], [-5.0, 5.0]])
    rgb = Gradient.from_colors(["black", "white"]).colorize(field)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    # Out-of-range scalars clamp to the end stops rather than wrapping.
    assert rgb[0, 0].tolist() == [0, 0, 0]
    assert rgb[1, 0].tolist() == [0, 0, 0]
    assert rgb[1, 1].tolist() == [255, 255, 255]


def test_gradient_rejects_positions_outside_the_unit_range():
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        Gradient.from_colors(["black", "white"], positions=[0.0, 1.5])


def test_gradient_rejects_repeated_positions():
    # Two stops at the same spot make the color between them order-dependent.
    with pytest.raises(ValueError, match="must be distinct"):
        Gradient.from_colors(
            ["black", "white", "red"], positions=[0.0, 0.5, 0.5]
        )


def test_color_at_interpolates_alpha_with_the_color_channels():
    gradient = Gradient.from_colors([Color(0, 0, 0, 0), Color(0, 0, 0, 200)])
    assert gradient.color_at(0.0).a == 0
    assert gradient.color_at(0.5).a == 100
    assert gradient.color_at(1.0).a == 200


def test_colorize_accepts_any_array_like():
    # The field is converted with np.asarray, so nested lists work too.
    rgb = Gradient.from_colors(["black", "white"]).colorize([[0.0, 1.0]])

    assert rgb.shape == (1, 2, 3)
    assert rgb[0, 0].tolist() == [0, 0, 0]
    assert rgb[0, 1].tolist() == [255, 255, 255]


def test_gradient_module_imports_without_numpy(
    monkeypatch: pytest.MonkeyPatch,
):
    """NumPy is not a base dependency; this module must import without it."""
    # A None entry makes `import numpy` fail.
    monkeypatch.setitem(sys.modules, "numpy", None)

    # Load a second copy from source, so the real module stays untouched. The
    # package name keeps its relative imports resolvable.
    spec = importlib.util.spec_from_file_location(
        "luxonis_ml.utils.color._gradient_probe", gradient_module.__file__
    )
    assert spec is not None
    assert spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    assert probe.Gradient.from_colors(["black", "white"]).color_at(
        0.5
    ).rgb == (
        128,
        128,
        128,
    )


def test_resolve_gradient_accepts_a_name_or_an_instance():
    assert resolve_gradient("grayscale") is GRADIENTS["grayscale"]
    custom = Gradient.from_colors(["black", "white"])
    assert resolve_gradient(custom) is custom


def test_resolve_gradient_names_the_alternatives_when_unknown():
    with pytest.raises(KeyError, match="unknown gradient") as excinfo:
        resolve_gradient("not-a-gradient")
    assert "grayscale" in str(excinfo.value)
