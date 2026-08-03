"""Coverage for `ScaleBar` and `Ruler`: calibration, rounding, and drawing."""

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    DARK_THEME,
    LIGHT_THEME,
    BBox,
    Corner,
    Image,
    RenderOptions,
    Ruler,
    ScaleBar,
    Style,
    Tooltip,
    grid,
)
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.annotations.layout import LabelLayout
from luxonis_ml.vizlab.annotations.scalebar import format_units, nice_length
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.render.context import layer_scope
from luxonis_ml.vizlab.scene.html import draws_anything


def _canvas(w: int = 400, h: int = 240) -> Image:
    return Image(np.full((h, w, 3), 60, np.uint8))


def _drawn(image: Image, size: tuple[int, int] | None = None) -> int:
    base = Image(image.base_rgba()).render(size)
    return int(
        (np.abs(image.render(size).astype(int) - base).sum(-1) > 8).sum()
    )


# --- the round-number choice ------------------------------------------------


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (1.0, 1.0),
        (1.9, 1.0),
        (2.0, 2.0),
        (4.99, 2.0),
        (5.0, 5.0),
        (9.99, 5.0),
        (37.0, 20.0),
        (0.9, 0.5),
        (0.04, 0.02),
        (250_000.0, 200_000.0),
    ],
)
def test_nice_length_is_one_two_or_five_times_a_power_of_ten(
    limit: float, expected: float
) -> None:
    chosen = nice_length(limit)
    assert chosen == pytest.approx(expected)
    assert chosen <= limit


@pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), float("nan")])
def test_nice_length_rejects_a_limit_it_cannot_round(limit: float) -> None:
    with pytest.raises(ValueError, match="positive limit"):
        nice_length(limit)


def test_format_units_drops_noise_digits() -> None:
    assert format_units(50.0) == "50"
    assert format_units(12.44) == "12.4"
    assert format_units(0.05) == "0.05"
    assert format_units(0.0) == "0"


# --- ScaleBar ---------------------------------------------------------------


def test_the_bar_is_the_largest_round_length_that_fits() -> None:
    bar = ScaleBar(pixels_per_unit=8.0, unit="m")
    # A quarter of 400px is 100px, i.e. 12.5m; 10m is the round length under it.
    assert bar._choose(400) == (10.0, 80.0)
    # A wider allowance reaches the next round step up.
    assert ScaleBar(pixels_per_unit=8.0, max_fraction=0.5)._choose(400) == (
        20.0,
        160.0,
    )


def test_the_calibration_follows_the_render_size() -> None:
    calibrated = ScaleBar(pixels_per_unit=8.0, reference_width=400)
    assert calibrated._choose(400) == (10.0, 80.0)
    # Half the render size: the same 10m is half as many display pixels, so the
    # bar shrinks with the picture instead of overstating the distance.
    assert calibrated._choose(200) == (10.0, 40.0)
    # Without a reference the calibration is read against the canvas drawn.
    assert ScaleBar(pixels_per_unit=8.0)._choose(200) == (5.0, 40.0)


@pytest.mark.parametrize(
    "pixels_per_unit", [0.0, -4.0, float("inf"), float("nan")]
)
def test_an_unusable_calibration_draws_nothing(pixels_per_unit: float) -> None:
    bar = ScaleBar(pixels_per_unit=pixels_per_unit)
    assert bar._choose(400) is None
    assert _drawn(_canvas().add(bar)) == 0


def test_the_bar_draws_and_carries_its_unit() -> None:
    assert _drawn(_canvas().add(ScaleBar(pixels_per_unit=8.0, unit="m"))) > 0
    bar = ScaleBar(pixels_per_unit=8.0, unit="m")
    assert bar._caption(10.0) == "10 m"
    assert ScaleBar()._caption(50.0) == "50 px"  # uncalibrated: a pixel ruler
    assert ScaleBar(unit="")._caption(50.0) == "50"


def test_the_unit_accepts_inline_markup() -> None:
    def render(unit: str) -> np.ndarray:
        return _canvas().add(ScaleBar(pixels_per_unit=8.0, unit=unit)).render()

    plain = render("um")
    assert not np.array_equal(
        plain, render("<i>u</i>m")
    )  # the tag took effect

    def spread(image: np.ndarray) -> int:
        return int((np.abs(image.astype(int) - plain).sum(-1) > 8).sum())

    # A recognized tag is consumed; an unrecognized one is drawn verbatim.
    assert spread(render("<zz>u</zz>m")) > spread(render("<i>u</i>m"))


def test_the_bar_is_reserved_so_label_chips_avoid_it() -> None:
    layout = LabelLayout(400, 240)
    ctx = RenderContext(canvas=Canvas.blank(400, 240), layout=layout)
    ScaleBar(pixels_per_unit=8.0).reserve(ctx)
    assert layout.placed  # the corner is taken before any chip is placed
    reserved = layout.placed[0]
    assert reserved.right <= 400
    assert reserved.bottom <= 240


@pytest.mark.parametrize(
    "corner",
    [Corner.TOP_LEFT, Corner.TOP_RIGHT, Corner.BOTTOM_LEFT],
)
def test_the_corner_moves_the_bar(corner: Corner) -> None:
    default = _canvas().add(ScaleBar(pixels_per_unit=8.0)).render()
    moved = (
        _canvas().add(ScaleBar(pixels_per_unit=8.0, corner=corner)).render()
    )
    assert not np.array_equal(default, moved)


def test_the_card_follows_the_theme() -> None:
    def render(theme: object) -> np.ndarray:
        image = Image(
            np.full((240, 400, 3), 128, np.uint8),
            options=RenderOptions(theme=theme),  # type: ignore[arg-type]
        )
        return image.add(ScaleBar(pixels_per_unit=8.0)).render()

    assert not np.array_equal(render(DARK_THEME), render(LIGHT_THEME))


# --- Ruler ------------------------------------------------------------------


def test_the_ruler_measures_in_the_image_plane() -> None:
    span = Ruler(
        start=(0.1, 0.5), end=(0.6, 0.5), pixels_per_unit=8.0, unit="m"
    )
    assert span.measure(160, 100) == "10 m"
    # A diagonal is one distance, not two: 30px by 40px is 50px.
    assert Ruler(start=(0.0, 0.0), end=(0.3, 0.4)).measure(100, 100) == "50 px"
    # Rendering larger does not change what the ruler says it measured.
    calibrated = Ruler(
        start=(0.0, 0.0),
        end=(0.5, 0.0),
        pixels_per_unit=8.0,
        reference_width=160,
    )
    assert calibrated.measure(160, 100) == calibrated.measure(320, 200)


def test_the_ruler_draws_its_span_and_ticks() -> None:
    span = Ruler(start=(0.1, 0.3), end=(0.8, 0.7))
    ticked = _drawn(_canvas().add(span))
    plain = _drawn(_canvas().add(span.model_copy(update={"ticks": False})))
    assert ticked > plain > 0
    # A zero-length ruler has no direction to lay a tick across; it still
    # measures, and does not raise.
    dot = Ruler(start=(0.5, 0.5), end=(0.5, 0.5))
    assert dot.measure(400, 240) == "0 px"
    assert _drawn(_canvas().add(dot)) >= 0


def test_the_measurement_rides_in_as_the_chip_payload() -> None:
    ctx = RenderContext(
        canvas=Canvas.blank(400, 240), layout=LabelLayout(400, 240)
    )
    chips: list[str] = []

    def record(
        _ctx: object,
        _region: object,
        label: str | None,
        _score: object,
        payload: object,
        *_rest: object,
    ) -> None:
        chips.append(f"{label}|{payload}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "luxonis_ml.vizlab.annotations.scalebar.place_label", record
        )
        white = Color(255, 255, 255)
        Ruler(start=(0.0, 0.5), end=(0.5, 0.5)).draw_label(ctx, Style(), white)
        Ruler(
            start=(0.0, 0.5), end=(0.5, 0.5), label="gap", payload="fixed"
        ).draw_label(ctx, Style(), white)
    assert chips[0] == "None|200 px"  # the measurement fills the payload
    assert chips[1] == "gap|fixed"  # an explicit payload is left alone


def test_the_measurement_can_be_switched_off() -> None:
    span = Ruler(start=(0.1, 0.5), end=(0.6, 0.5))
    quiet = span.model_copy(update={"measurement": False})
    assert _drawn(_canvas().add(span)) > _drawn(_canvas().add(quiet)) > 0


def test_the_ruler_reports_a_hover_region() -> None:
    span = Ruler(
        start=(0.2, 0.3), end=(0.8, 0.7), tooltip=Tooltip(title="span")
    )
    _, hits = _canvas().add(span).render_hits()
    assert len(hits.items) == 1
    assert hits.hit(200.0, 120.0) is not None  # the middle of the run
    assert hits.hit(5.0, 5.0) is None


# --- render targets and composition -----------------------------------------


def test_measurements_are_vector_in_svg_and_survive_composition() -> None:
    image = _canvas().add(BBox(x=0.1, y=0.1, w=0.3, h=0.3, label="car"))
    image.add(ScaleBar(pixels_per_unit=8.0, unit="m"))
    image.add(Ruler(start=(0.2, 0.8), end=(0.8, 0.8), pixels_per_unit=8.0))
    with layer_scope({"overlay"}, chrome=False, labels=False):
        marks = image.render_svg()
    assert draws_anything(marks)
    assert b"<image" not in marks  # vector chrome, not a rasterized card
    stacked = grid([image, image.copy()], ncols=2)
    assert stacked.render().shape[1] > image.render().shape[1]
    assert len(stacked.render_svg()) > 0
