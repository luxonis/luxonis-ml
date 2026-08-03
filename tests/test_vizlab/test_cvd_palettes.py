"""Colorblind-safe palettes, and the CVD simulation that verifies them."""

import math

import pytest

from luxonis_ml.utils.color import BRAND_COLORS
from luxonis_ml.utils.color.cvd import (
    DEFICIENCIES,
    VISION,
    Deficiency,
    Lab,
    _ciede2000,
    delta_e,
    min_separation,
    simulate,
    to_lab,
)
from luxonis_ml.vizlab import (
    DARK_THEME,
    DEFAULT_PALETTE,
    PALETTES,
    ColormapColors,
    CVDDistinctColors,
    GoldenRatioColors,
    Gradient,
    Palette,
    RenderOptions,
)

#: The separation every shipped palette must keep between all of its colors,
#: under normal vision and all three dichromacies. CIEDE2000 calls ~1.0 a
#: just-noticeable difference between adjacent patches; annotation colors are
#: small, scattered, and sit on photographic backgrounds, so they need a much
#: wider margin. 8.0 is set just under the worst shipped set (tol-vibrant, whose
#: orange and red close to ~8.6 under tritanopia) — tight enough that swapping in
#: a worse color fails the test rather than silently degrading the palette.
MIN_SEPARATION = 8.0

# Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference Formula", table 1.
# The formula has enough hue wrap-arounds and near-neutral special cases that
# only published data can show the branches are all right.
SHARMA_CASES = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -0.9009, -85.5211), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),
    ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0009), 7.1792),
    ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0011), 7.2195),
    ((50.0, -0.0010, 2.4900), (50.0, 0.0009, -2.4900), 4.8045),
    ((50.0, 2.5000, 0.0), (50.0, 0.0, -2.5000), 4.3065),
    ((50.0, 2.5000, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5000, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((50.0, 2.5000, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((50.0, 2.5000, 0.0), (58.0, 24.0, 15.0), 19.4535),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize(("first", "second", "expected"), SHARMA_CASES)
def test_ciede2000_matches_published_data(
    first: Lab, second: Lab, expected: float
):
    assert _ciede2000(first, second) == pytest.approx(expected, abs=1e-4)


def test_ciede2000_is_symmetric_and_zero_on_itself():
    assert delta_e("#4477aa", "#4477aa") == 0.0
    assert delta_e("#4477aa", "#ee6677") == delta_e("#ee6677", "#4477aa")


def test_lab_anchors_the_scale_at_black_and_white():
    assert to_lab("#000000")[0] == pytest.approx(0.0)
    assert to_lab("#ffffff")[0] == pytest.approx(100.0, abs=1e-4)
    # Black to white is the full L* range, and CIEDE2000 leaves it unweighted.
    assert delta_e("#000000", "#ffffff") == pytest.approx(100.0, abs=1e-3)


@pytest.mark.parametrize("deficiency", DEFICIENCIES)
def test_simulation_preserves_grays(deficiency: Deficiency):
    # Every dichromacy keeps the achromatic axis intact; a model that moved
    # grays would be wrong in a way no palette check could catch.
    for gray in ("#000000", "#404040", "#808080", "#c0c0c0", "#ffffff"):
        assert delta_e(simulate(gray, deficiency), gray) < 1.0


def test_simulation_of_normal_vision_is_the_identity():
    assert simulate("#4477aa", None).rgb == (68, 119, 170)


def test_simulation_preserves_alpha():
    assert simulate((230, 159, 0, 128), "protanopia").a == 128


def test_simulation_rejects_an_unknown_deficiency():
    with pytest.raises(ValueError, match="unknown deficiency"):
        simulate("#4477aa", "achromatopsia")  # type: ignore[arg-type]


def test_simulation_collapses_a_known_confusion_pair():
    # The canonical red/green failure: obviously different, then nearly the
    # same color. This is the property the palettes are designed against.
    assert delta_e("#d62728", "#2ca02c") > 50
    assert (
        delta_e(
            simulate("#d62728", "deuteranopia"),
            simulate("#2ca02c", "deuteranopia"),
        )
        < 10
    )


def test_min_separation_has_nothing_to_measure_below_two_colors():
    assert min_separation([]) == math.inf
    assert min_separation(["#ff0000"]) == math.inf


def test_min_separation_defaults_to_the_worst_vision_type():
    pair = ["#d62728", "#2ca02c"]
    assert min_separation(pair) == min(
        min_separation(pair, [vision]) for vision in VISION
    )


@pytest.mark.parametrize("name", sorted(PALETTES))
@pytest.mark.parametrize("vision", VISION)
def test_shipped_palettes_stay_separable(name: str, vision: Deficiency | None):
    """Every shipped palette keeps its colors apart for every viewer.

    This is what makes "colorblind-safe" a property rather than a claim: the
    same check runs for normal vision and for each of the three dichromacies,
    and a palette only ships if it clears the bar in all four.
    """
    assert min_separation(PALETTES[name], [vision]) >= MIN_SEPARATION


@pytest.mark.parametrize("name", sorted(PALETTES))
def test_shipped_palettes_have_no_duplicates(name: str):
    colors = PALETTES[name]
    assert len(colors) >= 3
    assert len(set(colors)) == len(colors)


def test_default_palette_is_not_colorblind_safe():
    """Characterizes why this feature exists — and pins the claim to numbers.

    `GoldenRatioColors` spaces hues perfectly for normal vision and has no
    notion of how those hues collapse under a deficiency, so eight classes
    already contain a pair a deuteranope cannot separate. Changing the default
    is a product decision, not a test fix.
    """
    golden = [GoldenRatioColors()(i) for i in range(8)]
    assert min_separation(golden, [None]) > MIN_SEPARATION
    assert min_separation(golden) < 2.0
    assert isinstance(DEFAULT_PALETTE.at(0), type(golden[0]))


def test_named_palette_hands_out_its_colors_in_order():
    palette = Palette(generator="okabe-ito")
    anchors = PALETTES["okabe-ito"]
    assert [palette.at(i) for i in range(len(anchors))] == list(anchors)


def test_named_palette_is_reachable_from_palette_theme_and_options():
    """One name works everywhere a palette is chosen."""
    expected = PALETTES["tol-bright"][0]
    assert Palette(generator="tol-bright").color_for("car") == expected

    theme = DARK_THEME.with_palette("tol-bright")
    assert theme.palette.color_for("car") == expected

    options = RenderOptions(theme=theme)
    assert options.theme.palette.color_for("car") == expected
    # The default theme is untouched: `with_palette` returns a new theme.
    assert DARK_THEME.palette is not theme.palette


def test_unknown_palette_name_lists_the_known_ones():
    with pytest.raises(KeyError, match="okabe-ito"):
        Palette(generator="okabe_ito")


def test_named_palette_overflow_never_repeats_a_color():
    """The answer to "more classes than the palette has colors"."""
    palette = Palette(generator="okabe-ito")
    colors = [palette.at(i) for i in range(40)]
    assert len(set(colors)) == 40


def test_named_palette_overflow_beats_the_default_generator():
    """Overflow colors are chosen against CVD, not just against hue.

    The guarantee is comparative, not absolute: past the published colors the
    separation shrinks with every class, but stays far above what unbounded
    hue spacing manages.
    """
    okabe = [Palette(generator="okabe-ito").at(i) for i in range(20)]
    golden = [GoldenRatioColors()(i) for i in range(20)]
    assert min_separation(okabe) > 5.0
    assert min_separation(okabe) > 5 * min_separation(golden)


def test_cvd_generator_is_deterministic_across_instances():
    first = [CVDDistinctColors()(i) for i in range(12)]
    second = [CVDDistinctColors()(i) for i in range(12)]
    assert first == second
    # Jumping straight to a high index computes the same prefix.
    assert CVDDistinctColors()(11) == first[11]


def test_cvd_generator_steers_clear_of_the_colors_it_avoids():
    anchors = PALETTES["okabe-ito"]
    generated = [CVDDistinctColors(avoid=anchors)(i) for i in range(6)]
    assert min_separation([*anchors, *generated]) > 5.0
    # Without the seed it is free to land on top of one of them.
    naive = [CVDDistinctColors()(i) for i in range(6)]
    assert min_separation([*anchors, *naive]) < min_separation(
        [*anchors, *generated]
    )


def test_colormap_palette_reuses_the_gradient_registry():
    from_name = Palette.from_colormap("viridis", 5)
    from_object = Palette.from_colormap(
        Gradient.from_colors(["#440154", "#fde725"]), 5
    )
    assert from_name.at(0).rgb == (68, 1, 84)
    assert from_object.at(0).rgb == (68, 1, 84)
    with pytest.raises(KeyError, match="unknown gradient"):
        Palette.from_colormap("not-a-colormap")


def test_colormap_palette_with_a_count_spans_the_whole_colormap():
    palette = Palette.from_colormap("grayscale", 5)
    assert [palette.at(i).r for i in range(5)] == [0, 64, 128, 191, 255]
    # Past the count it wraps, which is the documented cost of pinning one.
    assert palette.at(5) == palette.at(0)


def test_colormap_palette_of_one_color_does_not_divide_by_zero():
    palette = Palette.from_colormap("grayscale", 1)
    assert palette.at(0).r == palette.at(3).r == 0


def test_lazy_colormap_palette_keeps_colors_stable_as_classes_arrive():
    """The reason lazy sampling uses bit-reversal instead of ``i / (n - 1)``.

    A class's color is fixed the first time it is seen, so a scheme that
    depended on the eventual class count would hand later images a different
    palette than earlier ones for the same data.
    """
    palette = Palette.from_colormap("viridis")
    first = palette.color_for("car")
    for name in ("bus", "truck", "bike", "person", "sign"):
        palette.color_for(name)
    assert palette.color_for("car") == first
    # And a fresh palette agrees, given the same first-seen order.
    assert Palette.from_colormap("viridis").color_for("car") == first


def test_lazy_colormap_sampling_spreads_every_prefix():
    """Any prefix covers the colormap; powers of two land on the even grid."""
    gradient = Gradient.from_colors(["#000000", "#ffffff"])
    lazy = ColormapColors(gradient)
    assert [lazy(i).r for i in range(5)] == [0, 255, 128, 64, 191]
    assert sorted(lazy(i).r for i in range(9)) == [
        0,
        32,
        64,
        96,
        128,
        159,
        191,
        223,
        255,
    ]


def test_lazy_colormap_palette_never_repeats_within_a_realistic_class_count():
    palette = Palette.from_colormap("viridis")
    assert len({palette.at(i) for i in range(64)}) == 64


def test_colormap_separation_degrades_with_the_class_count():
    """Substantiates the limitation `ColormapColors` documents.

    A colormap is a one-dimensional path, so classes crowd each other as they
    accumulate — there is no colormap for which this is not true, which is why
    the docs point past eight classes at the qualitative palettes.
    """
    separations = [
        min_separation(
            [Palette.from_colormap("viridis", n).at(i) for i in range(n)]
        )
        for n in (4, 8, 16)
    ]
    assert separations == sorted(separations, reverse=True)
    assert separations[-1] < MIN_SEPARATION


def test_a_named_palette_beats_a_colormap_at_the_same_size():
    """Why both exist: order costs separation, and here is how much."""
    size = len(PALETTES["okabe-ito"])
    sampled = [
        Palette.from_colormap("viridis", size).at(i) for i in range(size)
    ]
    assert min_separation(PALETTES["okabe-ito"]) > min_separation(sampled)


@pytest.mark.parametrize(
    ("vision", "floor"),
    [
        (None, 13.0),
        ("protanopia", 9.0),
        ("deuteranopia", 3.0),
        ("tritanopia", 5.0),
    ],
)
def test_brand_colors_report_their_separation(
    vision: Deficiency | None, floor: float
):
    """Documents how the Luxonis brand sequence fares — it is not CVD-safe.

    `BRAND_COLORS` is a brand decision, not a palette decision, and class labels
    do not use it by default. The floors here are the values measured today, so
    a change to the brand sequence surfaces as a failing test with real numbers
    rather than silently degrading an opt-in brand-anchored palette. Its worst
    pairing is mint/salmon at ~3.2 under deuteranopia, well under the
    :data:`MIN_SEPARATION` a shipped palette has to clear.
    """
    assert min_separation(BRAND_COLORS, [vision]) >= floor
