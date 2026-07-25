"""Coverage for the ClassDistribution prediction overlay."""

import numpy as np
import pytest

from luxonis_ml.vizlab import ClassDistribution, Image, Palette

_DIST = {"cat": 0.6, "dog": 0.25, "fox": 0.1, "owl": 0.03, "bat": 0.02}


def _canvas(w: int = 200, h: int = 140) -> Image:
    return Image(np.full((h, w, 3), 30, np.uint8))


# --- input normalization & selection ----------------------------------------


def test_dict_and_list_inputs_rank_identically() -> None:
    as_list = [("cat", 0.6), ("dog", 0.25), ("fox", 0.1)]
    from_dict = ClassDistribution(
        probabilities={"cat": 0.6, "dog": 0.25, "fox": 0.1}
    )
    from_list = ClassDistribution(probabilities=as_list)
    assert from_dict._ranked() == from_list._ranked()


def test_ranked_is_sorted_descending() -> None:
    ranked = ClassDistribution(probabilities=_DIST)._ranked()
    probs = [p for _, p in ranked]
    assert probs == sorted(probs, reverse=True)
    assert ranked[0][0] == "cat"


def test_from_scores_pairs_names_and_scores() -> None:
    dist = ClassDistribution.from_scores(["a", "b", "c"], [0.1, 0.7, 0.2])
    assert dist._ranked() == [("b", 0.7), ("c", 0.2), ("a", 0.1)]


def test_from_scores_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shorter"):
        ClassDistribution.from_scores(["a", "b"], [0.5])


def test_top_k_caps_rows() -> None:
    selected = ClassDistribution(probabilities=_DIST, top_k=2)._selected()
    assert [name for name, _ in selected] == ["cat", "dog"]


def test_ground_truth_outside_top_k_is_appended() -> None:
    # "bat" is the least likely, well outside top_k=2, but must stay visible.
    selected = ClassDistribution(
        probabilities=_DIST, top_k=2, ground_truth="bat"
    )._selected()
    names = [name for name, _ in selected]
    assert names == ["cat", "dog", "bat"]


def test_top_k_none_shows_all() -> None:
    selected = ClassDistribution(probabilities=_DIST, top_k=None)._selected()
    assert len(selected) == len(_DIST)


# --- rendering --------------------------------------------------------------


@pytest.mark.parametrize("mode", ["bars", "chips", "gauge", "stacked"])
def test_each_mode_renders(mode: str) -> None:
    out = (
        _canvas()
        .add(
            ClassDistribution(
                probabilities=_DIST, mode=mode, ground_truth="dog"
            )
        )
        .render()
    )
    assert out.shape == (140, 200, 4)
    assert out[..., 3].max() > 0
    assert not np.array_equal(out[..., :3], _canvas().render()[..., :3])


def test_empty_distribution_is_noop() -> None:
    base = _canvas()
    plain = base.copy().render()
    empty = base.copy().add(ClassDistribution(probabilities={})).render()
    assert np.array_equal(plain, empty)


def test_empty_distribution_has_no_extent() -> None:
    assert ClassDistribution(probabilities={}).extent() is None


@pytest.mark.parametrize("mode", ["bars", "chips", "stacked"])
def test_ground_truth_marker_changes_pixels(mode: str) -> None:
    base = _canvas()
    without = (
        base.copy()
        .add(ClassDistribution(probabilities=_DIST, mode=mode))
        .render()
    )
    with_gt = (
        base.copy()
        .add(
            ClassDistribution(
                probabilities=_DIST, mode=mode, ground_truth="dog"
            )
        )
        .render()
    )
    # The highlight (outline / bold) is visible: the two renders differ.
    assert not np.array_equal(without, with_gt)


def test_gauge_verdict_differs_for_correct_vs_wrong() -> None:
    # Same distribution, but the ✓/✗ badge flips with the ground-truth class.
    base = _canvas()
    correct = (
        base.copy()
        .add(
            ClassDistribution(
                probabilities=_DIST, mode="gauge", ground_truth="cat"
            )
        )
        .render()
    )
    wrong = (
        base.copy()
        .add(
            ClassDistribution(
                probabilities=_DIST, mode="gauge", ground_truth="dog"
            )
        )
        .render()
    )
    assert not np.array_equal(correct, wrong)


def test_bar_uses_palette_color_for_class() -> None:
    # A class's bar is filled with its palette color (matching its box color).
    palette = Palette(["cat", "dog", "fox"])
    color = palette.color_for("cat")
    out = (
        _canvas()
        .add(
            ClassDistribution(
                probabilities={"cat": 1.0}, palette=palette, top_k=1
            )
        )
        .render()
    )
    match = (
        (out[..., 0] == color.r)
        & (out[..., 1] == color.g)
        & (out[..., 2] == color.b)
    )
    assert match.any()
