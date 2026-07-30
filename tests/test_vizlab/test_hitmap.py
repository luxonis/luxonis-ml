"""Tests for the `HitMap` / `Tooltip` core types."""

from luxonis_ml.vizlab import HitMap, Tooltip
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.interaction.maps import ClickMap, InteractionCapture


def _entry(
    left: float,
    top: float,
    right: float,
    bottom: float,
    title: str,
) -> tuple[Rect, Tooltip]:
    return (Rect(left, top, right, bottom), Tooltip(title=title))


def _title_at(hm: HitMap, x: float, y: float) -> str | None:
    tooltip = hm.hit(x, y)
    return tooltip.title if tooltip is not None else None


def test_tooltip_is_empty() -> None:
    assert Tooltip().is_empty
    assert Tooltip(title="").is_empty
    assert not Tooltip(title="car").is_empty
    assert not Tooltip(rows=(("id", "7"),)).is_empty


def test_hit_picks_smallest_containing_box() -> None:
    hm = HitMap(
        [
            _entry(0.0, 0.0, 100.0, 100.0, "big"),
            _entry(10.0, 10.0, 40.0, 40.0, "small"),
        ]
    )
    assert _title_at(hm, 25, 25) == "small"  # inside both -> smaller wins
    assert _title_at(hm, 80, 80) == "big"  # inside only the big box
    assert hm.hit(200, 200) is None  # outside all


def test_hit_empty_map() -> None:
    assert HitMap.empty().hit(0, 0) is None


def test_offset_shifts_every_rect() -> None:
    hm = HitMap([_entry(0.0, 0.0, 10.0, 10.0, "a")]).offset(5.0, 7.0)
    rect, tooltip = hm.items[0]
    corners = (rect.left, rect.top, rect.right, rect.bottom)
    assert corners == (5.0, 7.0, 15.0, 17.0)
    assert tooltip.title == "a"
    # A point that only lands inside the shifted box hits it.
    assert _title_at(hm, 12, 12) == "a"


def test_scaled_scales_every_rect() -> None:
    hm = HitMap([_entry(2.0, 4.0, 6.0, 8.0, "a")]).scaled(2.0)
    rect, _ = hm.items[0]
    assert (rect.left, rect.top, rect.right, rect.bottom) == (
        4.0,
        8.0,
        12.0,
        16.0,
    )


def test_merge_and_or_concatenate() -> None:
    a = HitMap([_entry(0.0, 0.0, 1.0, 1.0, "a")])
    b = HitMap([_entry(2.0, 2.0, 3.0, 3.0, "b")])
    assert [t.title for _, t in a.merge(b).items] == ["a", "b"]
    assert [t.title for _, t in (a | b).items] == ["a", "b"]


def test_clickmap_or_concatenates() -> None:
    a = ClickMap([(Rect(0.0, 0.0, 1.0, 1.0), "a")])
    b = ClickMap([(Rect(2.0, 2.0, 3.0, 3.0), "b")])
    assert [action for _, action in (a | b).items] == ["a", "b"]


def test_interaction_capture_adds_clickmap_with_transform() -> None:
    capture = InteractionCapture().transformed(5.0, 7.0, 2.0, 3.0)
    capture.add_clickmap(ClickMap([(Rect(1.0, 2.0, 3.0, 4.0), "class:car")]))

    rect, action = capture.clicks[0]
    assert rect == Rect(7.0, 13.0, 11.0, 19.0)
    assert action == "class:car"
