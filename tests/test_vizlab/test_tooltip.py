"""Tooltip construction and row resolution."""

import pytest

from luxonis_ml.vizlab import Tooltip


def test_rows_given_as_a_list_are_normalized() -> None:
    """A list of rows must behave exactly like a tuple of rows.

    Regression: `Tooltip` is a plain dataclass, so its ``rows`` annotation is
    not enforced and a list was stored as-is. Nothing complained until
    something drew the tooltip, at which point `resolved_rows` concatenated it
    with a tuple and raised ``TypeError`` — far from the call that caused it.
    """
    tooltip = Tooltip(title="car", rows=[("track", "7")])  # type: ignore[arg-type]

    assert tooltip.rows == (("track", "7"),)
    assert isinstance(tooltip.rows, tuple)
    assert tooltip.resolved_rows == (("track", "7"),)


def test_rows_from_a_list_survive_being_flattened_with_data() -> None:
    """The list case must also work on the path that actually broke."""
    tooltip = Tooltip(
        rows=[("track", "7")],  # type: ignore[arg-type]
        data={"speed": 41.2},
    )

    assert tooltip.resolved_rows == (("track", "7"), ("speed: ", "41.2"))


def test_row_entries_are_coerced_to_strings() -> None:
    """Numbers in rows are drawn, so they must arrive as text."""
    tooltip = Tooltip(rows=[("track", 7)])  # type: ignore[arg-type]

    assert tooltip.resolved_rows == (("track", "7"),)


def test_a_tooltip_stays_hashable() -> None:
    """Normalizing to tuples is what keeps a frozen dataclass hashable."""
    assert hash(Tooltip(title="car", rows=[("track", "7")]))  # type: ignore[arg-type]


@pytest.mark.parametrize("rows", [(), []])
def test_empty_rows_are_equivalent(rows: object) -> None:
    """An empty list and an empty tuple describe the same tooltip."""
    assert Tooltip(rows=rows).rows == ()  # type: ignore[arg-type]
