from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from luxonis_ml.typing import (
    all_not_none,
    any_not_none,
    check_type,
)

# Includes values that are falsy but not ``None``, which is what the
# ``*_not_none`` helpers must not confuse with a missing value.
values = st.none() | st.just(0) | st.just("") | st.just(False) | st.text()

# Mutually exclusive types. ``bool`` and ``float`` are left out because
# they are interchangeable with ``int`` under the typing rules.
SCALARS = [(st.integers(), int), (st.text(), str), (st.binary(), bytes)]


@st.composite
def typed_values(draw: st.DrawFn) -> tuple[Any, type]:
    strategy, expected_type = draw(st.sampled_from(SCALARS))
    return draw(strategy), expected_type


@given(values=st.lists(values, max_size=8))
def test_none_helpers_only_look_at_none(values: list[Any]):
    present = [value for value in values if value is not None]

    assert all_not_none(values) == (len(present) == len(values))
    assert any_not_none(values) == bool(present)


@given(values=st.lists(values, max_size=8))
def test_none_helpers_accept_any_iterable(values: list[Any]):
    assert all_not_none(iter(values)) == all_not_none(values)
    assert any_not_none(iter(values)) == any_not_none(values)


@given(value_and_type=typed_values())
def test_check_type_accepts_only_the_matching_type(
    value_and_type: tuple[Any, type],
):
    value, expected_type = value_and_type

    assert check_type(value, expected_type)
    for _, other_type in SCALARS:
        if other_type is not expected_type:
            assert not check_type(value, other_type)


@given(numbers=st.lists(st.integers()), strings=st.lists(st.text()))
def test_check_type_looks_inside_containers(
    numbers: list[int], strings: list[str]
):
    assert check_type(numbers, list)
    assert check_type(numbers, list[int])
    assert check_type(strings, list[str])
    assert check_type(
        dict(zip(strings, numbers, strict=False)), dict[str, int]
    )

    # An empty container matches any element type.
    assert check_type(numbers, list[str]) == (numbers == [])
