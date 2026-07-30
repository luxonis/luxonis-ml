from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from luxonis_ml.typing import (
    all_not_none,
    any_not_none,
    check_type,
)

# Falsy non-None values catch a truthiness check standing in for a
# proper ``is None`` check.
values = st.sampled_from([None, 0, "", False]) | st.text()


@given(values=st.lists(values, max_size=8))
def test_not_none_helpers(values: list[Any]):
    present = [value for value in values if value is not None]

    assert all_not_none(values) == (len(present) == len(values))
    assert any_not_none(values) == bool(present)


@given(values=st.lists(values, max_size=8))
def test_not_none_helpers_accept_iterators(values: list[Any]):
    assert all_not_none(iter(values)) == all_not_none(values)
    assert any_not_none(iter(values)) == any_not_none(values)


# ``bool`` and ``float`` are left out because they are interchangeable
# with ``int`` under the typing rules.
@given(value=st.integers() | st.text() | st.binary())
def test_check_type_scalars(value: Any):
    for typ in (int, str, bytes):
        assert check_type(value, typ) == isinstance(value, typ)


@given(numbers=st.lists(st.integers()), strings=st.lists(st.text()))
def test_check_type_containers(numbers: list[int], strings: list[str]):
    assert check_type(numbers, list)
    assert check_type(numbers, list[int])
    assert check_type(strings, list[str])
    assert check_type(
        dict(zip(strings, numbers, strict=False)), dict[str, int]
    )

    # An empty container matches any element type.
    assert check_type(numbers, list[str]) == (numbers == [])
