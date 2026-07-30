import uuid

from hypothesis import assume, given
from hypothesis import strategies as st

from luxonis_ml.data.utils import merge_uuids

uuid_strings = st.uuids().map(str)
uuid_lists = st.lists(uuid_strings, min_size=1, max_size=8)


@st.composite
def uuids_and_permutation(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    uuids = draw(uuid_lists)
    return uuids, draw(st.permutations(uuids))


@given(uuids_and_permutation())
def test_merging_does_not_depend_on_order(
    uuids_and_shuffled: tuple[list[str], list[str]],
):
    uuids, shuffled = uuids_and_shuffled

    assert merge_uuids(uuids) == merge_uuids(shuffled)


@given(uuids=uuid_lists)
def test_merging_is_deterministic(uuids: list[str]):
    merged = merge_uuids(uuids)

    assert isinstance(merged, uuid.UUID)
    assert merged == merge_uuids(uuids)
    assert merged == merge_uuids(iter(uuids))


@given(first=uuid_lists, second=uuid_lists)
def test_different_uuids_merge_differently(
    first: list[str], second: list[str]
):
    assume(sorted(first) != sorted(second))

    assert merge_uuids(first) != merge_uuids(second)
