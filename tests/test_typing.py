from typing import List

from luxonis_ml.typing import all_not_none, any_not_none, check_type


def test_all_not_none():
    assert all_not_none([1, 2, 3])
    assert not all_not_none([1, 2, None])
    assert not all_not_none([None, None, None])
    assert all_not_none([])


def test_any_not_none():
    assert any_not_none([1, 2, 3])
    assert any_not_none([1, 2, None])
    assert not any_not_none([None, None, None])
    assert not any_not_none([])


def test_check_type():
    assert check_type(1, int)
    assert not check_type(1, str)
    assert check_type([1, 2, 3], List[int])
    assert not check_type([1, 2, 3], List[str])
