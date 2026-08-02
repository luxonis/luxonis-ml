"""Isolation fixtures shared by the whole parser test package."""

from collections.abc import Iterator

import pytest

from luxonis_ml.data.parsers.parser_plugin import PARSERS_REGISTRY
from tests.test_data.parsers.helpers import _SyntheticSplitParser


@pytest.fixture(autouse=True)
def _isolate_parsers_registry() -> Iterator[None]:
    """Restore the process-wide parser registry after every test.

    Tests register synthetic plugins into `PARSERS_REGISTRY`, and
    `Registry` exposes no removal API, so a leaked registration would
    have `get_parser_plugin` invoking the leaked plugin's `detect` in
    every later test - an outcome depending on collection order.
    """
    saved = dict(PARSERS_REGISTRY._module_dict)
    yield
    PARSERS_REGISTRY._module_dict.clear()
    PARSERS_REGISTRY._module_dict.update(saved)


@pytest.fixture(autouse=True)
def _reset_synthetic_split_parser() -> Iterator[None]:
    """Restore `_SyntheticSplitParser`'s detection state after every test.

    `recognized` and `splits` are class attributes on a class the whole
    package shares, so a test that sets either and does not restore it
    would change what `detect` returns for every test after it. The
    dictionary is restored as a fresh copy so mutated contents cannot
    leak either.
    """
    previous = (
        _SyntheticSplitParser.recognized,
        dict(_SyntheticSplitParser.splits),
    )
    yield
    _SyntheticSplitParser.recognized, _SyntheticSplitParser.splits = previous
