"""Small shared predicates for the vizlab package."""

from collections.abc import Sequence
from typing import TypeGuard


def is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Whether ``value`` is a sequence but not a ``str``/``bytes`` scalar."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
