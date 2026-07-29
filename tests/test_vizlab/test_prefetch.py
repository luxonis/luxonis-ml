"""Ordered background-prefetch coverage."""

from collections.abc import Iterator
from contextvars import ContextVar
from threading import Event, Thread, get_ident

import pytest

from luxonis_ml.vizlab.viewer import PrefetchIterator


def test_prefetch_loads_in_background_and_preserves_order() -> None:
    producer_threads: list[int] = []
    started = Event()

    def source() -> Iterator[int]:
        for value in range(4):
            producer_threads.append(get_ident())
            started.set()
            yield value

    consumer_thread = get_ident()
    with PrefetchIterator(source(), capacity=2) as items:
        assert started.wait(timeout=1.0)
        assert list(items) == [0, 1, 2, 3]

    assert producer_threads
    assert set(producer_threads) != {consumer_thread}


def test_prefetch_reraises_source_exceptions_in_consumer() -> None:
    def source() -> Iterator[int]:
        yield 1
        raise RuntimeError("loader failed")

    with PrefetchIterator(source(), capacity=1) as items:
        assert next(items) == 1
        with pytest.raises(RuntimeError, match="loader failed"):
            next(items)


def test_prefetch_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        PrefetchIterator([1], capacity=0)


def test_prefetch_inherits_the_callers_context() -> None:
    value = ContextVar("prefetch_test_value", default="default")
    token = value.set("caller")

    def source() -> Iterator[str]:
        yield value.get()

    try:
        with PrefetchIterator(source(), capacity=1) as items:
            assert next(items) == "caller"
    finally:
        value.reset(token)


def test_close_waits_for_in_flight_source_work() -> None:
    source_started = Event()
    release_source = Event()
    source_finished = Event()
    close_started = Event()
    close_finished = Event()

    def source() -> Iterator[int]:
        source_started.set()
        assert release_source.wait(timeout=1.0)
        source_finished.set()
        yield 1

    items = PrefetchIterator(source(), capacity=1)
    assert source_started.wait(timeout=1.0)

    def close_items() -> None:
        close_started.set()
        items.close()
        close_finished.set()

    closer = Thread(target=close_items)
    closer.start()
    assert close_started.wait(timeout=1.0)
    try:
        assert not close_finished.wait(timeout=0.25)
    finally:
        release_source.set()
        closer.join(timeout=1.0)

    assert source_finished.is_set()
    assert close_finished.is_set()
