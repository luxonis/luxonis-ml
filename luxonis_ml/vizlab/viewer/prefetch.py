"""Bounded, ordered background prefetching for interactive data sources."""

from collections.abc import Iterable, Iterator
from contextvars import Context, copy_context
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from types import TracebackType
from typing import Generic, TypeVar

from typing_extensions import Self

ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class _Value(Generic[ItemT]):
    """One successfully loaded value."""

    value: ItemT


@dataclass(frozen=True, slots=True)
class _Failure:
    """An exception raised while advancing the source iterator."""

    error: BaseException


@dataclass(frozen=True, slots=True)
class _End:
    """Source-exhaustion marker."""


_END = _End()


class PrefetchIterator(Iterator[ItemT]):
    """Advance an iterable on one daemon thread into a bounded FIFO queue.

    Values are yielded in source order. Exceptions raised by the source are
    re-raised in the consuming thread at the corresponding position. Closing
    sets a cooperative stop flag, discards queued values, and waits for the
    current source advance to finish. This prevents native work such as image
    rendering from outliving its viewer. The worker inherits the caller's context
    variables, so scoped render options and styles remain in effect when
    rendering is prefetched.

    Args:
        source: Iterable whose items should be loaded ahead.
        capacity: Maximum number of completed items retained in memory.

    """

    def __init__(self, source: Iterable[ItemT], *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Prefetch capacity must be at least 1.")
        self._source = iter(source)
        self._queue: Queue[_Value[ItemT] | _Failure | _End] = Queue(capacity)
        self._stop = Event()
        self._closed = False
        self._context: Context = copy_context()
        self._thread = Thread(
            target=self._run,
            name="vizlab-prefetch",
            daemon=True,
        )
        self._thread.start()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> ItemT:
        if self._closed:
            raise StopIteration
        item = self._queue.get()
        if isinstance(item, _Value):
            return item.value
        self.close()
        if isinstance(item, _Failure):
            raise item.error
        raise StopIteration

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop accepting work and wait for the producer to finish safely."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._discard_queued()
        self._thread.join()
        self._discard_queued()

    def _discard_queued(self) -> None:
        """Remove queued events, including any enqueued during shutdown."""
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _run(self) -> None:
        """Run the producer inside the context captured by the caller."""
        self._context.run(self._produce)

    def _produce(self) -> None:
        """Advance the source and enqueue values until exhausted or stopped."""
        try:
            for value in self._source:
                if not self._put(_Value(value)):
                    return
        except BaseException as error:
            self._put(_Failure(error))
            return
        self._put(_END)

    def _put(self, item: _Value[ItemT] | _Failure | _End) -> bool:
        """Cooperatively enqueue one event, returning false after closure."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
            except Full:
                continue
            return not self._stop.is_set()
        return False
