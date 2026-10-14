from collections.abc import Iterator
from types import TracebackType
from typing import cast

import numpy as np
import pytest
from typing_extensions import Self

import luxonis_ml.vizlab.viewer as viewer_module
from luxonis_ml.vizlab import HitMap
from vizlab_examples.demo import show
from vizlab_examples.demo.slides import GLOSSARY, SLIDES, Slide


def test_first_slide_introduces_glossary_tooltips() -> None:
    first = SLIDES[0]
    column_w = show.build([first]).column_w
    _, hits = show.page.compose_slide(
        first.title,
        [first.body],
        "",
        None,
        column_w=column_w,
        glossary=GLOSSARY,
    )
    tooltip = GLOSSARY["highlighted text"]
    assert sum(tip is tooltip for _, tip in hits.items) == 1


def test_build_defers_slide_evaluation_and_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    frame = (np.zeros((1, 1, 4), dtype=np.uint8), HitMap.empty())

    def evaluate(source: str, namespace: dict) -> None:
        calls.append(f"evaluate {source}")

    def compose(*args: object, **kwargs: object) -> tuple[np.ndarray, HitMap]:
        calls.append("compose")
        return frame

    monkeypatch.setattr(show, "SETUP", "")
    monkeypatch.setattr(show, "_evaluate", evaluate)
    monkeypatch.setattr(show.page, "compose_slide", compose)

    deck = show.build([Slide("Lazy", "Not rendered yet.", "example")])

    assert calls == []
    assert list(deck) == [frame]
    assert calls == ["evaluate example", "compose"]


def test_present_prefetches_and_reuses_visited_slides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    produced: list[int] = []
    shown: list[int] = []
    capacities: list[int] = []
    closed: list[bool] = []
    keys = iter(["right", "left", "q"])

    class Frames:
        def __len__(self) -> int:
            return 3

        def __iter__(self) -> Iterator[tuple[np.ndarray, HitMap]]:
            for index in range(3):
                produced.append(index)
                yield (
                    np.full((1, 1, 4), index, dtype=np.uint8),
                    HitMap.empty(),
                )

    class ImmediatePrefetch:
        def __init__(self, source: Frames, *, capacity: int) -> None:
            capacities.append(capacity)
            self._items = iter(source)

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            pass

        def __next__(self) -> tuple[np.ndarray, HitMap]:
            return next(self._items)

    class FakeViewer:
        def __init__(self, *, hud: bool) -> None:
            assert not hud

        def show(self, name: str, frame: object) -> None:
            shown.append(len(shown))

        def wait(self) -> str:
            return next(keys)

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(viewer_module, "PrefetchIterator", ImmediatePrefetch)
    monkeypatch.setattr(viewer_module, "Viewer", FakeViewer)

    show.present(cast(show.Deck, Frames()))

    assert capacities == [show.PREFETCH]
    assert produced == [0, 1]
    assert len(shown) == 3
    assert closed == [True]
