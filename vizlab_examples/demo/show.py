"""Measure the deck and render it as it is presented in a window.

`build` determines the sizes shared by the deck without rendering any slides.
Iteration runs `SETUP` once, then evaluates and lays out each requested slide in
one shared namespace. `present` feeds that iterator through the same bounded
background prefetcher and `Viewer` that ``luxonis_ml data inspect`` uses.

A slide that raises does not stop the demo: its traceback is drawn where its
picture would have been, so a missing optional dependency costs one slide
rather than the presentation.
"""

import ast
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from rich import print

from luxonis_ml.vizlab import (
    DARK_THEME,
    Color,
    HitMap,
    Image,
    Renderable,
    RenderOptions,
    set_default_options,
)

from . import page
from .slides import GLOSSARY, SETUP, SLIDES, Slide

#: The window's title bar, and the keys that drive the deck. The title is
#: deliberately ASCII: it is passed straight to OpenCV's C API, which looks
#: windows up by that exact string, and a non-ASCII one does not survive the
#: round trip on the Qt build.
WINDOW = "vizlab - a tour drawn by vizlab"
#: Match ``luxonis_ml data inspect``: keep two fully rendered slides ready
#: while the viewer waits for input.
PREFETCH = 2
#: Three ways to page, because a reader reaches for whichever they already
#: know: the arrows, vim's hjkl, and the space bar. `Viewer.wait` reports the
#: arrow and page keys by name, so they sit in the same set as the letters.
NEXT_KEYS = frozenset(
    {" ", "n", "f", "l", "j", "\r", "\n", ".", "right", "down", "pagedown"}
)
PREV_KEYS = frozenset({"p", "b", "h", "k", ",", "left", "up", "pageup"})
FIRST_KEYS = frozenset({"home"})
LAST_KEYS = frozenset({"end"})
QUIT_KEYS = frozenset({"q", "\x1b"})


def _evaluate(source: str, namespace: dict) -> Renderable | None:
    """Run ``source``; return its final value when that value is drawable.

    The trailing expression is evaluated on its own so its value is available
    here — the same trick a notebook uses to decide what to display, and what
    keeps a slide's snippet honest: what you read is what drew the picture.
    """
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop().value  # type: ignore[attr-defined]
        exec(compile(tree, "<slide>", "exec"), namespace)  # noqa: S102
        value = eval(  # noqa: S307 - the demo's own source
            compile(ast.Expression(last), "<slide>", "eval"), namespace
        )
    else:
        exec(compile(tree, "<slide>", "exec"), namespace)  # noqa: S102
        value = None
    return value if isinstance(value, Renderable) else None


def _failure(message: str) -> np.ndarray:
    """Draw a slide's traceback in place of the picture it did not make."""
    lines = [line for line in message.strip().splitlines() if line.strip()]
    return page.text_block(
        [page.markdown(line) for line in lines[-5:]],
        title="this slide did not run",
        size=page.CODE_SIZE,
        color=Color(226, 138, 138),
        leading=page.CODE_LEADING,
        title_color=Color(240, 170, 170),
    )


SlideFrame = tuple[np.ndarray, HitMap]


@dataclass(frozen=True, slots=True)
class Deck:
    """A consistently sized deck whose slides are rendered as it is iterated."""

    slides: tuple[Slide, ...]
    column_w: int
    code_size: float

    def __len__(self) -> int:
        return len(self.slides)

    def __iter__(self) -> Iterator[SlideFrame]:
        namespace: dict = {
            "__name__": "__vizlab_demo__",
            "FRAME_W": page.PICTURE_RIGHT
            - (page.CONTENT_X + self.column_w + page.COLUMN_GAP),
            "FRAME_H": page.PICTURE_BOTTOM - page.CONTENT_TOP,
        }
        exec(compile(SETUP, "<setup>", "exec"), namespace)  # noqa: S102

        for number, slide in enumerate(self.slides, start=1):
            picture: Renderable | np.ndarray | None
            try:
                picture = _evaluate(slide.source, namespace)
            except Exception:  # one bad slide must not end the demo
                picture = _failure(traceback.format_exc())
            yield page.compose_slide(
                slide.title,
                [slide.body],
                slide.source.strip(),
                picture,
                position=f"{number} / {len(self)}",
                column_w=self.column_w,
                progress=number / len(self),
                glossary=GLOSSARY,
                code_size=self.code_size,
            )


def build(slides: Sequence[Slide] = SLIDES) -> Deck:
    """Measure the deck without rendering its slides."""
    slides = tuple(slides)
    # `hover_metadata` makes the LDF adapter turn a detection's metadata into
    # a tooltip, so the earliest slides already have something under the
    # cursor — the feature is met before it is explained. PrefetchIterator
    # carries these context-local options into its rendering thread.
    set_default_options(RenderOptions(theme=DARK_THEME, hover_metadata=True))

    # One column width for the whole deck, wide enough for its widest
    # snippet, so nothing is clipped and every slide's column lines up.
    terms = frozenset(GLOSSARY)
    column_w = min(
        page.MAX_COLUMN_W,
        max(
            page.code_block(s.source.strip(), terms=terms).shape[1]
            for s in slides
        ),
    )
    # One code size for the deck, chosen so its tallest snippet still fits.
    prose_heights = [
        page.text_block(
            page.wrap_prose([s.body], width=column_w, terms=terms),
            size=page.PROSE_SIZE,
            color=page.PROSE_FG,
            leading=page.PROSE_LEADING,
            width=column_w,
        ).shape[0]
        for s in slides
    ]
    code_size = page.fitted_code_size(
        [s.source for s in slides], prose_heights
    )

    return Deck(slides, column_w, code_size)


def present(deck: Deck) -> None:
    """Show the deck in a window; page through it until told to stop.

    Every slide goes through `Viewer.show`, including one with nothing to
    hover. Mixing in `show_blocking` does not work: it drops the window's
    hover state on the way past, so the next slide that *does* have tooltips
    inherits a live mouse callback with nothing behind it and hovering it
    silently does nothing. One path, one state.

    If the hover path raises, the slide it raised on is named and the deck
    tries the next one rather than giving up on hover for good: one slide with
    a bad tooltip used to take every later slide's tooltips down with it, and
    silently, since the message goes to the terminal and the deck is on screen.
    Only a run of failures — a build that genuinely cannot install a mouse
    callback — falls back for the rest of the deck.

    The viewer is imported here rather than at module scope so building the
    deck never pulls in a windowing backend — the subpackage is opt-in.
    """
    from luxonis_ml.vizlab import Frame
    from luxonis_ml.vizlab.viewer import PrefetchIterator, Viewer

    viewer = Viewer(hud=False)
    index, hover, failures = 0, True, 0
    try:
        with PrefetchIterator(deck, capacity=PREFETCH) as slides:
            rendered: list[SlideFrame] = []
            while True:
                while len(rendered) <= index:
                    rendered.append(next(slides))
                slide, hits = rendered[index]
                if hover:
                    try:
                        viewer.show(WINDOW, Frame(Image(slide), hits))
                        key = viewer.wait().lower()
                        failures = 0
                    except Exception as error:
                        failures += 1
                        print(
                            f"[yellow]slide {index + 1}: hover failed:[/yellow] "
                            f"{error}"
                        )
                        hover = failures < 3
                        continue
                else:
                    key = viewer.show_blocking(WINDOW, Image(slide)).lower()
                if key in QUIT_KEYS:
                    return
                if key in NEXT_KEYS:
                    index = min(index + 1, len(deck) - 1)
                elif key in PREV_KEYS:
                    index = max(index - 1, 0)
                elif key in FIRST_KEYS:
                    index = 0
                elif key in LAST_KEYS:
                    index = len(deck) - 1
    finally:
        viewer.close()
