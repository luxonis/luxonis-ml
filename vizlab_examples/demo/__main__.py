"""A tour of vizlab, presented in a window and drawn by vizlab itself.

    python -m vizlab_examples.demo             # page through it on screen
    python -m vizlab_examples.demo --save out  # write the slides instead

Every slide carries three panels: what the feature is, the snippet that draws
it, and the picture that snippet evaluated to. The prose card, the
syntax-coloured code card and the frame around them are all drawn with the same
`Canvas`, fonts and inline markup the examples themselves use, and the window is
the same `Viewer` that ``luxonis_ml data inspect`` opens.

Keys: **space**, ``n``/``j``/``l`` or the arrows to go on; ``p``/``k``/``h``
or the arrows to go back; **Home**/**End** for the ends; ``q`` to quit.
"""

import sys
from pathlib import Path

from rich import print

from .show import build, present


def save(deck: list, directory: Path) -> None:
    """Write the deck to ``directory`` as numbered PNG files."""
    from luxonis_ml.vizlab import Image

    directory.mkdir(parents=True, exist_ok=True)
    for number, (slide, _hits) in enumerate(deck, start=1):
        Image(slide).save(directory / f"slide-{number:02d}.png")
    print(f"{len(deck)} slides -> {directory}")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    target: Path | None = None
    if "--save" in args:
        position = args.index("--save")
        if position + 1 >= len(args):
            raise SystemExit("--save needs a directory")
        target = Path(args[position + 1])

    deck = build()
    if target is not None:
        save(deck, target)
        return
    print(
        f"{len(deck)} slides — [b]space[/b]/n/j/l or [b]→[/b] next, "
        "[b]p[/b]/k/h or [b]←[/b] back, [b]Home[/b]/[b]End[/b] for the ends, "
        "[b]q[/b] quit"
    )
    present(deck)


if __name__ == "__main__":
    main()
