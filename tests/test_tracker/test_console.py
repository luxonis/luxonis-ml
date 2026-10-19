from io import StringIO
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from loguru import logger
from rich.console import Console

import luxonis_ml.tracker.backends.console as console_module
from luxonis_ml.tracker import ConsoleBackend, RunContext

VAL_BLOCK = [
    "",
    "───────────────── val · epoch 3 · step 300 ─────────────────",
    "┏━━━━━━━━━┳━━━━━━━━━┓",
    "┃ Name    ┃   Value ┃",
    "┡━━━━━━━━━╇━━━━━━━━━┩",
    "│ loss    │ 0.31000 │",
    "│   head  │         │",
    "│     ce  │ 0.30000 │",
    "│ metric  │         │",
    "│   head  │         │",
    "│     acc │ 0.94000 │",
    "└─────────┴─────────┘",
    "          head/cm",
    "┏━━━━━━━━━━━━━┳━━━━━┳━━━━━┓",
    "┃ true \\ pred ┃ cat ┃ dog ┃",
    "┡━━━━━━━━━━━━━╇━━━━━╇━━━━━┩",
    "│ cat         │   4 │   0 │",
    "│ dog         │   1 │   3 │",
    "└─────────────┴─────┴─────┘",
    "─" * 60,
]
"""The block of `log_val_epoch`, 60 columns wide."""


def lines(text: str) -> list[str]:
    """Split the output into lines, without the padding of a title."""
    return [line.rstrip() for line in text.split("\n")]


def log_val_epoch(backend: ConsoleBackend) -> None:
    """Log what Lightning sends at the end of a validation epoch: the
    matrices first, then the scalars, the losses children first.
    """
    matrix = np.array([[4, 0], [1, 3]])
    backend.log_matrix(matrix, "val/head/cm", 3, ["cat", "dog"])
    backend.log_metrics(
        {
            "val/loss/head/ce": 0.3,
            "val/loss": 0.31,
            "val/metric/head/acc": 0.94,
            "train/loss": 1.0,
            "epoch": 3,
        },
        300,
    )


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Stand in for the global console of ``rich``, 60 columns wide."""
    console = Console(
        file=StringIO(),
        width=60,
        record=True,
        force_terminal=True,
        color_system="standard",
    )
    monkeypatch.setattr(console_module, "get_console", lambda: console)
    return console


@pytest.fixture
def records(terminal: Console) -> Any:
    """Collect the log records of the test."""
    collected: list[Any] = []
    sink = logger.add(lambda m: collected.append(m.record), format="{message}")
    yield collected
    logger.remove(sink)


def plain(run: RunContext, **options: Any) -> ConsoleBackend:
    return ConsoleBackend(run, rich=False, **options)


def messages(records: list[dict[str, Any]]) -> list[list[str]]:
    return [lines(record["message"]) for record in records]


def test_a_block_shows_its_prefix_as_a_tree_with_its_matrices(
    run: RunContext, records: list[dict[str, Any]], terminal: Console
):
    backend = plain(run, prefixes=["val/", "test/"])
    backend.log_matrix(np.eye(2), "train/head/cm", 3, ["cat", "dog"])

    log_val_epoch(backend)
    backend.close("success")

    # the train keys and the train matrix have no prefix of the backend
    assert messages(records) == [VAL_BLOCK]
    assert terminal.export_text() == ""


def test_a_matrix_without_a_block_prints_at_the_flush(
    run: RunContext, records: list[dict[str, Any]]
):
    backend = plain(run, prefixes=["val/", "test/"])
    backend.log_matrix(np.array([[1]]), "test/head/cm", 3, ["cat"])
    backend.flush()
    backend.flush()

    assert messages(records) == [
        [
            "",
            "       head/cm",
            "┏━━━━━━━━━━━━━┳━━━━━┓",
            "┃ true \\ pred ┃ cat ┃",
            "┡━━━━━━━━━━━━━╇━━━━━┩",
            "│ cat         │   1 │",
            "└─────────────┴─────┘",
        ]
    ]


def test_each_prefix_gets_a_block(
    run: RunContext, records: list[dict[str, Any]]
):
    backend = plain(run, prefixes=["val/", "test/"])

    backend.log_metrics({"val/loss": 0.5, "test/loss": 0.6}, 5)

    rules = [message[1].strip("─ ") for message in messages(records)]
    assert rules == ["val · step 5", "test · step 5"]


def test_without_prefixes_each_key_prints_and_epoch_goes_to_the_rule(
    run: RunContext, records: list[dict[str, Any]]
):
    backend = plain(run)

    backend.log_metrics({"lr": 2e-6, "epoch": 1}, 10)
    backend.log_metrics({"loss": 1}, 11)

    assert messages(records) == [
        [
            "",
            "──────────────────── epoch 1 · step 10 ─────────────────────",
            "┏━━━━━━┳━━━━━━━━━━━┓",
            "┃ Name ┃     Value ┃",
            "┡━━━━━━╇━━━━━━━━━━━┩",
            "│ lr   │ 2.000e-06 │",
            "└──────┴───────────┘",
            "─" * 60,
        ],
        [
            "",
            "───────────────────────── step 11 ──────────────────────────",
            "┏━━━━━━┳━━━━━━━┓",
            "┃ Name ┃ Value ┃",
            "┡━━━━━━╇━━━━━━━┩",
            "│ loss │     1 │",
            "└──────┴───────┘",
            "─" * 60,
        ],
    ]


def test_a_table_prints_at_once(
    run: RunContext, records: list[dict[str, Any]]
):
    backend = plain(run, prefixes=["val/"])

    backend.log_table("quant", [("map", 0.41)], ["Name", "INT8"], 7)

    assert messages(records) == [
        [
            "",
            "  quant · step 7",
            "┏━━━━━━┳━━━━━━━━━┓",
            "┃ Name ┃    INT8 ┃",
            "┡━━━━━━╇━━━━━━━━━┩",
            "│ map  │ 0.41000 │",
            "└──────┴─────────┘",
        ]
    ]


def test_rich_prints_in_colour_and_copies_to_the_log_file(
    run: RunContext, records: list[dict[str, Any]], terminal: Console
):
    backend = ConsoleBackend(run, prefixes=["val/"])

    log_val_epoch(backend)

    assert lines(terminal.export_text(clear=False)) == [*VAL_BLOCK[1:], ""]
    # the diagonal is bold green
    assert "\x1b[1;32m  4\x1b[0m" in terminal.export_text(styles=True)
    assert [record["extra"] for record in records] == [{"file_only": True}]
    assert messages(records) == [VAL_BLOCK]


def test_the_hyperparameters_and_the_images_do_not_print(
    run: RunContext,
    records: list[dict[str, Any]],
    terminal: Console,
    image: npt.NDArray[np.uint8],
):
    backend = ConsoleBackend(run)

    backend.start()
    backend.log_hyperparams({"lr": 0.1})
    backend.log_image("val/image", image, 1)
    backend.close("failed")

    assert records == []
    assert terminal.export_text() == ""


def test_an_unknown_option_is_rejected(run: RunContext):
    with pytest.raises(TypeError, match="unknown option 'color'"):
        ConsoleBackend(run, color=True)  # pyright: ignore[reportCallIssue]
