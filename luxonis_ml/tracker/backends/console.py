# pyright: strict
"""The console backend of the tracker.

`ConsoleBackend` prints the results of a run as tables, to the terminal
and to the log file. It turns on with ``LuxonisTracker(console=True)``,
or with a mapping of `ConsoleOptions`, and needs no extra.

See:
    `luxonis_ml.tracker` for what each backend does with each logging
    call, and for an example of the output.

"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from io import StringIO
from typing import Any, TypedDict

import numpy.typing as npt
from loguru import logger
from rich import get_console
from rich.cells import cell_len
from rich.console import Console, RenderableType
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from typing_extensions import Unpack

from luxonis_ml.typing import ParamValue

from .base import (
    RunContext,
    RunStatus,
    TableCell,
    TrackerBackend,
    check_options,
)


class ConsoleOptions(TypedDict, total=False):
    """The options of `ConsoleBackend`.

    Attributes:
        rich: Print in colour on the console of ``rich``, and send a
            copy without colour to the log file only. ``False`` sends
            the copy without colour through ``logger.info``, to the
            terminal and to the log file. The default is ``True``.
        prefixes: Print only the metrics and the matrices whose names
            start with one of these prefixes, such as
            ``["val/", "test/"]``. Each prefix gets a block of its own.
            ``None`` prints all of them.

    """

    rich: bool
    prefixes: Sequence[str] | None


class ConsoleBackend(TrackerBackend, register_name="console"):
    """Print the metrics, the confusion matrices and the tables.

    A `log_metrics` call prints one block for each prefix that some of
    its keys start with. The block opens with a rule that names the
    prefix, the epoch and the step. One table follows, with the keys
    as a tree: each ``/`` of a key is one level, and a level that has
    no value of its own is a heading. A key comes after its parent and
    after the keys that share the parent and came first. A key
    ``epoch``, as Lightning adds it to each call, goes into the rule
    and not into the table.

    A confusion matrix waits for the next block of its prefix, and
    prints at the end of it. Lightning logs the scalars of an epoch
    after the module logs its matrices, so the matrices of an epoch
    print with its scalars. `flush` and `close` print the matrices that
    still wait. A table prints at once, whatever its name.

    The backend prints nothing for the hyperparameters and the images,
    and it stores no artifacts.

    With ``rich`` on, the output goes to the global console of
    ``rich``. The progress bars of ``rich``, such as the one of
    Lightning, draw on that console, so the tables print above a live
    bar. The copy without colour goes to the sinks of ``loguru`` with
    ``file_only`` set, which `luxonis_ml.utils.setup_logging` keeps off
    the terminal.

    Attributes:
        rich: Whether the output is in colour, see `ConsoleOptions`.
        prefixes: The prefixes of the names to print. An empty prefix
            stands for all names.

    """

    def __init__(
        self, run: RunContext, **options: Unpack[ConsoleOptions]
    ) -> None:
        """Check the options.

        Args:
            run: The run to log to.
            **options: See `ConsoleOptions`.

        Raises:
            TypeError: If an option is unknown.

        """
        super().__init__(run)
        check_options(options, ConsoleOptions.__optional_keys__)
        self.rich = options.get("rich", True)
        self.prefixes = list(options.get("prefixes") or [""])
        self._matrices: defaultdict[str, list[Table]] = defaultdict(list)

    def start(self) -> None:
        """Do nothing, because the terminal needs no connection."""

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        """Do nothing, because the configuration of a run is too long to
        print.

        Args:
            params: Ignored.

        """

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Print a block for each prefix, with the matrices that wait
        for it.

        Args:
            metrics: The metric values, keyed by metric name.
            step: The training step of the values, which the rule of the
                block shows.

        """
        epoch = metrics.get("epoch")
        for prefix in self.prefixes:
            values = {
                key.removeprefix(prefix): value
                for key, value in metrics.items()
                if key.startswith(prefix) and key != "epoch"
            }
            if not values:
                continue
            title = [prefix.rstrip("/")] if prefix else []
            if epoch is not None:
                title.append(f"epoch {epoch:g}")
            title.append(f"step {step}")
            self._print(
                [
                    Rule(Text(" · ".join(title)), style="bold magenta"),
                    _tree_table(values),
                    *self._matrices.pop(prefix, []),
                    Rule(style="bold magenta"),
                ]
            )

    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        """Do nothing, because a terminal shows no images.

        Args:
            name: Ignored.
            image: Ignored.
            step: Ignored.

        """

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        labels: Sequence[str],
    ) -> None:
        r"""Keep the confusion matrix for the next block of its prefix.

        The table has the true classes on the rows and the predicted
        ones on the columns. Its title is the name without the prefix.
        The diagonal, which holds the right predictions, is green, and
        a zero is grey. A matrix whose name has none of the prefixes is
        not printed.

        Args:
            matrix: The counts, of shape :math:`\left(N, N\right)`.
            name: Name of the matrix.
            step: Ignored, because the block shows the step.
            labels: The :math:`N` class names.

        """
        for prefix in self.prefixes:
            if name.startswith(prefix):
                table = _matrix_table(
                    name.removeprefix(prefix), matrix, labels
                )
                self._matrices[prefix].append(table)
                return

    def log_table(
        self,
        name: str,
        rows: Sequence[Sequence[TableCell]],
        columns: Sequence[str],
        step: int,
    ) -> None:
        """Print the table, titled ``<name> · step <step>``.

        Args:
            name: Name of the table.
            rows: The rows, each with one cell for each column.
            columns: Names of the columns.
            step: The training step of the table.

        """
        table = _table(f"{name} · step {step}", columns)
        for row in rows:
            table.add_row(*(Text(_format(cell)) for cell in row))
        self._print([table])

    def flush(self) -> None:
        """Print the matrices that wait for a block."""
        matrices = [
            table for tables in self._matrices.values() for table in tables
        ]
        self._matrices.clear()
        if matrices:
            self._print(matrices)

    def close(self, status: RunStatus) -> None:
        """Print the matrices that wait for a block.

        Args:
            status: Ignored.

        """
        self.flush()

    def _print(self, renderables: Sequence[RenderableType]) -> None:
        """Print to the terminal and to the log file, as the ``rich``
        option says.
        """
        console = get_console()
        buffer = StringIO()
        plain = Console(
            file=buffer, width=console.width, no_color=True, highlight=False
        )
        for renderable in renderables:
            plain.print(renderable)
        # the tables start on a line of their own after the log prefix
        text = "\n" + buffer.getvalue().rstrip("\n")
        if self.rich:
            for renderable in renderables:
                console.print(renderable)
            logger.bind(file_only=True).info(text)
        else:
            logger.info(text)


def _format(value: TableCell) -> str:
    """Write a float with five decimals, or in the ``e`` notation when
    the decimals would hide it.

    Example:
        >>> [_format(v) for v in (0.412340164, 2e-06, 0.0, 812, "car")]
        ['0.41234', '2.000e-06', '0.00000', '812', 'car']

    """
    if not isinstance(value, float):
        return str(value)
    if not value or abs(value) >= 1e-3:
        return f"{value:.5f}"
    return f"{value:.3e}"


def _table(title: str | None, columns: Sequence[str]) -> Table:
    """Create a table with the columns, the first of them magenta and
    the others right-aligned.
    """
    table = Table(
        title=None if title is None else Text(title),
        header_style="bold magenta",
        title_style="bold",
        # rich would wrap a title that is wider than the columns
        min_width=cell_len(title or ""),
    )
    for i, column in enumerate(columns):
        if i == 0:
            table.add_column(Text(column), style="magenta")
        else:
            table.add_column(Text(column), justify="right")
    return table


def _tree_order(keys: Sequence[str]) -> list[str]:
    """Put each key after its parent, in the order of first sight.

    Example:
        >>> _tree_order(["loss/head/bce", "metric/head/map", "loss"])
        ['loss', 'loss/head/bce', 'metric/head/map']

    """
    first: dict[str, int] = {}
    for i, key in enumerate(keys):
        parts = key.split("/")
        for depth in range(1, len(parts) + 1):
            first.setdefault("/".join(parts[:depth]), i)

    def position(key: str) -> list[int]:
        parts = key.split("/")
        return [
            first["/".join(parts[:depth])]
            for depth in range(1, len(parts) + 1)
        ]

    return sorted(keys, key=position)


def _tree_table(values: Mapping[str, float]) -> Table:
    """Show the values in one table, with each level of their keys
    indented by two spaces.
    """
    table = _table(None, ["Name", "Value"])
    shown: set[str] = set()
    for key in _tree_order(list(values)):
        parts = key.split("/")
        for depth, part in enumerate(parts):
            path = "/".join(parts[: depth + 1])
            if path in shown:
                continue
            shown.add(path)
            name = Text("  " * depth + part)
            if path in values:
                table.add_row(name, Text(_format(values[path])))
            else:
                name.stylize("bold bright_black")
                table.add_row(name, Text(""))
    return table


def _matrix_table(
    title: str, matrix: npt.NDArray[Any], labels: Sequence[str]
) -> Table:
    """Show a confusion matrix, with the diagonal green and the zeros
    grey.
    """
    table = _table(title, ["true \\ pred", *labels])
    table.title_style = "italic"
    for i, (label, row) in enumerate(
        zip(labels, matrix.tolist(), strict=True)
    ):
        cells: list[Text] = []
        for j, count in enumerate(row):
            style = (
                "bold green"
                if i == j
                else "bright_black"
                if count == 0
                else ""
            )
            cells.append(Text(_format(count), style=style))
        table.add_row(Text(label), *cells)
    return table
