from importlib.metadata import version

import rich
import rich.box
import typer
from rich.markup import escape
from rich.table import Table

app = typer.Typer(
    name="Luxonis ML CLI",
    add_completion=True,
    pretty_exceptions_show_locals=False,
)

try:
    from luxonis_ml.data.__main__ import app as data_app

    app.add_typer(data_app, name="data", help="Dataset utilities.")
except ImportError:
    pass

try:
    from luxonis_ml.utils.__main__ import app as utils_app

    app.add_typer(utils_app, name="fs", help="Filesystem utilities.")
except ImportError:
    pass

try:
    from luxonis_ml.nn_archive.__main__ import app as utils_app

    app.add_typer(utils_app, name="archive", help="NN Archive utilities.")
except ImportError:
    pass


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"LuxonisML: {version('luxonis_ml')}")
        raise typer.Exit


@app.callback()
def main(
    _: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    return


@app.command()
def checkhealth():
    """Check the health of the Luxonis ML library."""
    table = Table(
        title="Health Check",
        box=rich.box.ROUNDED,
    )
    table.add_column("Module", header_style="magenta i")
    table.add_column("Status", header_style="magenta i")
    table.add_column("Error", header_style="magenta i", max_width=50)
    for submodule in ["data", "utils", "nn_archive"]:
        error_message = ""
        try:
            __import__(f"luxonis_ml.{submodule}")
            status = "✅"
            style = "green"
        except ImportError as e:
            status = "❌"
            style = "red"
            error_message = escape(str(e.args[0]))
            if len(e.args) > 1:
                error_message += f" [bold]{escape(e.args[1])}[/bold]"

        table.add_row(submodule, status, error_message, style=style)

    console = rich.console.Console()
    console.print(table)


if __name__ == "__main__":
    app()
