from importlib.metadata import version

import rich
import rich.box
from cyclopts import App, Group
from rich.markup import escape
from rich.table import Table

app = App(
    name="luxonis_ml",
    version=lambda: f"LuxonisML: {version('luxonis_ml')}",
    version_flags=["-v", "--version"],
    help="MLOps tools for training models for Luxonis devices.",
)
global_group = Group("Global Parameters", sort_key=0)
app["--help"].group = app["--version"].group = global_group


app.register_install_completion_command(group=global_group)


def _register_subapp(module: str, name: str) -> None:
    try:
        imported = __import__(module, fromlist=["app"])
        subapp = imported.app
    except ImportError:
        return

    if not isinstance(subapp, App):
        return

    app.command(subapp, name=name)


_register_subapp("luxonis_ml.data.__main__", "data")
_register_subapp("luxonis_ml.utils.__main__", "fs")
_register_subapp("luxonis_ml.nn_archive.__main__", "archive")


@app.command
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
