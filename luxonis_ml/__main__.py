from importlib.metadata import version

import typer

from luxonis_ml.utils import setup_logging

setup_logging(use_rich=True)

app = typer.Typer(name="Luxonis ML CLI", add_completion=True)

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


def version_callback(value: bool):
    if value:
        print(f"LuxonisML: {version(__package__)}")
        raise typer.Exit()


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
    # Do other global stuff, handle other global options here
    return


if __name__ == "__main__":
    app()
