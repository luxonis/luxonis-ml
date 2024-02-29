from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from luxonis_ml.utils import LuxonisFileSystem

app = typer.Typer()

UrlArgument = Annotated[str, typer.Argument(..., help="URL of the file.")]


@app.command()
def get(
    url: UrlArgument,
    save_dir: Annotated[
        Optional[Path], typer.Argument(help="Directory to save the file.")
    ] = None,
):
    """Downloads file from remote storage."""
    fs = LuxonisFileSystem(url)
    fs.get_file(None, str(save_dir or Path.cwd()))


@app.command()
def put(
    file: Annotated[Path, typer.Argument(help="Path to the file to upload.")],
    url: UrlArgument,
):
    """Uploads file to remote storage."""
    fs = LuxonisFileSystem(url)
    fs.put_file(str(file), None)


@app.command()
def delete(url: UrlArgument):
    """Deletes file from remote storage."""
    fs = LuxonisFileSystem(url)
    fs.delete_file(None)


@app.command()
def ls(url: UrlArgument):
    """Lists files in the remote directory."""
    fs = LuxonisFileSystem(url.rstrip("/"))
    for file in fs.walk_dir():
        print(file)
