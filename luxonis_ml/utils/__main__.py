from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich import print
from typing_extensions import Annotated

from luxonis_ml.utils import LuxonisFileSystem

app = typer.Typer()

UrlArgument = Annotated[str, typer.Argument(..., help="URL of the file.")]


class TypeEnum(str, Enum):
    FILE = "file"
    DIR = "directory"
    ALL = "all"


@app.command()
def get(
    url: UrlArgument,
    save_dir: Annotated[
        Optional[Path], typer.Argument(help="Directory to save the file.")
    ] = None,
):
    """Downloads file from remote storage."""
    LuxonisFileSystem.download(url, save_dir or Path.cwd())


@app.command()
def put(
    file: Annotated[Path, typer.Argument(help="Path to the file to upload.")],
    url: UrlArgument,
):
    """Uploads file to remote storage."""
    LuxonisFileSystem.upload(file, url)


@app.command()
def delete(url: UrlArgument):
    """Deletes file from remote storage."""
    fs = LuxonisFileSystem(url)
    fs.delete_file("")


@app.command()
def ls(
    url: UrlArgument,
    recursive: Annotated[
        bool,
        typer.Option(..., "--recursive", "-r", help="List files recursively."),
    ] = False,
    typ: Annotated[
        TypeEnum,
        typer.Option(
            ...,
            "--type",
            "-t",
            help="Type of the files to list. If not provided, all files will be listed.",
        ),
    ] = TypeEnum.ALL,
):
    """Lists files in the remote directory."""
    fs = LuxonisFileSystem(url.rstrip("/"))
    for file in fs.walk_dir("", recursive=recursive, typ=typ.value):
        print(file)
