from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter
from rich import print

from luxonis_ml.utils import LuxonisFileSystem

app = App(help="Filesystem utilities.")


@app.command(alias="pull")
def get(
    url: str,
    save_dir: Path | None = None,
):
    """Download a file from remote storage.

    Args:
        url (str): URL of the file to download.
        save_dir (Path, optional): Directory to save the file.
            Defaults to current working directory.
    """
    LuxonisFileSystem.download(url, save_dir or Path.cwd())


@app.command(alias="push")
def put(
    file: Path,
    url: str,
):
    """Upload a file to remote storage.

    Args:
        file (Path): Path to the file to upload.
        url (str): URL of the file.
    """
    LuxonisFileSystem.upload(file, url)


@app.command(alias=["rm", "remove"])
def delete(url: str):
    """Delete a file from remote storage.

    Args:
        url (str): URL of the file to delete.
    """
    LuxonisFileSystem(url).delete_file("")


@app.command
def ls(
    url: str,
    *,
    recursive: Annotated[
        bool,
        Parameter(alias="-r", negative=""),
    ] = False,
    typ: Annotated[
        Literal["file", "dir", "all"],
        Parameter(
            name=["--type", "-t"],
        ),
    ] = "all",
):
    """List files in the remote directory.

    Args:
        url (str): URL of the directory to list.
        recursive (bool): Whether to list files recursively.
        typ (str): Type of files to list.
    """
    if not url.endswith("://"):
        url = url.rstrip("/")
    print(url)
    fs = LuxonisFileSystem(url)
    print(fs.url)
    for file in fs.walk_dir("", recursive=recursive, typ=typ):
        print(file)
