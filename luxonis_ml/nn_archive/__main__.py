import json
import logging
import tarfile
from pathlib import Path

import typer
from rich import print
from rich.panel import Panel
from rich.pretty import Pretty
from typing_extensions import Annotated, TypeAlias

from luxonis_ml.nn_archive import Config

logger = logging.getLogger(__name__)
app = typer.Typer()


PathArgument: TypeAlias = Annotated[
    Path, typer.Argument(..., help="Path to the NN Archive.")
]


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(..., help="Path to the NN Archive.")],
    inputs: Annotated[
        bool, typer.Option(..., "-i", "--inputs", help="Print inputs info.")
    ] = False,
    metadata: Annotated[
        bool, typer.Option(..., "-m", "--metadata", help="Print metadata.")
    ] = False,
    outputs: Annotated[
        bool, typer.Option(..., "-o", "--outputs", help="Print outputs info.")
    ] = False,
    heads: Annotated[
        bool, typer.Option(..., "-h", "--heads", help="Print heads info.")
    ] = False,
):
    """Prints NN Archive configuration.

    If no options are provided, all info is printed.
    """

    with tarfile.open(path) as tar:
        extracted_cfg = tar.extractfile("config.json")

        if extracted_cfg is None:
            raise RuntimeError("Config JSON not found in the archive.")

        archive_config = Config(**json.loads(extracted_cfg.read().decode()))

    if not any([inputs, metadata, outputs, heads]):
        inputs = metadata = outputs = heads = True

    if metadata:
        print(Panel.fit(Pretty(archive_config.model.metadata), title="Metadata"))
    if heads:
        print(Panel.fit(Pretty(archive_config.model.heads), title="Heads"))
    if inputs:
        print(Panel.fit(Pretty(archive_config.model.inputs), title="Inputs"))
    if outputs:
        print(Panel.fit(Pretty(archive_config.model.outputs), title="Outputs"))


@app.command()
def extract(
    path: Annotated[Path, typer.Argument(..., help="Path to the NN Archive.")],
    destination: Annotated[
        Path,
        typer.Option("-d", "--dest", help="Path where to extract the Archive."),
    ] = Path("."),
):
    """Extracts NN Archive.

    Extracts the NN Archive to the destination path. By default, the Archive is
    extracted to the current working directory.
    """

    extract_path = destination / (path.name.split(".")[0])
    extract_path.mkdir(exist_ok=True, parents=True)

    with tarfile.open(path) as tar:
        tar.extractall(extract_path)

    typer.echo(f"Archive extracted to: {extract_path}")
