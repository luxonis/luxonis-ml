import json
import tarfile
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter, validators
from rich import print
from rich.panel import Panel
from rich.pretty import Pretty

from luxonis_ml.nn_archive import Config

app = App(help="NN Archive utilities.", help_flags="--help")


@app.command
def inspect(
    path: Annotated[
        Path,
        Parameter(
            validator=validators.Path(exists=True),
        ),
    ],
    *,
    inputs: Annotated[
        bool,
        Parameter(alias="-i", negative=""),
    ] = False,
    metadata: Annotated[
        bool,
        Parameter(alias="-m", negative=""),
    ] = False,
    outputs: Annotated[
        bool,
        Parameter(alias="-o", negative=""),
    ] = False,
    heads: Annotated[
        bool,
        Parameter(alias="-h", negative=""),
    ] = False,
    buildinfo: Annotated[
        bool,
        Parameter(alias="-b", negative=""),
    ] = False,
):
    """Print NN Archive configuration.

    If no options are provided, all info is printed.

    Args:
        path (Path): Path to the NN Archive.
        inputs (bool, optional): Print inputs info.
        metadata (bool, optional): Print metadata.
        outputs (bool, optional): Print outputs info.
        heads (bool, optional): Print heads info.
        buildinfo (bool, optional): Print build info if available.
    """

    if not any([inputs, metadata, outputs, heads, buildinfo]):
        inputs = metadata = outputs = heads = True

    with tarfile.open(path) as tar:
        extracted_cfg = tar.extractfile("config.json")
        if buildinfo:
            extracted_buildinfo = tar.extractfile("buildinfo.json")
            if extracted_buildinfo is not None:
                print(
                    Panel.fit(
                        Pretty(json.loads(extracted_buildinfo.read())),
                        title="Build Info",
                    )
                )

        if extracted_cfg is None:
            raise RuntimeError("Config JSON not found in the archive.")

        cfg = Config.model_validate_json(extracted_cfg.read())

    if metadata:
        print(Panel.fit(Pretty(cfg.model.metadata), title="Metadata"))
    if heads:
        print(Panel.fit(Pretty(cfg.model.heads), title="Heads"))
    if inputs:
        print(Panel.fit(Pretty(cfg.model.inputs), title="Inputs"))
    if outputs:
        print(Panel.fit(Pretty(cfg.model.outputs), title="Outputs"))


@app.command
def extract(
    path: Annotated[
        Path,
        Parameter(
            validator=validators.Path(exists=True),
        ),
    ],
    destination: Annotated[
        Path | None,
        Parameter(
            name="--dest",
            alias="-d",
        ),
    ] = None,
):
    """Extract an NN Archive.

    Extracts the NN Archive to the destination path. By default, the
    Archive is extracted to the current working directory.

    Args:
        path (Path): Path to the NN Archive.
        destination (Path or None, optional): Path where to extract the Archive.
            If not provided, the Archive is extracted to the current
            working directory.
    """

    destination = destination or Path.cwd()
    extract_path = destination / path.name.split(".")[0]
    extract_path.mkdir(exist_ok=True, parents=True)

    def safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
        """Filter members to prevent path traversal attacks."""
        safe_files = []
        root = extract_path.resolve()
        for member in tar.getmembers():
            # Normalize path and ensure it's within the extraction folder
            if member.issym() or member.islnk():
                print(f"Skipping unsafe link: {member.name}")
                continue

            target = (root / member.name).resolve()
            if target == root or root in target.parents:
                safe_files.append(member)
            else:
                print(f"Skipping unsafe file: {member.name}")
        return safe_files

    with tarfile.open(path) as tf:
        for member in safe_members(tf):
            tf.extract(member, extract_path)

    print(f"Archive extracted to: {extract_path}")
