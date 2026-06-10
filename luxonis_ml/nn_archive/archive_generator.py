import json
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Literal

from luxonis_ml.typing import PathType

from .config import Config


class ArchiveGenerator:
    """Build NN Archive files from configuration and model executables.

    The generated archive is a compressed tar file that contains
    ``config.json`` and the provided executable files.

    Attributes:
        archive_name: Archive file name, including the ``.tar`` and
            compression suffix.
        save_path: Directory where the archive is written.
        executables_paths: Paths to executable files included in the
            archive.
        compression: Compression algorithm used for the tar archive.
        cfg: Validated archive configuration.

    """

    def __init__(
        self,
        archive_name: str,
        save_path: PathType,
        cfg_dict: dict,
        executables_paths: list[PathType],
        compression: Literal["xz", "gz", "bz2"] = "xz",
    ):
        """Create a generator for an NN Archive.

        Args:
            archive_name: Desired archive file name. The ``.tar`` and
                compression suffix are added if missing.
            save_path: Directory where the archive should be written.
            cfg_dict: Archive configuration used to build ``config.json``.
            executables_paths: Paths to model executable files to include.
            compression: Compression algorithm for the tar archive.

        Raises:
            ValueError: If ``compression`` is not one of ``"xz"``,
                ``"gz"``, or ``"bz2"``.

        """
        self.save_path = Path(save_path)
        self.executables_paths = executables_paths

        if compression not in ["xz", "gz", "bz2"]:
            raise ValueError(
                "Invalid compression type. Must be one of 'xz', 'gz', 'bz2'."
            )
        self.compression = compression

        self.archive_name = (
            archive_name
            if archive_name.endswith(f".tar.{self.compression}")
            else f"{archive_name}.tar.{self.compression}"
        )

        self.cfg = Config(  # pydantic config check
            config_version=cfg_dict["config_version"], model=cfg_dict["model"]
        )

    def make_archive(self) -> Path:
        """Generate the NN Archive file.

        Returns:
            Path to the generated archive.

        """

        # create an in-memory file-like config object
        json_data, json_buffer = self._make_json()

        # construct .tar archive
        archive_path = self.save_path / self.archive_name
        with tarfile.open(
            archive_path,
            f"w:{self.compression}",  # type: ignore
        ) as tar:
            # add executables
            for executable_path in map(Path, self.executables_paths):
                tar.add(executable_path, arcname=executable_path.name)
            # add config JSON
            tarinfo = tarfile.TarInfo(name="config.json")
            tarinfo.size = len(json_data)
            json_buffer.seek(0)  # reset the buffer to the beginning
            tar.addfile(tarinfo, json_buffer)

        return archive_path

    def _make_json(self) -> tuple[bytes, BytesIO]:
        """Create an in-memory ``config.json`` file.

        Returns:
            Encoded JSON data and a file-like buffer containing it.

        """

        # read-in config data as dict
        data = json.loads(self.cfg.model_dump_json())

        # create an in-memory file-like object
        json_buffer = BytesIO()

        # encode the dictionary as bytes and write it to the in-memory file
        json_data = json.dumps(data, indent=4).encode("utf-8")
        json_buffer.write(json_data)

        return json_data, json_buffer
