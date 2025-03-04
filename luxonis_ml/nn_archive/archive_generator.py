import json
import tarfile
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Tuple

from luxonis_ml.typing import PathType

from .config import Config


class ArchiveGenerator:
    """Generator of abstracted NN archive (.tar) files containing config
    and model files (executables).

    @type archive_name: str
    @ivar archive_name: Desired archive file name.
    @type save_path: str
    @ivar save_path: Path to where we want to save the archive file.
    @type cfg_dict: dict
    @ivar cfg_dict: Archive configuration dict.
    @type executables_paths: list
    @ivar executables_paths: Paths to relevant model executables.
    @type compression: str
    @ivar compression: Type of archive file compression ("xz" for LZMA,
        "gz" for gzip, or "bz2" for bzip2 compression).
    """

    def __init__(
        self,
        archive_name: str,
        save_path: PathType,
        cfg_dict: dict,
        executables_paths: List[PathType],
        compression: Literal["xz", "gz", "bz2"] = "xz",
    ):
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
        """Run NN archive (.tar) file generation."""

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

    def _make_json(self) -> Tuple[bytes, BytesIO]:
        """Create an in-memory config data file-like object."""

        # read-in config data as dict
        data = json.loads(self.cfg.model_dump_json())

        # create an in-memory file-like object
        json_buffer = BytesIO()

        # encode the dictionary as bytes and write it to the in-memory file
        json_data = json.dumps(data, indent=4).encode("utf-8")
        json_buffer.write(json_data)

        return json_data, json_buffer
