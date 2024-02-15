import json
import os
import tarfile
from io import BytesIO
from typing import List

from .config import Config


class ArchiveGenerator:
    """Generator of abstracted NN archive (.tar) files containing config and model files
    (executables).

    @type archive_name: str
    @ivar archive_name: Desired archive file name.
    @type save_path: str
    @ivar save_path: Path to where we want to save the archive file.
    @type cfg_dict: dict
    @ivar cfg_dict: Archive configuration dict.
    @type executables_paths: list
    @ivar executables_paths: Paths to relevant model executables.
    """

    def __init__(
        self,
        archive_name: str,
        save_path: str,
        cfg_dict: dict,
        executables_paths: List[str],
    ):
        self.archive_name = (
            archive_name
            if archive_name.endswith(".tar.gz")
            else f"{archive_name}.tar.gz"
        )
        self.mode = "w:gz"

        self.save_path = save_path
        self.executables_paths = executables_paths

        self.cfg = Config(  # pydantic config check
            config_version=cfg_dict["config_version"], stages=cfg_dict["stages"]
        )

    def make_archive(self):
        """Run NN archive (.tar) file generation."""

        # create an in-memory file-like config object
        json_data, json_buffer = self._make_json()

        # construct .tar archive
        with tarfile.open(
            os.path.join(self.save_path, self.archive_name), self.mode
        ) as tar:
            # add executables
            for executable_path in self.executables_paths:
                tar.add(executable_path, arcname=os.path.basename(executable_path))

            # add config JSON
            tarinfo = tarfile.TarInfo(name=f"{self.archive_name}.json")
            tarinfo.size = len(json_data)
            json_buffer.seek(0)  # reset the buffer to the beginning
            tar.addfile(tarinfo, json_buffer)

    def _make_json(self):
        """Create an in-memory config data file-like object."""

        # read-in config data as dict
        data = json.loads(self.cfg.model_dump_json())

        # create an in-memory file-like object
        json_buffer = BytesIO()

        # encode the dictionary as bytes and write it to the in-memory file
        json_data = json.dumps(data, indent=4).encode("utf-8")
        json_buffer.write(json_data)

        return json_data, json_buffer