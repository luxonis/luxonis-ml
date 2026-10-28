"""Generator of NN Archive files from a config and model executables."""

import json
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Literal

from luxonis_ml.typing import PathType

from .config import Config

_WRITE_MODES: dict[
    Literal["xz", "gz", "bz2"], Literal["w:xz", "w:gz", "w:bz2"]
] = {"xz": "w:xz", "gz": "w:gz", "bz2": "w:bz2"}


class ArchiveGenerator:
    """Build NN Archive files from configuration and model executables.

    The generated archive is a compressed tar file that contains
    ``config.json``, the provided executables, and any declared runtime
    dependency libraries.

    Attributes:
        archive_name: Archive file name, including the ``.tar`` and
            compression suffix.
        save_path: Directory where the archive is written.
        executables_paths: Paths to executable files included in the
            archive.
        runtime_dependency_paths: Mapping from declared archive paths to
            local runtime library files.
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
        *,
        runtime_dependency_paths: dict[str, PathType] | None = None,
    ) -> None:
        """Create a generator for an NN Archive.

        Args:
            archive_name: Desired archive file name. The ``.tar`` and
                compression suffix are added if missing.
            save_path: Directory where the archive should be written.
            cfg_dict: Archive configuration used to build ``config.json``.
            executables_paths: Paths to model executable files to include.
            compression: Compression algorithm for the tar archive.
            runtime_dependency_paths: Local files to bundle at the paths
                declared in ``model.runtime_dependencies``.

        Raises:
            ValueError: If ``compression`` is not one of ``"xz"``,
                ``"gz"``, or ``"bz2"``, or if the supplied library paths do
                not match the manifest or collide with other archive members.
            FileNotFoundError: If a runtime dependency file does not exist.

        """
        self.save_path = Path(save_path)
        self.executables_paths = executables_paths

        if compression not in ["xz", "gz", "bz2"]:
            raise ValueError(
                "Invalid compression type. Must be one of 'xz', 'gz', 'bz2'."
            )
        self.compression: Literal["xz", "gz", "bz2"] = compression

        self.archive_name = (
            archive_name
            if archive_name.endswith(f".tar.{self.compression}")
            else f"{archive_name}.tar.{self.compression}"
        )

        self.cfg = Config(  # pydantic config check
            config_version=cfg_dict["config_version"], model=cfg_dict["model"]
        )
        self.runtime_dependency_paths = dict(runtime_dependency_paths or {})
        self._validate_runtime_dependency_paths()

    def make_archive(self) -> Path:
        """Generate the NN Archive file.

        Returns:
            Path to the generated archive.

        """
        # create an in-memory file-like config object
        json_data, json_buffer = self._make_json()

        # construct .tar archive
        archive_path = self.save_path / self.archive_name
        with tarfile.open(archive_path, _WRITE_MODES[self.compression]) as tar:
            # add executables
            for executable_path in map(Path, self.executables_paths):
                tar.add(executable_path, arcname=executable_path.name)
            for (
                archive_name,
                local_path,
            ) in self.runtime_dependency_paths.items():
                tar.add(Path(local_path).resolve(), arcname=archive_name)
            # add config JSON
            tarinfo = tarfile.TarInfo(name="config.json")
            tarinfo.size = len(json_data)
            json_buffer.seek(0)  # reset the buffer to the beginning
            tar.addfile(tarinfo, json_buffer)

        return archive_path

    def _validate_runtime_dependency_paths(self) -> None:
        """Check that supplied files match the declared archive paths.

        Raises:
            ValueError: If library paths are missing or extra, or archive
                member names collide.
            FileNotFoundError: If a runtime dependency file is missing.

        """
        declared_paths: set[str] = set()
        dependencies = self.cfg.model.runtime_dependencies
        if dependencies is not None:
            for package in dependencies.snpe_udo:
                declared_paths.add(package.registration)
                declared_paths.update(
                    implementation.path
                    for implementation in package.implementations
                )
        supplied_paths = set(self.runtime_dependency_paths)
        if declared_paths != supplied_paths:
            missing = sorted(declared_paths - supplied_paths)
            extra = sorted(supplied_paths - declared_paths)
            raise ValueError(
                "Runtime dependency files must match the paths declared in "
                f"the config (missing: {missing}, extra: {extra})."
            )

        archive_paths = [Path(path).name for path in self.executables_paths]
        archive_paths.extend(self.runtime_dependency_paths)
        archive_paths.append("config.json")
        if len(archive_paths) != len(set(archive_paths)) or any(
            member.startswith(f"{parent}/")
            for parent in archive_paths
            for member in archive_paths
        ):
            raise ValueError(
                "Archive member paths must be unique and cannot be prefixes "
                "of other member paths."
            )
        for local_path in self.runtime_dependency_paths.values():
            dependency_path = Path(local_path)
            if not dependency_path.is_file():
                raise FileNotFoundError(dependency_path)

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
