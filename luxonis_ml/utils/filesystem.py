import os.path as osp
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

import fsspec
from loguru import logger
from typeguard import typechecked

from luxonis_ml.typing import PathType, PosixPathType

from .environ import environ
from .registry import Registry


class PutFile(Protocol):
    def __call__(
        self,
        local_path: PathType,
        remote_path: PosixPathType,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> str: ...


PUT_FILE_REGISTRY: Registry[PutFile] = Registry(name="put_file")  # type: ignore


class FSType(Enum):
    MLFLOW = "mlflow"
    FSSPEC = "fsspec"


class LuxonisFileSystem:
    @typechecked
    def __init__(
        self,
        path: str,
        allow_active_mlflow_run: Optional[bool] = False,
        allow_local: Optional[bool] = True,
        cache_storage: Optional[str] = None,
        put_file_plugin: Optional[str] = None,
    ):
        """Abstraction over remote and local sources.

        Helper class which abstracts uploading and downloading files
        from remote and local sources. Supports S3, MLflow, GCS, and
        local file systems.

        @type path: str
        @param path: Input path consisting of protocol and actual path
            or just path for local files
        @type allow_active_mlflow_run: Optional[bool]
        @param allow_active_mlflow_run: Flag if operations are allowed
            on active MLFlow run. Defaults to False.
        @type allow_local: Optional[bool]
        @param allow_local: Flag if operations are allowed on local file
            system. Defaults to True.
        @type cache_storage: Optional[str]
        @param cache_storage: Path to cache storage. No cache is used if
            set to None. Defaults to None.
        @type put_file_plugin: Optional[str]
        @param put_file_plugin: The name of a registered function under
            the PUT_FILE_REGISTRY to override C{self.put_file}.
        """
        self.cache_storage = cache_storage
        self.url = path

        self.protocol, _path = _get_protocol_and_path(path)
        supported_protocols = ["s3", "gcs", "file", "mlflow"]
        if self.protocol not in supported_protocols:
            raise ValueError(
                f"Protocol '{self.protocol}://' not supported. "
                f"Choose from {supported_protocols}."
            )

        _check_package_installed(self.protocol)

        self.allow_local = allow_local
        if self.protocol == "file" and not self.allow_local:
            raise ValueError("Local filesystem is not allowed.")

        if self.protocol == "mlflow":
            self.fs_type = FSType.MLFLOW

            self.allow_active_mlflow_run = allow_active_mlflow_run
            self.is_mlflow_active_run = False
            if _path is not None:
                (self.experiment_id, self.run_id, self.artifact_path) = (
                    self._split_mlflow_path(_path)
                )
            elif _path is None and self.allow_active_mlflow_run:
                self.is_mlflow_active_run = True
                _path = ""
            else:
                raise ValueError(
                    "Using active MLFlow run is not allowed. Specify full MLFlow path."
                )
            self.tracking_uri = environ.MLFLOW_TRACKING_URI

            if self.tracking_uri is None:
                raise KeyError(
                    "There is no 'MLFLOW_TRACKING_URI' in environment variables"
                )
        else:
            self.fs_type = FSType.FSSPEC
            self.fs = self.init_fsspec_filesystem()
        self.path = PurePosixPath(cast(str, _path))

        if put_file_plugin:
            self.put_file = PUT_FILE_REGISTRY.get(put_file_plugin)

    @property
    def is_mlflow(self) -> bool:
        """Returns True if the filesystem is MLFlow.

        @type: bool
        """
        return self.fs_type == FSType.MLFLOW

    @property
    def is_fsspec(self) -> bool:
        """Returns True if the filesystem is fsspec.

        @type: bool
        """
        return self.fs_type == FSType.FSSPEC

    @property
    def full_path(self) -> str:
        """Returns full remote path.

        @type: str
        """
        return f"{self.protocol}://{self.path}"

    def init_fsspec_filesystem(self) -> fsspec.AbstractFileSystem:
        """Initializes L{fsspec} filesystem based on the used protocol.

        @rtype: L{fsspec.AbstractFileSystem}
        @return: Initialized fsspec filesystem.
        """
        if self.protocol == "s3":
            # NOTE: In theory boto3 should look in environment variables automatically but it doesn't seem to work
            fs = fsspec.filesystem(
                self.protocol,
                key=environ.AWS_ACCESS_KEY_ID,
                secret=environ.AWS_SECRET_ACCESS_KEY,
                endpoint_url=environ.AWS_S3_ENDPOINT_URL,
            )
        elif self.protocol == "gcs":
            if environ.GOOGLE_APPLICATION_CREDENTIALS is None:
                raise KeyError(
                    "There is no 'GOOGLE_APPLICATION_CREDENTIALS' in environment variables"
                )
            # NOTE: This should automatically read from GOOGLE_APPLICATION_CREDENTIALS
            fs = fsspec.filesystem(self.protocol)
        elif self.protocol == "file":
            fs = fsspec.filesystem(self.protocol)
        else:
            raise NotImplementedError
        if self.cache_storage is None:
            return fs

        if self.protocol == "file":
            logger.warning("Ignoring cache storage for local filesystem.")
            return fs

        return fsspec.filesystem(
            "filecache", fs=fs, cache_storage=self.cache_storage
        )

    def put_file(
        self,
        local_path: PathType,
        remote_path: PosixPathType,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> str:
        """Copy a single file to remote storage.

        @type local_path: PathType
        @param local_path: Path to local file
        @type remote_path: PosixPathType
        @param remote_path: Relative path to remote file
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active
            run. Defaults to C{None}.
        @rtype: str
        @return: The full remote path of the uploded file.
        """
        local_path = str(local_path)
        if self.is_mlflow:
            # NOTE: remote_path not used in mlflow since it creates new folder each time
            if self.is_mlflow_active_run:
                if mlflow_instance is not None:
                    mlflow_instance.log_artifact(local_path)
                else:
                    raise KeyError("No active mlflow_instance provided.")
            else:
                import mlflow

                client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
                client.log_artifact(run_id=self.run_id, local_path=local_path)

        elif self.is_fsspec:
            if self.protocol == "file":
                Path(self.path / remote_path).parent.mkdir(
                    parents=True, exist_ok=True
                )

            self.fs.put_file(local_path, str(self.path / remote_path))
        return self.protocol + "://" + str(self.path / remote_path)

    def put_dir(
        self,
        local_paths: Union[PathType, Iterable[PathType]],
        remote_dir: PosixPathType,
        uuid_dict: Optional[Dict[str, str]] = None,
        mlflow_instance: Optional[ModuleType] = None,
        copy_contents: bool = False,
    ) -> Optional[Dict[str, str]]:
        """Uploads files to remote storage.

        @type local_paths: Union[PathType, Sequence[PathType]]
        @param local_paths: Either a string specifying a directory to
            walk the files or a list of files which can be in different
            directories.
        @type remote_dir: PosixPathType
        @param remote_dir: Relative path to remote directory
        @type uuid_dict: Optional[Dict[str, str]]
        @param uuid_dict: Stores paths as keys and corresponding UUIDs
            as values to replace the file basename.
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active
            run. Defaults to None.
        @type copy_contents: bool
        @param copy_contents: If True, only copy the content of the
            folder specified in local_paths. Defaults to False.
        @rtype: Optional[Dict[str, str]]
        @return: When local_paths is a list, this maps local_paths to
            remote_paths
        """
        if self.is_mlflow:
            raise NotImplementedError
        if self.is_fsspec:
            if isinstance(local_paths, (Path, str)):
                local_paths = Path(local_paths)
                if not Path(local_paths).is_dir():
                    raise ValueError("Path must be a directory.")
                source_path = (
                    str(local_paths) + "/"
                    if copy_contents
                    else str(local_paths)
                )
                self.fs.put(
                    source_path,
                    str(self.path / remote_dir),
                    recursive=True,
                )
            else:
                upload_dict = {}
                with ThreadPoolExecutor() as executor:
                    for local_path in local_paths:
                        local_path = Path(local_path)
                        if uuid_dict is not None:
                            file_uuid = uuid_dict[str(local_path)]
                            ext = local_path.suffix
                            basename = file_uuid + ext
                        else:
                            basename = Path(local_path).name
                        remote_path = str(PurePosixPath(remote_dir) / basename)
                        upload_dict[str(local_path)] = remote_path
                        executor.submit(self.put_file, local_path, remote_path)
                return upload_dict

    def put_bytes(
        self,
        file_bytes: bytes,
        remote_path: PosixPathType,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> None:
        """Uploads a file to remote storage directly from file bytes.

        @type file_bytes: bytes
        @param file_bytes: the bytes for the file contents
        @type remote_path: PosixPathType
        @param remote_path: Relative path to remote file
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active
            run. Defaults to None.
        """
        if self.is_mlflow:
            raise NotImplementedError
        if self.is_fsspec:
            full_path = str(self.path / remote_path)
            with self.fs.open(full_path, "wb") as file:
                file.write(file_bytes)  # type: ignore

    def get_file(
        self,
        remote_path: PosixPathType,
        local_path: PathType,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> Path:
        """Copy a single file from remote storage.

        @type remote_path: PosixPathType
        @param remote_path: Relative path to remote file
        @type local_path: PathType
        @param local_path: Path to local file
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active
            run. Defaults to C{None}.
        @rtype: Path
        @return: Path to the downloaded file.
        """
        local_path = Path(local_path)
        if self.is_mlflow:
            raise NotImplementedError
        if self.is_fsspec:
            self.fs.get(
                str(self.path / remote_path), str(local_path), recursive=False
            )

        if local_path.is_file():
            return local_path

        return local_path / PurePosixPath(remote_path).name

    def delete_file(self, remote_path: PosixPathType) -> None:
        """Deletes a single file from remote storage.

        @type remote_path: PosixPathType
        @param remote_path: Relative path to remote file
        """
        if self.is_fsspec:
            full_remote_path = str(self.path / remote_path)
            self.fs.rm(full_remote_path)
        else:
            raise NotImplementedError

    def delete_files(self, remote_paths: List[PosixPathType]) -> None:
        """Deletes multiple files from remote storage.

        @type remote_paths: List[PosixPathType]
        @param remote_paths: Relative paths to remote files
        """
        if self.is_fsspec:
            full_remote_paths = [
                str(self.path / remote_path) for remote_path in remote_paths
            ]
            self.fs.rm(full_remote_paths)
        else:
            raise NotImplementedError

    def get_dir(
        self,
        remote_paths: Union[PosixPathType, Iterable[PosixPathType]],
        local_dir: PathType,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> Path:
        """Copies many files from remote storage to local storage.

        @type remote_paths: Union[PosixPathType,
            Sequence[PosixPathType]]
        @param remote_paths: Either a string specifying a directory to
            walk the files or a list of files which can be in different
            directories.
        @type local_dir: PathType
        @param local_dir: Path to local directory
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active
            run. Defaults to C{None}.
        @rtype: Path
        @return: Path to the downloaded directory.
        """
        local_dir = Path(local_dir)
        if self.is_mlflow:
            raise NotImplementedError
        if self.is_fsspec:
            if isinstance(remote_paths, (PurePosixPath, str)):
                existed = local_dir.exists()
                self.fs.get(
                    str(self.path / remote_paths),
                    str(local_dir),
                    recursive=True,
                )
                if not existed:
                    return local_dir
                return local_dir / PurePosixPath(remote_paths).name

            if isinstance(remote_paths, list):
                local_dir.mkdir(parents=True, exist_ok=True)
                with ThreadPoolExecutor() as executor:
                    for remote_path in remote_paths:
                        local_path = (
                            local_dir / PurePosixPath(remote_path).name
                        )
                        executor.submit(self.get_file, remote_path, local_path)

        return Path(local_dir)

    def delete_dir(
        self, remote_dir: PosixPathType = "", allow_delete_parent: bool = False
    ) -> None:
        """Deletes a directory and all its contents from remote storage.

        @type remote_dir: PosixPathType
        @param remote_dir: Relative path to remote directory.
        @type allow_delete_parent: bool
        @param allow_delete_parent: If True, allows deletion of the
            parent directory.
        """
        if not remote_dir and not allow_delete_parent:
            raise ValueError(
                f"No directory specified, this would the parent directory at `{self.path}`."
                "If this is your intention, you must pass `allow_delete_parent=True`."
            )

        if self.is_fsspec:
            full_remote_dir = str(self.path / remote_dir)
            self.fs.rm(full_remote_dir, recursive=True)
        else:
            raise NotImplementedError

    def walk_dir(
        self,
        remote_dir: PosixPathType,
        recursive: bool = True,
        typ: Literal["file", "directory", "all"] = "file",
    ) -> Iterator[str]:
        """Walks through the individual files in a remote directory.

        @type remote_dir: PosixPathType
        @param remote_dir: Relative path to remote directory
        @type recursive: bool
        @param recursive: If True, walks through the directory
            recursively.
        @type typ: Literal["file", "directory", "all"]
        @param typ: Specifies the type of files to walk through.
            Defaults to "file".
        @rtype: Iterator[str]
        @return: Iterator over the paths.
        """

        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            full_path = str(self.path / remote_dir)
            for file in self.fs.ls(full_path, detail=True):
                if self.protocol == "file":
                    name = str(
                        Path(file["name"])
                        .resolve()
                        .relative_to(Path(self.path).resolve())
                    )
                else:
                    name = str(
                        PurePosixPath(file["name"]).relative_to(self.path)
                    )
                if typ == "all" or file["type"] == typ:
                    yield name
                if recursive and file["type"] == "directory":
                    yield from self.walk_dir(name, recursive, typ)

    def read_text(self, remote_path: PosixPathType) -> Union[str, bytes]:
        """Reads a file into a string.

        @type remote_path: PosixPathType
        @param remote_path: Relative path to remote file.
        @rtype: Union[str, bytes]
        @return: The string containing the file contents.
        """
        if self.is_mlflow:
            raise NotImplementedError
        return self.fs.read_text(str(self.path / remote_path))

    def read_to_byte_buffer(
        self, remote_path: Optional[PosixPathType] = None
    ) -> BytesIO:
        """Reads a file into a byte buffer.

        @type remote_path: Optional[PosixPathType]
        @param remote_path: Relative path to remote file.
        @rtype: BytesIO
        @return: The byte buffer containing the file contents.
        """

        if self.is_mlflow:
            if self.is_mlflow_active_run:
                raise ValueError(
                    "Reading to byte buffer not available for active mlflow runs."
                )
            if self.artifact_path is None:
                raise ValueError("No relative artifact path specified.")
            import mlflow

            client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
            if self.run_id is None:
                raise RuntimeError(
                    "`run_id` cannot be `None` when using `mlflow`"
                )
            download_path = Path(
                client.download_artifacts(
                    run_id=self.run_id, path=self.artifact_path, dst_path="."
                )
            )
            with open(download_path, "rb") as f:
                buffer = BytesIO(f.read())
            download_path.unlink()

        else:
            if remote_path is not None:
                download_path = str(self.path / remote_path)
            else:
                download_path = str(self.path)
            with self.fs.open(download_path, "rb") as f:
                buffer = BytesIO(cast(bytes, f.read()))

        return buffer

    def get_file_uuid(self, path: PathType, local: bool = False) -> str:
        """Reads a file and returns the (unique) UUID generated from
        file bytes.

        @type path: PathType
        @param path: Relative path to remote file.
        @type local: bool
        @param local: Specifies a local path as opposed to a remote
            path.
        @rtype: str
        @return: The generated UUID.
        """

        if local:
            with open(path, "rb") as f:
                file_contents = cast(bytes, f.read())
        elif self.is_mlflow:
            raise NotImplementedError
        else:
            download_path = str(self.path / path)
            with self.fs.open(download_path, "rb") as f:
                file_contents = cast(bytes, f.read())

        return str(uuid.uuid5(uuid.NAMESPACE_URL, file_contents.hex()))

    def get_file_uuids(
        self, paths: Iterable[PathType], local: bool = False
    ) -> Dict[str, str]:
        """Computes the UUIDs for all files stored in the filesystem.

        @type paths: List[PathType]
        @param paths: A list of relative remote paths if remote else
            local paths.
        @type local: bool
        @param local: Specifies local paths as opposed to remote paths.
        @rtype: Dict[str, str]
        @return: A dictionary mapping the paths to their UUIDs
        """

        result = {}

        if self.is_fsspec:
            with ThreadPoolExecutor() as executor:
                for path in paths:
                    path = str(path)
                    future = executor.submit(self.get_file_uuid, path, local)
                    result[path] = future.result()

        return result

    @staticmethod
    def _split_mlflow_path(path: PathType) -> List[Optional[str]]:
        """Splits mlflow path into 3 parts."""
        path = Path(path)
        parts: List[Optional[str]] = list(path.parts)
        if len(parts) < 3:
            while len(parts) < 3:
                parts.append(None)
        elif len(parts) > 3:
            parts[2] = "/".join(cast(List[str], parts[2:]))
            parts = parts[:3]
        return parts

    def is_directory(self, remote_path: PosixPathType) -> bool:
        """Checks whether the given remote path is a directory.

        @type remote_path: PosixPathType
        @param remote_path: Relative path to remote file.
        @rtype: bool
        @return: True if the path is a directory.
        """

        full_path = str(self.path / remote_path)
        file_info = self.fs.info(full_path)
        return file_info["type"] == "directory"

    def exists(self, remote_path: PosixPathType = "") -> bool:
        """Checks whether the given remote path exists.

        @type remote_path: PosixPathType
        @param remote_path: Relative path to remote file. Defaults to ""
            (root).
        @rtype: bool
        @return: True if the path exists.
        """
        full_path = str(self.path / remote_path)
        return self.fs.exists(full_path)

    @staticmethod
    def split_full_path(path: PathType) -> Tuple[str, str]:
        """Splits the full path into protocol and absolute path.

        @type path: PathType
        @param path: Full path
        @rtype: Tuple[str, str]
        @return: Tuple of protocol and absolute path.
        """
        path = str(path).rstrip("/\\")
        return osp.split(path)

    @staticmethod
    def get_protocol(path: str) -> str:
        """Extracts the detected protocol from a path.

        @type path: str
        @param path: Full path
        @rtype: str
        @return: Protocol of the path.
        """
        return _get_protocol_and_path(path)[0]

    @staticmethod
    def download(url: str, dest: Optional[PathType]) -> Path:
        """Downloads file or directory from remote storage.

        Intended for downloading a single remote object, elevating the
        need to create an instance of L{LuxonisFileSystem}.

        @type url: str
        @param url: URL to the file or directory
        @type dest: Optional[PathType]
        @param dest: Destination directory. If unspecified, the current
            directory is used.
        @rtype: Path
        @return: Path to the downloaded file or directory.
        """

        if LuxonisFileSystem.get_protocol(url) == "file":
            return Path(url)

        dest = Path(dest or ".")
        absolute_path, remote_path = LuxonisFileSystem.split_full_path(url)
        if dest.suffix:
            local_path = dest
        else:
            local_path = dest / remote_path
        fs = LuxonisFileSystem(absolute_path)

        if fs.is_directory(remote_path):
            fs.get_dir(remote_path, local_path)
        else:
            fs.get_file(remote_path, str(local_path))

        return local_path

    @staticmethod
    def upload(local_path: PathType, url: str) -> None:
        """Uploads file or directory to remote storage.

        Intended for uploading a single local object, elevating the need
        to create an instance of L{LuxonisFileSystem}.

        @type local_path: PathType
        @param local_path: Path to the local file or directory
        @type url: str
        @param url: URL to the remote file or directory
        @rtype: str
        @return: URL to the uploaded file or directory.
        """

        absolute_path, remote_path = LuxonisFileSystem.split_full_path(url)
        fs = LuxonisFileSystem(absolute_path)
        if Path(local_path).is_dir():
            fs.put_dir(local_path, remote_path)
        else:
            fs.put_file(str(local_path), remote_path)


def _check_package_installed(protocol: str) -> None:  # pragma: no cover
    if protocol in {"gs", "gcs"} and find_spec("gcsfs") is None:
        _pip_install(protocol, "gcsfs>=2023.3.0")
    elif protocol == "s3" and find_spec("s3fs") is None:
        _pip_install(protocol, "s3fs>=2023.3.0")
    elif protocol == "mlflow" and find_spec("mlflow") is None:
        _pip_install(protocol, "mlflow~=2.10.0")


def _get_protocol_and_path(path: str) -> Tuple[str, Optional[str]]:
    """Gets the protocol and absolute path of a full path."""

    if "://" in path:
        protocol, path = path.split("://")
        if protocol == "gs":
            # ensure gs:// URLs are accepted as the gcs protocol
            protocol = "gcs"
    else:
        # assume that it is local path
        protocol = "file"

    return protocol, path if path else None


def _pip_install(protocol: str, package: str) -> None:
    logger.error(f"'{package}' is necessary for '{protocol}://' protocol.")
    logger.info(f"Installing {package}...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", package], check=True
    )
