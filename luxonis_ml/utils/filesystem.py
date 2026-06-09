import os.path as osp
import subprocess
import sys
import uuid
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Final, Literal, Protocol, cast, overload

import fsspec
from loguru import logger
from rich.progress import track
from typeguard import typechecked

from luxonis_ml.typing import PathType, PosixPathType
from luxonis_ml.utils import Registry, environ


class PutFile(Protocol):
    """Protocol for the ``put_file`` plugins."""

    def __call__(
        self,
        local_path: PathType,
        remote_path: PosixPathType,
        mlflow_instance: ModuleType | None = None,
    ) -> str: ...


PUT_FILE_REGISTRY: Final[Registry[PutFile]] = Registry(  # type: ignore
    name="put_file",
)


class FSType(Enum):
    """Enum for the type of filesystem."""

    MLFLOW = "mlflow"
    FSSPEC = "fsspec"


class LuxonisFileSystem:
    """An abstraction over remote and local sources.

    This class provides a unified interface for file operations across
    different storage backends, including local filesystems, S3, GCS,
    and MLflow artifact storage.

    For more flexibility, users can register custom implementations
    of the `put_file` method in the `PUT_FILE_REGISTRY`. It's name
    can be then passed as the ``put_file_plugin`` argument when
    initializing the `LuxonisFileSystem`. This allows for custom
    upload logic, such as additional processing before upload or
    integration with other services.

    Attributes:
        url: The original input URL.
        protocol: The detected protocol from the input URL.
        experiment_id: MLflow experiment ID when using MLflow filesystem.
        run_id: MLflow run ID when using MLflow filesystem.
        artifact_path: MLflow artifact path when using MLflow filesystem.
        tracking_uri: MLflow tracking URI when using MLflow filesystem.
        allow_active_mlflow_run: Whether operations are allowed on the active
            MLflow run.
        allow_local: Whether operations are allowed on the local file system.
        cache_storage: Path to cache storage, or ``None`` if no cache is used.
        fs_type: Type of the filesystem, either MLFLOW or FSSPEC.
        fs: Initialized fsspec filesystem when using FSSPEC.
        path: The path component of the input URL, with the protocol stripped.

    Example:
        >>> @PUT_FILE_REGISTRY.register()
        ... def put_file_plugin(*args, **kwargs) -> str:
        ...     print("Custom put_file called!")
        ...     return "remote_path"
        ...
        >>> fs = LuxonisFileSystem("s3://my-bucket/path", put_file_plugin="put_file_plugin")
        >>> remote_path = fs.put_file("local/file.txt", "remote/file.txt")
            Custom put_file called!
        >>> print(remote_path)
            remote_path
    """

    @typechecked
    def __init__(
        self,
        path: str,
        allow_active_mlflow_run: bool | None = False,
        allow_local: bool | None = True,
        cache_storage: str | None = None,
        put_file_plugin: str | None = None,
    ):
        """Initialize the ``LuxonisFileSystem``.

        Args:
            path: Input path consisting of a protocol and path, or
                only a path for local files.
            allow_active_mlflow_run: Whether operations
                are allowed on the active MLflow run.
            allow_local: Whether operations are allowed
                on the local file system.
            cache_storage: Path to cache storage. No cache is used if not set.
            put_file_plugin: Name of a registered
                function in `PUT_FILE_REGISTRY` to use instead of
                `LuxonisFileSystem.put_file`. The registered function
                must conform to the `PutFile` protocol.

        Raises:
            ValueError:
                1. If the protocol is not supported.
                2. If the protocol is ``"file"`` but local files are not allowed.
                3. If the protocol is ``"mlflow"`` but no MLflow path is
                    specified and using the active MLflow run is not allowed.
                4. If the protocol is ``"mlflow"`` but there
                    is no ``"MLFLOW_TRACKING_URI"`` in environment variables.
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
            self._is_mlflow_active_run = False
            if _path is not None:
                (self.experiment_id, self.run_id, self.artifact_path) = (
                    self._split_mlflow_path(_path)
                )
            elif _path is None and self.allow_active_mlflow_run:
                self._is_mlflow_active_run = True
                _path = ""
            else:
                raise ValueError(
                    "Using active MLFlow run is not allowed. Specify full MLFlow path."
                )
            self.tracking_uri = environ.MLFLOW_TRACKING_URI

            if self.tracking_uri is None:
                raise ValueError(
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
        """Check whether the filesystem is an MLflow filesystem.

        Returns:
            ``True`` if the filesystem is an MLflow filesystem,
        """
        return self.fs_type == FSType.MLFLOW

    @property
    def is_fsspec(self) -> bool:
        """Check whether the filesystem uses fsspec.

        Returns:
            ``True`` if the filesystem uses fsspec.
        """
        return self.fs_type == FSType.FSSPEC

    @property
    def full_path(self) -> str:
        """Full remote path.

        Returns:
            Full remote path prefixed with the protocol.
        """
        return f"{self.protocol}://{self.path}"

    def init_fsspec_filesystem(self) -> fsspec.AbstractFileSystem:
        """Initialize an ``fsspec`` filesystem for the configured
        protocol.

        Returns:
            Initialized ``fsspec`` filesystem.

        Raises:
            NotImplementedError: If the protocol is not supported by ``fsspec``.
            RuntimeError: If the credentials for the protocol are not
                properly set in environment variables.
        """
        if self.protocol == "s3":
            # NOTE: In theory boto3 should look in environment variables automatically but it doesn't seem to work
            fs = fsspec.filesystem(
                self.protocol,
                key=environ.AWS_ACCESS_KEY_ID.get_secret_value()
                if environ.AWS_ACCESS_KEY_ID is not None
                else None,
                secret=environ.AWS_SECRET_ACCESS_KEY.get_secret_value()
                if environ.AWS_SECRET_ACCESS_KEY is not None
                else None,
                endpoint_url=environ.AWS_S3_ENDPOINT_URL,
            )
        elif self.protocol == "gcs":
            if environ.GOOGLE_APPLICATION_CREDENTIALS is None:
                raise RuntimeError(
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
        mlflow_instance: ModuleType | None = None,
    ) -> str:
        """Copy a single file to remote storage.

        Args:
            local_path: Path to the local file.
            remote_path: Relative path to the remote file.
            mlflow_instance: MLflow instance to use when uploading
                to an active run.

        Returns:
            Full remote path of the uploaded file.

        Raises:
            ValueError: If using MLflow and there is no active run or
                no MLflow instance provided.
        """
        local_path = str(local_path)
        if self.is_mlflow:
            # NOTE: remote_path not used in mlflow since it creates new folder each time
            if self._is_mlflow_active_run:
                if mlflow_instance is not None:
                    mlflow_instance.log_artifact(local_path)
                else:
                    raise ValueError("No active mlflow_instance provided.")
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

    @overload
    def put_dir(
        self,
        local_paths: Iterable[PathType],
        remote_dir: PosixPathType,
        uuid_dict: dict[str, str] | None = None,
        mlflow_instance: ModuleType | None = None,
        copy_contents: bool = False,
    ) -> dict[str, str]: ...

    @overload
    def put_dir(
        self,
        local_paths: PathType,
        remote_dir: PosixPathType,
        uuid_dict: dict[str, str] | None = None,
        mlflow_instance: ModuleType | None = None,
        copy_contents: bool = False,
    ) -> None: ...

    def put_dir(
        self,
        local_paths: PathType | Iterable[PathType],
        remote_dir: PosixPathType,
        uuid_dict: dict[str, str] | None = None,
        mlflow_instance: ModuleType | None = None,
        copy_contents: bool = False,
    ) -> dict[str, str] | None:
        """Uploads files to remote storage.

        Args:
            local_paths: Either a path specifying a directory to walk,
                or an iterable of files that may be in different directories.
            remote_dir: Relative path to the remote directory.
            uuid_dict: Stores paths as keys and corresponding UUIDs
                as values to replace the file basename.
            mlflow_instance: MLflow instance to use when uploading
                to an active run.
            copy_contents: If ``True``, only copy the contents
                of the folder specified in ``local_paths``.

        Returns:
            Mapping of local paths to remote paths if
            ``local_paths`` is an iterable of files, otherwise ``None``.

        Raises:
            NotImplementedError: If using a protocol that is not
                yet supported.
            ValueError: If ``local_paths`` is a directory but
                ``copy_contents`` is ``False``.
        """
        if self.is_mlflow:
            raise NotImplementedError
        if self.is_fsspec:
            if isinstance(local_paths, PathType):
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
                futures = []
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
                        futures.append(
                            executor.submit(
                                self.put_file, local_path, remote_path
                            )
                        )
                    for _ in track(
                        as_completed(futures),
                        total=len(futures),
                        description="Uploading files",
                    ):
                        pass
                return upload_dict

    def put_bytes(
        self,
        file_bytes: bytes,
        remote_path: PosixPathType,
        mlflow_instance: ModuleType | None = None,
    ) -> None:
        """Uploads a file to remote storage directly from file bytes.

        Args:
            file_bytes: File contents to upload.
            remote_path: Relative path to the remote file.
            mlflow_instance: MLflow instance to use
                when uploading to an active run.
        Raises:
            NotImplementedError: If using a protocol that is not
                yet supported.
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
        mlflow_instance: ModuleType | None = None,
    ) -> Path:
        """Copy a single file from remote storage.

        Args:
            remote_path: Relative path to the remote file.
            local_path: Path to the local file.
            mlflow_instance: MLflow instance to use
                when downloading from an active run.

        Returns:
            Path to the downloaded file.

        Raises:
            NotImplementedError: If using a protocol that is not
                yet supported.
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

        Args:
            remote_path: Relative path to the remote file.

        Raises:
            NotImplementedError: If using a protocol that is not
                yet supported.
        """
        if self.is_fsspec:
            full_remote_path = str(self.path / remote_path)
            self.fs.rm(full_remote_path)
        else:
            raise NotImplementedError

    def delete_files(self, remote_paths: list[PosixPathType]) -> None:
        """Deletes multiple files from remote storage.

        Args:
            remote_paths: Relative paths to remote files.

        Raises:
            NotImplementedError: If using a protocol that is not
                yet supported.
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
        remote_paths: PosixPathType | Iterable[PosixPathType],
        local_dir: PathType,
        mlflow_instance: ModuleType | None = None,
    ) -> Path:
        """Copies many files from remote storage to local storage.

        Args:
            remote_paths: Either a path specifying a directory to walk,
                or an iterable of files that may be in different directories.
            local_dir: Path to the local directory.
            mlflow_instance: MLflow instance to use when downloading
                from an active run.

        Returns:
            Path to the downloaded directory.

        Raises:
            NotImplementedError: If using a protocol that is not
                yet supported.
        """
        local_dir = Path(local_dir)
        if self.is_mlflow:
            raise NotImplementedError
        if self.is_fsspec:
            if isinstance(remote_paths, PurePosixPath | str):
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

        Args:
            remote_dir: Relative path to the remote directory..
            allow_delete_parent: Whether to allow deleting the parent
                directory.

        Raises:
            ValueError: If no directory is specified and deleting the
                parent directory is not allowed.
            NotImplementedError: If using a protocol that is not
                yet supported.
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

        Args:
            remote_dir: Relative path to the remote directory.
            recursive: If True, walks through the directory
                recursively. Defaults to True.
            typ: Type of entries to yield. Corresponds to
                the ``"type"`` field in fsspec's ls output. Defaults to
                ``"file"``.

        Yields:
            Relative paths to the files in the remote directory.
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

    def read_text(self, remote_path: PosixPathType) -> str | bytes:
        """Reads a file into a string.

        Args:
            remote_path: Relative path to the remote file.

        Returns:
            The contents of the file.

        Raises:
            NotImplementedError: If using a protocol that is not
                yet supported.
        """
        if self.is_mlflow:
            raise NotImplementedError
        return self.fs.read_text(str(self.path / remote_path))

    def read_to_byte_buffer(
        self, remote_path: PosixPathType | None = None
    ) -> BytesIO:
        """Reads a file into a byte buffer.

        Args:
            remote_path: Relative path to the remote file.
                If omitted, reads from ``self.path``.

        Returns:
            Byte buffer containing the file contents.

        Raises:
            ValueError:
                1.  If using MLflow while there is already
                    an active run.
                2.  If using MLflow but no relative artifact path is
                    specified.
                3. If using MLflow but no `run_id` is specified.
        """

        if self.is_mlflow:
            if self._is_mlflow_active_run:
                raise ValueError(
                    "Reading to byte buffer not available for active mlflow runs."
                )
            if self.artifact_path is None:
                raise ValueError("No relative artifact path specified.")
            import mlflow

            client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
            if self.run_id is None:
                raise ValueError(
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

        Args:
            path: Relative path to the remote file, or a
                local path when ``local`` is ``True``.
            local: Specifies a local path as opposed
                to a remote path.

        Returns:
            UUID generated from the file bytes.

        Raises:
            NotImplementedError: If using a protocol that is not
                yet supported.
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
    ) -> dict[str, str]:
        """Computes the UUIDs for all files stored in the filesystem.

        Args:
            paths: Relative remote paths, or
                local paths when ``local`` is ``True``.
            local: Specifies local paths as opposed to
                remote paths.

        Returns:
            Dictionary mapping paths to their UUIDs.
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
    def _split_mlflow_path(path: PathType) -> list[str | None]:
        """Split an MLflow path into experiment, run, and artifact
        parts.

        Args:
            path: MLflow path to split.

        Returns:
            Three path components.
            Missing components are returned as ``None``.
        """
        path = Path(path)
        parts: list[str | None] = list(path.parts)
        if len(parts) < 3:
            while len(parts) < 3:
                parts.append(None)
        elif len(parts) > 3:
            parts[2] = "/".join(cast(list[str], parts[2:]))
            parts = parts[:3]
        return parts

    def is_directory(self, remote_path: PosixPathType) -> bool:
        """Checks whether the given remote path is a directory.

        Args:
            remote_path: Relative path to the remote path.

        Returns:
            ``True`` if the path is a directory, ``False`` otherwise.
        """

        full_path = str(self.path / remote_path)
        file_info = self.fs.info(full_path)
        return file_info["type"] == "directory"

    def exists(self, remote_path: PosixPathType = "") -> bool:
        """Checks whether the given remote path exists.

        Args:
            remote_path: Relative path to the remote file.
                Defaults to an empty string, which
                represents the root path of the filesystem.

        Returns:
            ``True`` if the path exists, ``False`` otherwise.
        """
        full_path = str(self.path / remote_path)
        return self.fs.exists(full_path)

    @staticmethod
    def split_full_path(path: PathType) -> tuple[str, str]:
        """Splits the full path into protocol and absolute path.

        Args:
            path: Full path optionally containing the protocol.

        Returns:
            The used protocol and absolute path.
        """
        path = str(path).rstrip("/\\")
        return osp.split(path)

    @staticmethod
    def get_protocol(path: str) -> str:
        """Extracts the detected protocol from a path.

        Args:
            path: Path optionally containing the protocol.

        Returns:
            Detected protocol. Defaults to ``"file"``
            if no protocol is specified.
        """
        return _get_protocol_and_path(path)[0]

    @staticmethod
    def download(url: str, dest: PathType | None) -> Path:
        """Downloads file or directory from remote storage.

        Intended for downloading a single remote object without needing
        to create a ``LuxonisFileSystem`` instance.

        Args:
            url: URL to the file or directory.
            dest: Destination directory. If ``None``,
                the current working directory is used.

        Returns:
            Path to the downloaded file or directory.
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

        Useful for uploading a single local object
        without having to create a `LuxonisFileSystem` instance.

        Args:
            local_path: Path to the local file or directory.
            url: URL to the remote file or directory.
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


def _get_protocol_and_path(path: str) -> tuple[str, str | None]:
    """Get the protocol and absolute path from a full path.

    Args:
        path: Full path optionally containing the protocol.
            If no protocol is specified, the entire path is
            treated as an absolute path and the protocol
            defaults to ``"file"``.

    Returns:
        Detected protocol and absolute path.
        The path is ``None`` when no path is present.
    """

    if "://" in path:
        protocol, path = path.split("://")
        if protocol == "gs":
            # ensure gs:// URLs are accepted as the gcs protocol
            protocol = "gcs"
    else:
        # assume that it is local path
        protocol = "file"

    return protocol, path or None


def _pip_install(protocol: str, package: str) -> None:
    logger.error(f"'{package}' is necessary for '{protocol}://' protocol.")
    logger.info(f"Installing {package}...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", package], check=True
    )
