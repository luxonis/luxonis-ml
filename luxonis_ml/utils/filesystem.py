import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from logging import getLogger
from types import ModuleType
from typing import Dict, Iterator, List, Optional, Tuple, Union

import fsspec

from .environ import environ

logger = getLogger(__name__)


class LuxonisFileSystem:
    def __init__(
        self,
        path: str,
        allow_active_mlflow_run: Optional[bool] = False,
        allow_local: Optional[bool] = True,
        cache_storage: Optional[str] = None,
    ):
        """Abstraction over remote and local sources.

        Helper class which abstracts uploading and downloading files from remote and
        local sources. Supports S3, MLflow, GCS, and local file systems.

        @type path: str
        @param path: Input path consisting of protocol and actual path or just path for
            local files
        @type allow_active_mlflow_run: Optional[bool]
        @param allow_active_mlflow_run: Flag if operations are allowed on active MLFlow
            run. Defaults to False.
        @type allow_local: Optional[bool]
        @param allow_local: Flag if operations are allowed on local file system.
            Defaults to True.
        @type cache_storage: Optional[str]
        @param cache_storage: Path to cache storage. No cache is used if set to None.
            Defaults to None.
        """
        if path is None:
            raise ValueError("No path provided to LuxonisFileSystem.")

        self.cache_storage = cache_storage

        self.protocol, self.path = _get_protocol_and_path(path)
        supported_protocols = ["s3", "gcs", "file", "mlflow"]
        if self.protocol not in supported_protocols:
            raise ValueError(
                f"Protocol `{self.protocol}` not supported. Choose from {supported_protocols}."
            )

        self.allow_local = allow_local
        if self.protocol == "file" and not self.allow_local:
            raise ValueError("Local filesystem is not allowed.")

        self.is_mlflow = False
        self.is_fsspec = False

        if self.protocol == "mlflow":
            self.is_mlflow = True

            self.allow_active_mlflow_run = allow_active_mlflow_run
            self.is_mlflow_active_run = False
            if len(self.path):
                (
                    self.experiment_id,
                    self.run_id,
                    self.artifact_path,
                ) = self._split_mlflow_path(self.path)
            elif len(self.path) == 0 and self.allow_active_mlflow_run:
                self.is_mlflow_active_run = True
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
            self.is_fsspec = True
            self.fs = self.init_fsspec_filesystem()

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

        return fsspec.filesystem("filecache", fs=fs, cache_storage=self.cache_storage)

    def put_file(
        self,
        local_path: str,
        remote_path: str,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> None:
        """Copy a single file to remote storage.

        @type local_path: str
        @param local_path: Path to local file
        @type remote_path: str
        @param remote_path: Relative path to remote file
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active run. Defaults to
            None.
        """
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
            self.fs.put_file(local_path, os.path.join(self.path, remote_path))

    def put_dir(
        self,
        local_paths: Union[str, List[str]],
        remote_dir: str,
        uuid_dict: Optional[Dict[str, str]] = None,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> Optional[Dict[str, str]]:
        """Uploads files to remote storage.

        @type local_paths: Union[str, List[str]]
        @param local_paths: Either a string specifying a directory to walk the files or
            a list of files which can be in different directories
        @type remote_dir: str
        @param remote_dir: Relative path to remote directory
        @type uuid_dict: Optional[Dict[str, str]]
        @param uuid_dict: Stores paths as keys and corresponding UUIDs as values to
            replace the file basename.
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active run. Defaults to
            None.
        @rtype: Optional[Dict[str, str]]
        @return: When local_paths is a list, this maps local_paths to remote_paths
        """
        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            if isinstance(local_paths, str) and os.path.isdir(local_paths):
                self.fs.put(
                    local_paths, os.path.join(self.path, remote_dir), recursive=True
                )
            else:
                upload_dict = {}
                with ThreadPoolExecutor() as executor:
                    for local_path in local_paths:
                        if uuid_dict:
                            file_uuid = uuid_dict[local_path]
                            ext = os.path.splitext(local_path)[1]
                            basename = file_uuid + ext
                        else:
                            basename = os.path.basename(local_path)
                        remote_path = os.path.join(remote_dir, basename)
                        upload_dict[local_path] = remote_path
                        executor.submit(self.put_file, local_path, remote_path)
                return upload_dict

    def put_bytes(
        self,
        file_bytes: bytes,
        remote_path: str,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> None:
        """Uploads a file to remote storage directly from file bytes.

        @type file_bytes: bytes
        @param file_bytes: the bytes for the file contents
        @type remote_path: str
        @param remote_path: Relative path to remote file
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active run. Defaults to
            None.
        """
        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            full_path = os.path.join(self.path, remote_path)
            with self.fs.open(full_path, "wb") as file:
                file.write(file_bytes)

    def get_file(
        self,
        remote_path: str,
        local_path: str,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> None:
        """Copy a single file from remote storage.

        @type remote_path: str
        @param remote_path: Relative path to remote file
        @type local_path: str
        @param local_path: Path to local file
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active run. Defaults to
            None.
        """

        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            self.fs.download(
                os.path.join(self.path, remote_path), local_path, recursive=False
            )

    def delete_file(self, remote_path: str) -> None:
        """Deletes a single file from remote storage.

        @type remote_path: str
        @param remote_path: Relative path to remote file
        """
        if self.is_fsspec:
            full_remote_path = os.path.join(self.path, remote_path)
            self.fs.rm(full_remote_path)
        else:
            raise NotImplementedError

    def delete_files(self, remote_paths: List[str]) -> None:
        """Deletes multiple files from remote storage.

        @type remote_paths: List[str]
        @param remote_paths: Relative paths to remote files
        """
        if self.is_fsspec:
            full_remote_paths = [
                os.path.join(self.path, remote_path) for remote_path in remote_paths
            ]
            self.fs.rm(full_remote_paths)
        else:
            raise NotImplementedError

    def get_dir(
        self,
        remote_dir: str,
        local_dir: str,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> None:
        """Copies many files from remote storage to local storage.

        @type remote_dir: str
        @param remote_dir: Relative path to remote directory
        @type local_dir: str
        @param local_dir: Path to local directory
        @type mlflow_instance: Optional[L{ModuleType}]
        @param mlflow_instance: MLFlow instance if uploading to active run. Defaults to
            None.
        """
        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            self.fs.download(
                os.path.join(self.path, remote_dir), local_dir, recursive=True
            )

    def delete_dir(self, remote_dir: str) -> None:
        """Deletes a directory and all its contents from remote storage.

        @type remote_dir: str
        @param remote_dir: Relative path to remote directory
        """
        if self.is_fsspec:
            full_remote_dir = os.path.join(self.path, remote_dir)
            self.fs.rm(full_remote_dir, recursive=True)
        else:
            raise NotImplementedError

    def walk_dir(self, remote_dir: str) -> Iterator[str]:
        """Recursively walks through the individual files in a remote directory.

        @type remote_dir: str
        @param remote_dir: Relative path to remote directory
        @rtype: Iterator[str]
        @return: Iterator over the paths.
        """

        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            full_path = os.path.join(self.path, remote_dir)
            for file in self.fs.glob(full_path + "/**", detail=True):
                if self.fs.info(file)["type"] == "file":
                    assert isinstance(file, str)

                    file_without_path = file.replace(self.path, "")
                    if file_without_path.startswith("/"):
                        file_without_path = file_without_path[1:]
                    yield os.path.relpath(file, self.path)

    def read_to_byte_buffer(self, remote_path: Optional[str] = None) -> BytesIO:
        """Reads a file into a byte buffer.

        @type remote_path: Optional[str]
        @param remote_path: Relative path to remote file.
        @rtype: BytesIO
        @return: The byte buffer containing the file contents.
        """

        if self.is_mlflow:
            if self.is_mlflow_active_run:
                raise ValueError(
                    "Reading to byte buffer not available for active mlflow runs."
                )
            else:
                if self.artifact_path is None:
                    raise ValueError("No relative artifact path specified.")
                import mlflow

                client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
                download_path = client.download_artifacts(
                    run_id=self.run_id, path=self.artifact_path, dst_path="."
                )
            with open(download_path, "rb") as f:
                buffer = BytesIO(f.read())
            os.remove(download_path)  # remove local file

        elif self.is_fsspec:
            if remote_path:
                download_path = os.path.join(self.path, remote_path)
            else:
                download_path = self.path
            with self.fs.open(download_path, "rb") as f:
                buffer = BytesIO(f.read())

        return buffer

    def get_file_uuid(self, path: str, local: bool = False) -> str:
        """Reads a file and returns the (unique) UUID generated from file bytes.

        @type path: str
        @param path: Relative path to remote file.
        @type local: bool
        @param local: Specifies a local path as opposed to a remote path.
        @rtype: str
        @return: The generated UUID.
        """

        if local:
            with open(path, "rb") as f:
                file_contents = f.read()
        else:
            if self.is_mlflow:
                raise NotImplementedError

            elif self.is_fsspec:
                download_path = os.path.join(self.path, path)
                with self.fs.open(download_path, "rb") as f:
                    file_contents = f.read()

        file_hash_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, file_contents.hex()))

        return file_hash_uuid

    def get_file_uuids(self, paths: List[str], local: bool = False) -> Dict[str, str]:
        """Computes the UUIDs for all files stored in the filesystem.

        @type paths: List[str]
        @param paths: A list of relative remote paths if remote else local paths.
        @type local: bool
        @param local: Specifies local paths as opposed to remote paths.
        @rtype: Dict[str, str]
        @return: A dictionary mapping the paths to their UUIDs
        """

        result = {}

        if self.is_fsspec:
            with ThreadPoolExecutor() as executor:
                for path in paths:
                    future = executor.submit(self.get_file_uuid, path, local)
                    result[path] = future.result()

        return result

    def _split_mlflow_path(self, path: str) -> List[Optional[str]]:
        """Splits mlflow path into 3 parts."""
        parts = path.split("/")
        if len(parts) < 3:
            while len(parts) < 3:
                parts.append(None)
        elif len(parts) > 3:
            parts[2] = "/".join(parts[2:])
            parts = parts[:3]
        return parts

    def is_directory(self, remote_path: str) -> bool:
        """Checks whether the given remote path is a directory.

        @type remote_path: str
        @param remote_path: Relative path to remote file.
        @rtype: bool
        @return: True if the path is a directory.
        """

        full_path = os.path.join(self.path, remote_path)
        file_info = self.fs.info(full_path)
        if file_info["type"] == "directory":
            return True
        else:
            return False

    def file_exists(self, remote_path: str) -> bool:
        """Checks whether the given remote path exists.

        @type remote_path: str
        @param remote_path: Relative path to remote file.
        @rtype: bool
        @return: True if the path exists.
        """
        full_path = os.path.join(self.path, remote_path)
        return self.fs.exists(full_path)

    @staticmethod
    def split_full_path(path: str) -> Tuple[str, str]:
        """Splits the full path into protocol and absolute path.

        @type path: str
        @param path: Full path
        @rtype: Tuple[str, str]
        @return: Tuple of protocol and absolute path.
        """
        path = path.rstrip("/\\")
        return os.path.split(path)

    @staticmethod
    def get_protocol(path: str) -> str:
        """Extracts the detected protocol from a path.

        @type path: str
        @param path: Full path
        @rtype: str
        @return: Protocol of the path.
        """
        return _get_protocol_and_path(path)[0]


def _get_protocol_and_path(path: str) -> Tuple[str, str]:
    """Gets the protocol and absolute path of a full path."""

    if "://" in path:
        protocol, path = path.split("://")
        if protocol == "gs":
            # ensure gs:// URLs are accepted as the gcs protocol
            protocol = "gcs"
    else:
        # assume that it is local path
        protocol = "file"

    return protocol, path
