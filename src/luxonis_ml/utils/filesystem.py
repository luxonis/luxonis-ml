import os
import uuid
from typing import Optional, Any, List, Dict, Union, Tuple, Generator
from types import ModuleType
import fsspec
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from .environ import environ
from logging import getLogger

logger = getLogger(__name__)


class LuxonisFileSystem:
    def __init__(
        self,
        path: str,
        allow_active_mlflow_run: Optional[bool] = False,
        allow_local: Optional[bool] = True,
        cache_storage: Optional[str] = None,
    ):
        """Helper class which abstracts uploading and downloading files from remote and
        local sources. Supports S3, MLflow and local file systems.

        Args:
            path (str): Input path consisting of protocol and actual path or just path for local files
            allow_active_mlflow_run (Optional[bool], optional): Flag if operations are allowed on active MLFlow run. Defaults to False.
            allow_local (Optional[bool], optional): Flag if operations are
                allowed on local file system. Defaults to True.
            cache_storage (Optional[str], optional): Path to cache storage. No cache
                is used if set to None. Defaults to None.
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

    def full_path(self) -> str:
        """Returns full path."""
        return f"{self.protocol}://{self.path}"

    def init_fsspec_filesystem(self) -> Any:
        """Returns fsspec filesystem based on protocol."""
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
        """Copy single file to remote.

        Args:
            local_path (str): Path to local file
            remote_path (str): Relative path to remote file
            mlflow_instance (Optional[ModuleType], optional): MLFlow instance if uploading to active run. Defaults to None.
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
        """Copies many files from local storage to remote storage in parallel.

        Args:
            local_paths (Union[str, List[str]]): Either a string specifying a directory to walk the files or a list of files
                which can be in different directories
            remote_dir (str): Relative path to remote directory
            uuid_dict (Optional[Dict[str, str]]): Stores paths as keys and corresponding UUIDs as values to replace the file basename.
            mlflow_instance (Optional[ModuleType], optional): MLFlow instance if uploading to active run. Defaults to None.
        Returns:
            upload_dict (Optional[Dict[str, str]]): When local_paths is a list, this maps local_paths to remote_paths
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

        Args:
            file_bytes (bytes): the bytes for the file contents
            remote_path (str): Relative path to remote file
            mlflow_instance (Optional[ModuleType], optional): MLFlow instance if uploading to active run. Defaults to None.
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
        """Copy a single file from remote.

        Args:
            remote_path (str): Relative path to remote file
            local_path (str): Path to local file
            mlflow_instance (Optional[ModuleType], optional): MLFlow instance if uploading to active run. Defaults to None.
        """

        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            self.fs.download(
                os.path.join(self.path, remote_path), local_path, recursive=False
            )
    
    def delete_file(self, remote_path: str) -> None:
        """Deletes a single file from remote storage.

        Args:
            remote_path (str): Relative path to the remote file to be deleted.
        """
        if self.is_fsspec:
            full_remote_path = os.path.join(self.path, remote_path)
            self.fs.rm(full_remote_path)
        else:
            raise NotImplementedError
    
    def delete_files(self, remote_paths: List[str]) -> None:
        """Deletes multiple files from remote storage.

        Args:
            remote_paths (List[str]): Relative paths to the remote files to be deleted.
        """
        if self.is_fsspec:
            full_remote_paths = [os.path.join(self.path, remote_path) for remote_path in remote_paths]
            self.fs.rm(full_remote_paths)
        else:
            raise NotImplementedError

    def get_dir(
        self,
        remote_dir: str,
        local_dir: str,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> None:
        """Copies many files from remote storage to local storage in parallel.

        Args:
            remote_dir (str): Relative path to remote directory
            local_dir (str): Path to the local directory
            mlflow_instance (Optional[ModuleType], optional): MLFlow instance if uploading to active run. Defaults to None.
        """
        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            self.fs.download(
                os.path.join(self.path, remote_dir), local_dir, recursive=True
            )
    
    def delete_dir(self, remote_dir: str) -> None:
        """Deletes a directory and all its contents from remote storage.

        Args:
            remote_dir (str): Relative path to the remote directory to be deleted.
        """
        if self.is_fsspec:
            full_remote_dir = os.path.join(self.path, remote_dir)
            self.fs.rm(full_remote_dir, recursive=True)
        else:
            raise NotImplementedError

    def walk_dir(self, remote_dir: str) -> Generator[str, None, None]:
        """Recursively walks through the individual files in a remote directory."""

        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            full_path = os.path.join(self.path, remote_dir)
            for file in self.fs.glob(full_path + "/**", detail=True):
                if self.fs.info(file)["type"] == "file":
                    file_without_path = file.replace(self.path, "")
                    if file_without_path.startswith("/"):
                        file_without_path = file_without_path[1:]
                    yield os.path.relpath(file, self.path)

    def read_to_byte_buffer(self, remote_path: Optional[str] = None) -> BytesIO:
        """Reads a file and returns Byte buffer.

        Args:
            remote_path (Optional[str]): If provided, the relative path to the remote file
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

        Args:
            path (str): If remote, relative path to the remote file. Else the local path
            local (bool): Specifies a local path as opposed to a remote path
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
        """Computes the UUID for files stored in the filesystem.

        Args:
            paths (List[str]): A list of relative remote paths if remote else local paths
            local (bool): Specifies local paths as opposed to remote paths
        Returns: A dictionary mapping the paths to their UUIDs
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
        """Returns True if a remote path points to a directory."""

        full_path = os.path.join(self.path, remote_path)
        file_info = self.fs.info(full_path)
        if file_info["type"] == "directory":
            return True
        else:
            return False

    def file_exists(self, remote_path: str) -> bool:
        """Returns True if there is a file at the given remote path"""
        full_path = os.path.join(self.path, remote_path)
        return self.fs.exists(full_path)

    @staticmethod
    def split_full_path(path: str) -> Tuple[str, str]:
        """Returns a tuple for the absolute and relative path given a full path."""

        path = path.rstrip("/\\")
        return os.path.split(path)

    @staticmethod
    def get_protocol(path: str) -> str:
        """Gets the detected protocol of a path."""

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
