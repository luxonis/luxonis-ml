import os
import uuid
from typing import Optional, Any, List, Dict, Union
from types import ModuleType
import fsspec
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor


class LuxonisFileSystem:
    def __init__(
        self,
        path: Optional[str],
        allow_active_mlflow_run: Optional[bool] = False,
        allow_local: Optional[bool] = True,
    ):
        """Helper class which abstracts uploading and downloading files from remote and local sources.
        Supports S3, MLflow and local file systems.

        Args:
            path (Optional[str]): Input path consisting of protocol and actual path or just path for local files
            allow_active_mlflow_run (Optional[bool], optional): Flag if operations are allowed on active MLFlow run. Defaults to False.
            allow_local (Optional[bool], optional): Flag if operations are allowed on local file system. Defaults to True.
        """
        if path is None:
            raise ValueError("No path provided to LuxonisFileSystem.")

        if "://" in path:
            self.protocol, self.path = path.split("://")
            supported_protocols = ["s3", "gcs", "file", "mlflow"]
            if self.protocol not in supported_protocols:
                raise KeyError(
                    f"Protocol `{self.protocol}` not supported. Choose from {supported_protocols}."
                )
        else:
            # assume that it is local path
            self.protocol = "file"
            self.path = path

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
            self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

            if self.tracking_uri is None:
                raise KeyError(
                    "There is no 'MLFLOW_TRACKING_URI' in environment variables"
                )
        else:
            self.is_fsspec = True
            self.fs = self.init_fsspec_filesystem()

    def full_path(self) -> str:
        """Returns full path"""
        return f"{self.protocol}://{self.path}"

    def init_fsspec_filesystem(self) -> Any:
        """Returns fsspec filesystem based on protocol"""
        if self.protocol == "s3":
            # NOTE: In theory boto3 should look in environment variables automatically but it doesn't seem to work
            return fsspec.filesystem(
                self.protocol,
                key=os.getenv("AWS_ACCESS_KEY_ID"),
                secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
                endpoint_url=os.getenv("AWS_S3_ENDPOINT_URL"),
            )
        elif self.protocol == "gcs":
            # NOTE: This should automatically read from GOOGLE_APPLICATION_CREDENTIALS
            return fsspec.filesystem(self.protocol)
        elif self.protocol == "file":
            return fsspec.filesystem(self.protocol)
        else:
            raise NotImplementedError

    def put_file(
        self,
        local_path: str,
        remote_path: str,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> None:
        """Copy single file to remote

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
    ) -> None:
        """Copies many files from local storage to remote storage in parallel.

        Args:
            local_paths (Union[str, List[str]]): Either a string specifying a directory to walk the files or a list of files
                which can be in different directories
            remote_dir (str): Relative path to remote directory
            uuid_dict (Optional[Dict[str, str]]): Stores paths as keys and corresponding UUIDs as values to replace the file basename.
            mlflow_instance (Optional[ModuleType], optional): MLFlow instance if uploading to active run. Defaults to None.
        """
        if self.is_mlflow:
            raise NotImplementedError
        elif self.is_fsspec:
            if isinstance(local_paths, str) and os.path.isdir(local_paths):
                self.fs.put(
                    local_paths, os.path.join(self.path, remote_dir), recursive=True
                )
            else:
                with ThreadPoolExecutor() as executor:
                    for local_path in local_paths:
                        if uuid_dict:
                            basename = uuid_dict[local_path]
                        else:
                            basename = os.path.basename(local_path)
                        remote_path = os.path.join(remote_dir, basename)
                        executor.submit(self.put_file, local_path, remote_path)

    def get_file(
        self,
        remote_path: str,
        local_path: str,
        mlflow_instance: Optional[ModuleType] = None,
    ) -> None:
        """Copy a single file from remote

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

    def read_to_byte_buffer(self, remote_path: Optional[str] = None) -> BytesIO:
        """Reads a file and returns Byte buffer

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
        """Reads a file and returns the (unique) UUID generated from file bytes

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
        """Splits mlflow path into 3 parts"""
        parts = path.split("/")
        if len(parts) < 3:
            while len(parts) < 3:
                parts.append(None)
        elif len(parts) > 3:
            parts[2] = "/".join(parts[2:])
            parts = parts[:3]
        return parts