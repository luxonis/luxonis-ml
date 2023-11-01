import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from .data_utils import generate_hashname
from typing import Optional


def download_file(
    bucket: "boto3.resources.factory.s3.Bucket", s3_file: str, local_dir: str
) -> None:
    """Helper function to download a file from S3"""

    try:
        local_file_path = str(Path(local_dir) / s3_file.key)
        if (
            os.path.exists(local_file_path)
            and generate_hashname(local_file_path) == s3_file.e_tag[1:-1]
        ):
            print(f"File {local_file_path} already exists and is up to date.")
        else:
            bucket.download_file(s3_file.key, local_file_path)
            print(f"Downloaded {s3_file.key} to {local_file_path}")
    except Exception as e:
        print(f"Failed to download {s3_file.key}. Reason: {e}")


def sync_from_s3(
    non_streaming_dir: str,
    bucket: str,
    bucket_dir: str,
    endpoint_url: Optional[str] = None,
) -> None:
    """Syncs a S3 directory of files to a local path"""

    os.makedirs(non_streaming_dir, exist_ok=True)

    print("Syncing from cloud...")
    s3 = boto3.resource("s3", endpoint_url=endpoint_url)
    bucket = s3.Bucket(bucket)
    try:
        with ThreadPoolExecutor(
            min(os.cpu_count(), int(os.getenv("MAX_S3_CONCURRENCY", 4)))
        ) as executor:
            for s3_file in bucket.objects.filter(Prefix=bucket_dir):
                if not os.path.exists(
                    os.path.dirname(non_streaming_dir + "/" + s3_file.key)
                ):
                    os.makedirs(os.path.dirname(non_streaming_dir + "/" + s3_file.key))
                executor.submit(download_file, bucket, s3_file, non_streaming_dir)

    except NoCredentialsError:
        print("No AWS credentials found")
    except Exception as e:
        print("Unable to download files. Reason:", e)


def upload_file(
    bucket: "boto3.resources.factory.s3.Bucket", local_file: str, s3_file: str
) -> None:
    """Helper function to upload a file to S3"""

    try:
        # Check if S3 object already exists
        try:
            s3_object = bucket.Object(s3_file)
            s3_object.load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":  # The object does not exist.
                s3_object = None
            else:  # Something else has gone wrong.
                raise RuntimeError(f"Error while validating the object on S3.")

        local_file_hash = generate_hashname(local_file)

        if s3_object is None or s3_object.e_tag[1:-1] != local_file_hash:
            bucket.upload_file(local_file, s3_file)
            print(f"Uploaded {local_file} to {s3_file}")
        else:
            print(f"File {s3_file} is already up to date.")
    except Exception as e:
        print(f"Failed to upload {local_file}. Reason: {e}")


def sync_to_s3(
    bucket: str, s3_dir: str, local_dir: str, endpoint_url: Optional[str] = None
) -> None:
    """Syncs a local directory of files to S3"""

    print("Syncing to cloud...")
    s3 = boto3.resource("s3", endpoint_url=endpoint_url)
    bucket = s3.Bucket(bucket)

    try:
        files_to_upload = [
            os.path.join(dp, f) for dp, _, fn in os.walk(local_dir) for f in fn
        ]

        with ThreadPoolExecutor() as executor:
            for local_file_path in files_to_upload:
                s3_dest = os.path.join(
                    s3_dir, os.path.relpath(local_file_path, local_dir)
                )
                executor.submit(upload_file, bucket, local_file_path, s3_dest)

    except NoCredentialsError:
        print("No AWS credentials found")
    except Exception as e:
        print("Unable to upload files. Reason:", e)


def check_s3_file_existence(
    bucket: str, s3_file: str, endpoint_url: Optional[str] = None
) -> None:
    """Helper function to check the existence of a file on S3"""

    s3 = boto3.resource("s3", endpoint_url=endpoint_url)
    bucket = s3.Bucket(bucket)

    try:
        s3_object = bucket.Object(s3_file)
        s3_object.load()
        return True
    except ClientError as e:
        return False
