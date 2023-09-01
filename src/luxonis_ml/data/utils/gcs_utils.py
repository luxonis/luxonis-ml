import os
import uuid
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import luxonis_ml.data.utils.data_utils as data_utils
from pathlib import Path


def upload_file(bucket, local_file, gcs_file):
    blob = bucket.blob(gcs_file)
    blob.upload_from_filename(local_file)


def download_file(gcs_blob, local_dir):
    try:
        local_file_path = str(Path(local_dir) / gcs_blob.name)
        if (
            os.path.exists(local_file_path)
            # TODO: check hash here
            # and get_file_hash(local_file_path) == s3_file.e_tag[1:-1]
        ):
            print(f"File {local_file_path} already exists")
        else:
            gcs_blob.download_to_filename(local_file_path)
            print(f"Downloaded {gcs_blob.name} to {local_file_path}")
    except Exception as e:
        print(f"Failed to download {gcs_blob.name}. Reason: {e}")


def copy_file(bucket, addition):
    for component_name in addition.keys():
        src_prefix = addition[component_name]["_old_filepath"]
        dst_prefix = addition[component_name]["filepath"]

        if src_prefix.startswith("gs://"):
            src_prefix = src_prefix.split(f"gs://{bucket.name}/")[1]

        if dst_prefix.startswith("/"):
            dst_prefix = dst_prefix[1:]

        blob = bucket.blob(src_prefix)
        bucket.copy_blob(blob, bucket, dst_prefix)


def get_uuid(bucket, gcp_path):
    file_contents = bucket.blob(gcp_path).download_as_bytes()
    file_hash_uuid = uuid.uuid5(uuid.NAMESPACE_URL, file_contents.hex())
    return file_hash_uuid


def update_paths(bucket, dataset, i, additions):
    addition = additions[i]
    for component_name in addition.keys():
        filepath = addition[component_name]["filepath"]
        additions[i][component_name]["_old_filepath"] = filepath
        if not data_utils.is_modified_filepath(dataset, filepath):
            prefix = filepath.split(f"gs://{dataset.bucket}/")[1]
            file_hash_uuid = get_uuid(bucket, prefix)
            # file_hash_uuid = bucket.blob(prefix).md5_hash
            hash = str(file_hash_uuid)
            hashpath = str(file_hash_uuid) + os.path.splitext(filepath)[1]
            additions[i][component_name]["instance_id"] = hash
            additions[i][component_name]["_new_image_name"] = hashpath
            granule = data_utils.get_granule(filepath, addition, component_name)
            new_filepath = f"/{dataset.team_id}/datasets/{dataset.dataset_id}/{component_name}/{granule}"
            additions[i][component_name]["filepath"] = new_filepath


def sync_to_gcs(bucket, gcs_dir, local_dir):
    print("Syncing to cloud...")
    bucket = storage.Client().bucket(bucket)

    try:
        files_to_upload = [
            os.path.join(dp, f) for dp, _, fn in os.walk(local_dir) for f in fn
        ]

        with ThreadPoolExecutor() as executor:
            for local_file_path in files_to_upload:
                gcs_dest = os.path.join(
                    gcs_dir, os.path.relpath(local_file_path, local_dir)
                )
                executor.submit(upload_file, bucket, local_file_path, gcs_dest)

    except Exception as e:
        print("Unable to upload files. Reason:", e)


def sync_from_gcs(non_streaming_dir, bucket, bucket_dir):
    os.makedirs(non_streaming_dir, exist_ok=True)

    print("Syncing from cloud...")
    bucket = storage.Client().bucket(bucket)
    try:
        with ThreadPoolExecutor(
            min(os.cpu_count(), int(os.getenv("MAX_S3_CONCURRENCY", 4)))
        ) as executor:
            for gcs_blob in bucket.list_blobs(prefix=bucket_dir):
                if not os.path.exists(
                    os.path.dirname(non_streaming_dir + "/" + gcs_blob.name)
                ):
                    os.makedirs(
                        os.path.dirname(non_streaming_dir + "/" + gcs_blob.name)
                    )
                executor.submit(download_file, gcs_blob, non_streaming_dir)
    except Exception as e:
        print("Unable to download files. Reason:", e)


def paths_from_gcs(dataset, additions):
    bucket = storage.Client().bucket(dataset.bucket)

    try:
        with ThreadPoolExecutor() as executor:
            for i in range(len(additions)):
                executor.submit(update_paths, bucket, dataset, i, additions)

    except Exception as e:
        print("GCS path update failed. Reason:", e)


def copy_to_gcs(dataset, additions):
    """Copy from one GCS bucket to another"""
    print("Copying to another bucket...")
    bucket = storage.Client().bucket(dataset.bucket)

    try:
        with ThreadPoolExecutor() as executor:
            for addition in additions:
                executor.submit(copy_file, bucket, addition)
    except Exception as e:
        print("GCS copy failed. Reason:", e)
