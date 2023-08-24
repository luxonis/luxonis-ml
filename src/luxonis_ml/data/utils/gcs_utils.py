import os
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor


def sync_from_gcs():
    raise NotImplementedError


def upload_file(bucket, local_file, gcs_file):
    blob = bucket.blob(gcs_file)
    blob.upload_from_filename(local_file)


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
