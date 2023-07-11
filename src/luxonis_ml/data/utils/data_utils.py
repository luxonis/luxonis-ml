import numpy as np
import hashlib
import fiftyone as fo
import os
import subprocess
import uuid


def get_granule(filepath, addition, component_name):
    granule = filepath.split("/")[-1]
    if "_new_image_name" in addition[component_name].keys():
        filepath = filepath.replace(
            granule, addition[component_name]["_new_image_name"]
        )
        granule = addition[component_name]["_new_image_name"]
    return granule


def check_media(dataset, filepath, mount_path, component_name, granule, from_bucket):
    source = dataset.source
    if dataset.bucket_type == "local":
        local_path = f"{dataset.path}/{mount_path}"
        md5 = hashlib.md5()
        with open(filepath, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                md5.update(chunk)
        old_checksum = md5.hexdigest()
    elif dataset.bucket_type == "aws":
        prefix = f"{dataset.bucket_path}/{component_name}/{granule}"
        resp = dataset.client.list_objects(
            Bucket=dataset.bucket, Prefix=prefix, Delimiter="/", MaxKeys=1
        )
        exists = "Contents" in resp
        if not exists:
            raise Exception("Issue with dataset! Sample file does not exist on S3!")
        old_checksum = resp["Contents"][0]["ETag"].strip('"')

    if from_bucket:
        if dataset.bucket_type == "aws":
            resp = dataset.client.list_objects(
                Bucket=dataset.bucket, Prefix=filepath, Delimiter="/", MaxKeys=1
            )
            exists = "Contents" in resp
            if not exists:
                raise Exception(f"File {filepath} not found on S3!")
            new_checksum = resp["Contents"][0]["ETag"].strip('"')
    else:
        md5 = hashlib.md5()
        with open(filepath, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                md5.update(chunk)
        new_checksum = md5.hexdigest()
    if new_checksum != old_checksum:
        if dataset.bucket_type == "local":
            new_path = f"{dataset.path}/{dataset.team}/datasets/{dataset.name}/archive/{source.name}/{component_name}/{granule}"
            os.makedirs(new_path, exist_ok=True)
            cmd = f"cp {local_path} {new_path}"
            subprocess.check_output(cmd, shell=True)
        elif dataset.bucket_type == "aws":
            new_prefix = f"{dataset.bucket_path}/archive/{source.name}/{component_name}/{granule}"
            dataset.client.copy_object(
                Bucket=dataset.bucket,
                Key=new_prefix,
                CopySource={"Bucket": dataset.bucket, "Key": prefix},
            )
        # update existing datasat sample to point to this new_prefix
        # limitation: all old versions point to the latest archive only (TODO: fix)
        for sample in sample_view:
            sample.filepath = mount_path.replace(
                f"/{dataset.team}/{dataset.name}/",
                f"/{dataset.team}/{dataset.name}/archive/",
            )
            sample.save()

        return False
    else:
        return True


def check_classification(val1, val2):
    if (val1 is None and val2 is not None) or (val2 is None and val1 is not None):
        return [{"boxes": val1}]
    if isinstance(val1, str):
        # prevent zip from taking letters only
        val1 = [val1]

    # TODO: Should be updated to work properly with multi-label classification
    # This includes the bottom for loop which seems to return only the first value where dict is not identical.

    if val2 is None and val1 is not None:
        val1 = fo.Classification(label=val1[0]).to_dict()
        return [{"class": val1}]

    for val1, val2 in list(zip(val1, val2["classifications"])):
        val1 = fo.Classification(label=val1).to_dict()
        if len(val1.keys()) == len(val2.keys()):
            for key in val1:
                if not key.startswith("_"):
                    if val1[key] != val2[key]:
                        return [{"class": val1}]
        else:
            return [{"class": val1}]
    return []


def check_boxes(dataset, val1, val2):
    if (val1 is None and val2 is not None) or (val2 is None and val1 is not None):
        return [{"boxes": val1}]

    if len(val1) == len(val2["detections"]):
        for val1, val2 in list(zip(val1, val2["detections"])):
            if isinstance(val1[0], str) and val2["label"] != val1[0]:
                return [{"boxes": val1}]
            if (
                not isinstance(val1[0], str)
                and val2["label"] != dataset.fo_dataset.classes["boxes"][val1[0]]
            ):
                val1[0] = dataset.fo_dataset.classes["boxes"][val1[0]]
                return [{"boxes": val1}]
            for c1, c2 in list(zip(val1[1:], val2["bounding_box"])):
                if abs(c1 - c2) > 1e-8:
                    return [{"boxes": val1}]
    else:
        return [{"boxes": val1}]
    return []


def check_segmentation(val1, val2):
    if (val1 is None and val2 is not None) or (val2 is None and val1 is not None):
        return [{"boxes": val1}]
    if (val1.shape != val2.shape) or (np.linalg.norm(val1 - val2) > 1e-8):
        return [{"segmentation": val1}]
    return []


def check_keypoints(dataset, val1, val2):
    if (val1 is None and val2 is not None) or (val2 is None and val1 is not None):
        return [{"boxes": val1}]
    if len(val1) == len(val2["keypoints"]):
        for val1, val2 in list(zip(val1, val2["keypoints"])):
            if isinstance(val1[0], str) and val2["label"] != val1[0]:
                return [{"keypoints": val1}]
            if (
                not isinstance(val1[0], str)
                and val2["label"] != dataset.fo_dataset.classes["keypoints"][val1[0]]
            ):
                val1[0] = dataset.fo_dataset.classes["keypoints"][val1[0]]
                return [{"keypoints": val1}]
            for c1, c2 in list(zip(val1[1], val2["points"])):
                if not (np.isnan(c1[0]) and isinstance(c2[0], dict)):
                    if not np.isnan(c1[0]) and not isinstance(c2[0], dict):
                        if c1[0] != c2[0] or c1[1] != c2[1]:
                            return [{"keypoints": val1}]
                    else:
                        return [{"keypoints": val1}]

    else:
        return [{"keypoints": val1}]
    return []


def check_fields(dataset, latest_sample, addition, component_name):
    changes = []

    ignore_fields_match = set(
        [
            dataset.source.name,
            "version",
            "metadata",
            "latest",
            "tags",
            "tid",
            "_group",
            "_old_filepath",
            "_new_image_name",
        ]
    )
    ignore_fields_check = set(
        ["filepath", "_group", "_old_filepath", "_new_image_name"]
    )

    sample_dict = latest_sample.to_dict()
    f1 = set(addition[component_name].keys())
    f2 = set(sample_dict.keys())
    new = f1 - f1.intersection(f2) - ignore_fields_match
    if len(new):
        for new_field in list(new):
            changes.append({new_field: addition[component_name][new_field]})

    check_fields = list(f1.intersection(f2) - ignore_fields_check)

    for field in check_fields:
        val1 = addition[component_name][field]
        val2 = sample_dict[field]

        if field in dataset.tasks:
            if field == "class":
                changes += check_classification(val1, val2)
            elif field == "boxes":
                changes += check_boxes(dataset, val1, val2)
            elif field == "segmentation":
                val2 = latest_sample.segmentation.mask
                changes += check_segmentation(val1, val2)
            elif field == "keypoints":
                changes += check_keypoints(dataset, val1, val2)
            else:
                raise NotImplementedError()
        elif val1 != val2:
            changes.append({field: val1})

    return changes


def construct_class_label(dataset, classes):
    if classes is None:
        return None
    if not isinstance(classes, list):  # fix for only one class
        classes = [classes]

    classifications = []
    for cls in classes:
        label = None
        if isinstance(cls, str):
            label = cls
        elif isinstance(cls, dict):
            label = cls.get("label")
        else:
            label = dataset.fo_dataset.classes["class"][int(cls)]

        classifications.append(fo.Classification(label=label))

    return fo.Classifications(classifications=classifications)


def construct_boxes_label(dataset, boxes):
    if boxes is None:
        return None
    if not isinstance(boxes[0], list):  # fix for only one box without a nested list
        boxes = [boxes]
    return fo.Detections(
        detections=[
            fo.Detection(
                label=box[0]
                if isinstance(box[0], str)
                else dataset.fo_dataset.classes["boxes"][int(box[0])],
                bounding_box=box[1:5],
            )
            for box in boxes
        ]
    )


def construct_segmentation_label(dataset, mask):
    if mask is None:
        return None
    if isinstance(mask, list):
        mask = np.array(mask)
    return fo.Segmentation(mask=mask)


def construct_keypoints_label(dataset, kps):
    if kps is None:
        return None
    if not isinstance(kps[0], list):  # fix for only one kp without a nested list
        kps = [kps]
    return fo.Keypoints(
        keypoints=[
            fo.Keypoint(
                label=kp[0]
                if isinstance(kp[0], str)
                else dataset.fo_dataset.classes["keypoints"][int(kp[0])],
                points=kp[1],
            )
            for kp in kps
        ]
    )


def generate_hashname(filepath):
    # Read the contents of the file
    with open(filepath, "rb") as file:
        file_contents = file.read()

    # Generate the UUID5 based on the file contents and the NAMESPACE_URL
    file_hash_uuid = uuid.uuid5(uuid.NAMESPACE_URL, file_contents.hex())

    return str(file_hash_uuid) + os.path.splitext(filepath)[1]


def is_modified_filepath(dataset, filepath):
    if filepath.startswith(f"/{dataset.team_id}"):
        return True
    else:
        return False


def get_group_from_sample(dataset, sample):
    group = sample[dataset.source.name]
    group = dataset.fo_dataset.get_group(group["id"])
    return group
