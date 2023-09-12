import numpy as np
import fiftyone as fo
import fiftyone.core.utils as fou
from fiftyone.core.labels import Classifications, Detections, Segmentation, Keypoints
import os
import uuid
from pathlib import Path
from fiftyone import ViewField as F
from luxonis_ml.data.utils.exceptions import *
from typing import Dict, List, Union, Optional


def get_granule(filepath: str, addition: Dict, component_name: str) -> str:
    """Returns the updated filename of an image and changes the name in the addition"""

    granule = filepath.split("/")[-1]
    if "_new_image_name" in addition[component_name].keys():
        filepath = filepath.replace(
            granule, addition[component_name]["_new_image_name"]
        )
        granule = addition[component_name]["_new_image_name"]
    return granule


def assert_classification_format(
    dataset: "luxonis_ml.data.LuxonisDataset", val: Union[List, str, None]
) -> None:
    """Asserts that the format for the "class" field in additions is correct"""

    if val is not None:
        if isinstance(val, str):
            if val not in dataset.fo_dataset.classes.get("class", []):
                raise ClassUnknownException(f"Class {val} is not found in dataset")
        elif isinstance(val, list):
            for v in val:
                if not isinstance(v, str):
                    raise ClassificationFormatException(
                        "All elements in list must be string"
                    )
                elif v not in dataset.fo_dataset.classes.get("class", []):
                    raise ClassUnknownException(f"Class {v} is not found in dataset")
        else:
            raise ClassificationFormatException(
                f"Classification annotation {val} must be a string or list of strings"
            )


def assert_boxes_format(
    dataset: "luxonis_ml.data.LuxonisDataset", val: Optional[List]
) -> None:
    """Asserts that the format for the "boxes" field in additions is correct"""

    if val is not None:
        if not isinstance(val, list) or not isinstance(val[0], list):
            raise BoundingBoxFormatException("Bounding boxes need to be a nested list!")

        for v in val:
            if not ((isinstance(v[0], int) or isinstance(v[0], str)) and len(v) == 5):
                raise BoundingBoxFormatException(
                    "Wrong bounding box format! It should start with int or str for the class label and contain four points"
                )

            if not isinstance(v[0], str):
                raise BoundingBoxFormatException("Classes must be strings")
            if v[0] not in dataset.fo_dataset.classes.get("boxes", []):
                raise ClassUnknownException(f"Class {v[0]} is not found in dataset")

            x, y, w, h = v[1:]
            if not (
                isinstance(x, float)
                and isinstance(y, float)
                and isinstance(w, float)
                and isinstance(h, float)
            ):
                raise BoundingBoxFormatException("Bbox x,y,w,h must be floats")
            if (
                (x < 0 or x > 1)
                or (y < 0 or y > 1)
                or (w < 0 or w > 1)
                or (h < 0 or h > 1)
            ):
                raise BoundingBoxFormatException("Bbox x,y,w,h must be between 0 and 1")
            if (x + w) > 1 or (y + h) > 1:
                raise BoundingBoxFormatException("Bbox goes outside of image")


def assert_segmentation_format(
    dataset: "luxonis_ml.data.LuxonisDataset", val: Optional[np.ndarray]
) -> None:
    """Asserts that the format for the "segmentation" field in additions is correct"""

    if val is not None:
        if not isinstance(val, np.ndarray):
            raise SegmentationFormatException(
                "Segmentation annotation must be a numpy array"
            )

        if len(val.shape) != 2:
            raise SegmentationFormatException("Array must be 2D")

        # checks for negative numbers or non-integers
        int_val = val.astype(np.uint16)
        if np.abs(np.sum(int_val - val)) > 0:
            raise SegmentationFormatException(
                "Array values change after uint16 converson"
            )


def assert_keypoints_format(
    dataset: "luxonis_ml.data.LuxonisDataset", val: Optional[List]
) -> None:
    """Asserts that the format for the "keypoints" field in additions is correct"""

    if val is not None:
        if (
            not isinstance(val, list)
            or len(val[0]) != 2
            or not isinstance(val[0][1], list)
        ):
            raise KeypointFormatException(
                "Keypoints need to be a list with the first element being the class and second being a list of points"
            )

        for kp in val:
            if not isinstance(kp[0], str):
                raise KeypointFormatException("Class must be a string")
            if kp[0] not in dataset.fo_dataset.classes.get("keypoints", []):
                raise ClassUnknownException(f"Class {kp[0]} is not found in dataset")
            for point in kp[1]:
                if len(point) != 2:
                    raise KeypointFormatException("Keypoints should be length 2 (x,y)")
                if not np.isnan(point[0]):
                    x, y = point
                    if (x < 0 or x > 1) or (y < 0 or y > 1):
                        raise KeypointFormatException(
                            "Keypoints should be in 0-1 range"
                        )


def check_classification(
    dataset: "luxonis_ml.data.LuxonisDataset",
    val1: Union[List, str, None],
    val2: Optional[Classifications],
) -> List[Dict]:
    """Returns possible changes between two "class" annotations"""

    assert_classification_format(dataset, val1)

    if (val1 is None and val2 is not None) or (val2 is None and val1 is not None):
        return [{"class": val1}]
    elif val1 is None and val2 is None:
        return []

    if isinstance(val1, str):
        val1 = [val1]

    # Only checks the classes themselves for now
    set1 = set(val1)
    set2 = set([v["label"] for v in val2["classifications"]])
    if set1 != set2:  # will handle arbitraty order for multi-class classification
        return [{"class": val1}]

    return []


def check_boxes(
    dataset: "luxonis_ml.data.LuxonisDataset",
    val1: Optional[List],
    val2: Optional[Detections],
) -> List[Dict]:
    """Returns possible changes between two "boxes" annotations"""

    assert_boxes_format(dataset, val1)

    if (val1 is None and val2 is not None) or (val2 is None and val1 is not None):
        return [{"boxes": val1}]
    elif val1 is None and val2 is None:
        return []

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


def check_segmentation(
    dataset: "luxonis_ml.data.LuxonisDataset",
    val1: Optional[np.ndarray],
    val2: Optional[Segmentation],
) -> List[Dict]:
    """Returns possible changes between two "segmentation" annotations"""

    assert_segmentation_format(dataset, val1)

    if (val1 is None and val2 is not None) or (val2 is None and val1 is not None):
        return [{"segmentation": val1}]
    elif val1 is None and val2 is None:
        return []

    if (val1.shape != val2.shape) or (np.linalg.norm(val1 - val2) > 1e-8):
        return [{"segmentation": val1}]
    return []


def check_keypoints(
    dataset: "luxonis_ml.data.LuxonisDataset",
    val1: Optional[List],
    val2: Optional[Keypoints],
) -> List[Dict]:
    """Returns possible changes between two "keypoints" annotations"""

    assert_keypoints_format(dataset, val1)

    if (val1 is None and val2 is not None) or (val2 is None and val1 is not None):
        return [{"keypoints": val1}]
    elif val1 is None and val2 is None:
        return []

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


def check_fields(
    dataset: "luxonis_ml.data.LuxonisDataset",
    latest_sample: fo.Sample,
    addition: Dict,
    component_name: str,
) -> List[Dict]:
    """Returns possible changes between arbitrary other fields"""

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
                changes += check_classification(dataset, val1, val2)
            elif field == "boxes":
                changes += check_boxes(dataset, val1, val2)
            elif field == "segmentation":
                val2 = latest_sample.segmentation.mask
                changes += check_segmentation(dataset, val1, val2)
            elif field == "keypoints":
                changes += check_keypoints(dataset, val1, val2)
            else:
                raise NotImplementedError("The CV task {field} is not implement yet")
        elif val1 != val2:
            changes.append({field: val1})

    return changes


def construct_class_label(
    dataset: "luxonis_ml.data.LuxonisDataset", classes: Union[List, str, None]
) -> Optional[Classifications]:
    """Contructs the fiftyone object for "class" annotation"""

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


def construct_boxes_label(
    dataset: "luxonis_ml.data.LuxonisDataset", boxes: Optional[List]
) -> Optional[Detections]:
    """Contructs the fiftyone object for "boxes" annotation"""

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


def construct_segmentation_label(
    dataset: "luxonis_ml.data.LuxonisDataset", mask: Optional[np.ndarray]
) -> Optional[Segmentation]:
    """Contructs the fiftyone object for "segmentation" annotation"""

    if mask is None:
        return None
    if isinstance(mask, list):
        mask = np.array(mask)
    elif isinstance(mask, bytes):
        mask = fou.deserialize_numpy_array(mask)
    mask = mask.astype(np.uint16)  # decrease size of binary
    return fo.Segmentation(mask=mask)


def construct_keypoints_label(
    dataset: "luxonis_ml.data.LuxonisDataset", kps: Optional[List]
) -> Optional[Keypoints]:
    """Contructs the fiftyone object for "keypoints" annotation"""

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


def generate_hashname(filepath: str) -> str:
    """Finds the UUID generated ID for a local file"""

    # Read the contents of the file
    with open(filepath, "rb") as file:
        file_contents = file.read()

    # TODO: check for a corrupted image by handling cv2.imread

    # Generate the UUID5 based on the file contents and the NAMESPACE_URL
    file_hash_uuid = uuid.uuid5(uuid.NAMESPACE_URL, file_contents.hex())

    return str(file_hash_uuid) + os.path.splitext(filepath)[1], str(file_hash_uuid)


def is_modified_filepath(
    dataset: "luxonis_ml.data.LuxonisDataset", filepath: str
) -> bool:
    """Returns True if the filepath for an addition is already in the modified format"""

    if filepath.startswith(f"/{dataset.team_id}"):
        return True
    else:
        return False


def get_group_from_sample(
    dataset: "luxonis_ml.data.LuxonisDataset", sample: fo.Sample
) -> fo.Group:
    """Find the fiftyone group belonging to a fiftyone sample"""

    group = sample[dataset.source.name]
    group = dataset.fo_dataset.get_group(group["id"])
    return group


def get_filepath_from_hash(
    dataset: "luxonis_ml.data.LuxonisDataset", hash: str
) -> Optional[str]:
    """Find the filepath from an instance ID"""

    instance_view = dataset.fo_dataset.match(F("instance_id") == hash)
    if len(instance_view):
        for sample in instance_view:
            break
        return sample.filepath
    else:
        return None
