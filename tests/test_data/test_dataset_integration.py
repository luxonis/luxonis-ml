import itertools
import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from luxonis_ml.data import (
    BucketStorage,
    DatasetIterator,
    LuxonisDataset,
    LuxonisLoader,
)
from luxonis_ml.typing import Params
from luxonis_ml.utils import LuxonisFileSystem

from .utils import gather_tasks


def get_annotations(sequence_path: Path):
    frame_data = sequence_path / "step0.frame_data.json"
    with frame_data.open() as f:
        data = json.load(f)["captures"][0]
        frame_data = data["annotations"]

    return {anno["@type"].split(".")[-1]: anno for anno in frame_data}


def test_parking_lot_generate(
    *,
    tempdir: Path,
    bucket_storage: BucketStorage,
    dataset_name: str,
    augmentation_config: list[Params],
    height: int,
    width: int,
    storage_url: str,
):
    data_path = LuxonisFileSystem.download(
        f"{storage_url}/D2_ParkingLot", tempdir
    )
    dataset = LuxonisDataset(
        dataset_name,
        delete_local=True,
        bucket_storage=bucket_storage,
        delete_remote=True,
    )
    dataset.add(generator(data_path, tempdir))
    dataset.make_splits((0.8, 0.1, 0.1))
    assert gather_tasks(dataset) == {
        "car/array",
        "car/boundingbox",
        "car/classification",
        "car/instance_segmentation",
        "car/keypoints",
        "car/metadata/brand",
        "car/metadata/color",
        "color/classification",
        "color/segmentation",
        "motorbike/array",
        "motorbike/boundingbox",
        "motorbike/classification",
        "motorbike/instance_segmentation",
        "motorbike/keypoints",
        "motorbike/metadata/brand",
        "motorbike/metadata/color",
    }
    accumulated_tasks = set()
    loader = LuxonisLoader(
        dataset,
        augmentation_config=augmentation_config,
        height=height,
        width=width,
        color_space="BGR",
        keep_aspect_ratio=True,
    )
    for _, labels in loader:
        accumulated_tasks.update(labels.keys())

    assert accumulated_tasks == gather_tasks(dataset)


def generator(base_path: Path, tempdir: Path):
    """Yield the records of 2x2 mosaics of the parking lot sequences."""
    array_paths = (tempdir / f"{i}.npy" for i in itertools.count())
    seen: set[str] = set()
    batch = []
    for sequence_path in base_path.glob("sequence.*"):
        item = _read_vehicle(sequence_path, seen)
        if item is None:
            continue
        batch.append(item)
        if len(batch) == 4:
            yield from _mosaic_records(batch, tempdir, array_paths)
            batch = []


def _read_vehicle(sequence_path: Path, seen: set[str]) -> dict | None:
    """Read the vehicle of a sequence; ``None`` for a duplicate or an odd one."""
    filepath = sequence_path / "step0.camera.jpg"
    if not filepath.exists():
        filepath = sequence_path / "step0.camera_0.jpg"

    file_hash_uuid = str(
        uuid.uuid5(uuid.NAMESPACE_URL, filepath.read_bytes().hex())
    )
    if file_hash_uuid in seen:
        return None
    seen.add(file_hash_uuid)

    annotations = get_annotations(sequence_path)
    W, H = annotations["SemanticSegmentationAnnotation"]["dimension"]
    bbox = annotations["BoundingBox2DAnnotation"]["values"][0]
    x, y = bbox["origin"]
    w, h = bbox["dimension"]
    label_name: str = bbox["labelName"]
    *brand, color, vehicle_type = label_name.split("-")
    vehicle_type = vehicle_type.lower()
    if vehicle_type == "motorbiek" or color not in ["RED", "BLUE", "GREEN"]:
        return None

    mask_path = annotations["SemanticSegmentationAnnotation"]["filename"]
    mask = cv2.imread(str(sequence_path / mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None

    return {
        "filepath": filepath,
        "W": W,
        "H": H,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "vehicle_type": vehicle_type,
        "brand": "-".join(brand),
        "color": color,
        "instance_id": bbox["instanceId"],
        "mask": mask.astype(bool),
        "keypoints": _vehicle_keypoints(annotations, vehicle_type, W, H),
    }


def _vehicle_keypoints(
    annotations: dict, vehicle_type: str, W: int, H: int
) -> list[list[float]]:
    """Normalize the keypoints of a vehicle; a motorbike has the first 3."""
    kp_annotations = annotations["KeypointAnnotation"]["values"][0][
        "keypoints"
    ]
    if vehicle_type == "motorbike":
        kp_annotations = kp_annotations[:3]
    else:
        kp_annotations = kp_annotations[3:]

    keypoints = []
    for kp in kp_annotations:
        kpt_x, kpt_y = kp["location"]
        state = 2 if vehicle_type == "motorbike" else kp["state"]
        keypoints.append([kpt_x / W, kpt_y / H, state])
    return keypoints


def _mosaic_records(
    batch: list[dict], tempdir: Path, array_paths: Iterator[Path]
) -> DatasetIterator:
    """Tile four vehicles into one image and yield its records."""
    images: list[MatLike] = [
        cv2.imread(str(item["filepath"])) for item in batch
    ]  # pyright: ignore[reportAssignmentType]
    assert all(img is not None for img in images)
    H_img, W_img, C = images[0].shape
    combined_image = np.empty((2 * H_img, 2 * W_img, C), dtype=images[0].dtype)
    color_mask = np.zeros((3, 2 * H_img, 2 * W_img), dtype=np.uint8)

    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    annotations_list = []
    for idx, item in enumerate(batch):
        i, j = positions[idx]
        combined_image[
            j * H_img : (j + 1) * H_img, i * W_img : (i + 1) * W_img, :
        ] = images[idx]
        H_mask, W_mask = item["mask"].shape
        color_mask[
            ["RED", "GREEN", "BLUE"].index(item["color"].upper()),
            j * H_mask : (j + 1) * H_mask,
            i * W_mask : (i + 1) * W_mask,
        ] = item["mask"]
        annotations_list.append(
            _quadrant_annotation(item, i, j, next(array_paths))
        )

    combined_filename = (
        "_".join([str(item["instance_id"]) for item in batch]) + ".jpg"
    )
    combined_filepath = tempdir / combined_filename
    cv2.imwrite(str(combined_filepath), combined_image)

    for annotation in annotations_list:
        yield {
            "file": combined_filepath,
            "task_name": annotation["class"],
            "annotation": annotation,
        }
    for i, color in enumerate(["red", "green", "blue"]):
        yield {
            "file": combined_filepath,
            "task_name": "color",
            "annotation": {
                "class": color,
                "segmentation": {"mask": color_mask[i]},
            },
        }


def _quadrant_annotation(item: dict, i: int, j: int, array_path: Path) -> dict:
    """Move the labels of a vehicle into quadrant ``(i, j)`` of a mosaic."""
    x_orig = item["x"] / item["W"]
    y_orig = item["y"] / item["H"]
    w_orig = item["w"] / item["W"]
    h_orig = item["h"] / item["H"]

    for kp in item["keypoints"]:
        if kp[2] != 0:
            kp[0] = (kp[0] + i) / 2
            kp[1] = (kp[1] + j) / 2

    H_mask, W_mask = item["mask"].shape
    adjusted_mask = np.zeros((2 * H_mask, 2 * W_mask), dtype=bool)
    adjusted_mask[
        j * H_mask : (j + 1) * H_mask,
        i * W_mask : (i + 1) * W_mask,
    ] = item["mask"]

    # dummy array to simulate the array field in the annotation
    np.save(array_path, np.random.rand(2, 2))

    return {
        "class": item["vehicle_type"],
        "instance_id": item["instance_id"],
        "boundingbox": {
            "x": (x_orig + i) / 2,
            "y": (y_orig + j) / 2,
            "w": w_orig / 2,
            "h": h_orig / 2,
        },
        "array": {"path": array_path},
        "keypoints": {"keypoints": item["keypoints"]},
        "instance_segmentation": {"mask": adjusted_mask},
        "metadata": {
            "color": item["color"],
            "brand": item["brand"],
        },
    }
