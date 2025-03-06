import json
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np

from luxonis_ml.data import BucketStorage, LuxonisDataset, LuxonisLoader
from luxonis_ml.typing import Params
from luxonis_ml.utils import LuxonisFileSystem

from .utils import gather_tasks


def get_annotations(sequence_path: Path):
    frame_data = sequence_path / "step0.frame_data.json"
    with open(frame_data) as f:
        data = json.load(f)["captures"][0]
        frame_data = data["annotations"]

    return {anno["@type"].split(".")[-1]: anno for anno in frame_data}


def test_parking_lot_generate(
    tempdir: Path,
    bucket_storage: BucketStorage,
    dataset_name: str,
    augmentation_config: List[Params],
    height: int,
    width: int,
    storage_url: str,
):
    data_path = LuxonisFileSystem.download(
        f"{storage_url}/D2_ParkingLot", tempdir
    )
    dataset = LuxonisDataset(
        dataset_name,
        delete_existing=True,
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


# TODO: Simplify the dataset so the code can be cleaner
def generator(base_path: Path, tempdir: Path):
    array_counter = 0
    seen = set()
    batch = []
    for sequence_path in base_path.glob("sequence.*"):
        filepath = sequence_path / "step0.camera.jpg"
        if not filepath.exists():
            filepath = sequence_path / "step0.camera_0.jpg"

        file_hash_uuid = str(
            uuid.uuid5(uuid.NAMESPACE_URL, filepath.read_bytes().hex())
        )
        if file_hash_uuid in seen:
            continue
        seen.add(file_hash_uuid)
        annotations = get_annotations(sequence_path)
        W, H = annotations["SemanticSegmentationAnnotation"]["dimension"]
        bbox = annotations["BoundingBox2DAnnotation"]["values"][0]
        instance_id = bbox["instanceId"]
        x, y = bbox["origin"]
        w, h = bbox["dimension"]
        label_name: str = bbox["labelName"]
        *brand, color, vehicle_type = label_name.split("-")
        brand = "-".join(brand)
        vehicle_type = vehicle_type.lower()
        if vehicle_type == "motorbiek" or color not in [
            "RED",
            "BLUE",
            "GREEN",
        ]:
            continue

        keypoints = []
        kp_annotations = annotations["KeypointAnnotation"]["values"][0][
            "keypoints"
        ]
        if vehicle_type == "motorbike":
            kp_annotations = kp_annotations[:3]
        else:
            kp_annotations = kp_annotations[3:]

        for kp in kp_annotations:
            kpt_x, kpt_y = kp["location"]
            state = kp["state"]
            if vehicle_type == "motorbike":
                state = 2
            keypoints.append([kpt_x / W, kpt_y / H, state])

        mask_path = annotations["SemanticSegmentationAnnotation"]["filename"]
        mask = cv2.imread(
            str(sequence_path / mask_path), cv2.IMREAD_GRAYSCALE
        ).astype(bool)

        batch.append(
            {
                "filepath": filepath,
                "W": W,
                "H": H,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "vehicle_type": vehicle_type,
                "brand": brand,
                "color": color,
                "instance_id": instance_id,
                "mask": mask,
                "keypoints": keypoints,
            }
        )

        if len(batch) == 4:
            images = [cv2.imread(str(item["filepath"])) for item in batch]
            H_img, W_img, C = images[0].shape
            combined_image = np.empty(
                (2 * H_img, 2 * W_img, C), dtype=images[0].dtype
            )
            color_mask = np.zeros((3, 2 * H_img, 2 * W_img), dtype=np.uint8)

            positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
            annotations_list = []

            for idx, item in enumerate(batch):
                i, j = positions[idx]
                combined_image[
                    j * H_img : (j + 1) * H_img, i * W_img : (i + 1) * W_img, :
                ] = images[idx]

                x_orig = item["x"] / item["W"]
                y_orig = item["y"] / item["H"]
                w_orig = item["w"] / item["W"]
                h_orig = item["h"] / item["H"]

                x_new = (x_orig + i) / 2
                y_new = (y_orig + j) / 2
                w_new = w_orig / 2
                h_new = h_orig / 2

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
                color_mask[
                    ["RED", "GREEN", "BLUE"].index(item["color"].upper()),
                    j * H_mask : (j + 1) * H_mask,
                    i * W_mask : (i + 1) * W_mask,
                ] = item["mask"]

                # dummy array to simulate the array field in the annotation
                array = np.random.rand(2, 2)
                array_path = tempdir / f"{array_counter}.npy"
                np.save(array_path, array)
                array_counter += 1

                adjusted_annotation = {
                    "class": item["vehicle_type"],
                    "instance_id": item["instance_id"],
                    "boundingbox": {
                        "x": x_new,
                        "y": y_new,
                        "w": w_new,
                        "h": h_new,
                    },
                    "array": {"path": array_path},
                    "keypoints": {"keypoints": item["keypoints"]},
                    "instance_segmentation": {"mask": adjusted_mask},
                    "metadata": {
                        "color": item["color"],
                        "brand": item["brand"],
                    },
                }
                annotations_list.append(adjusted_annotation)

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

            batch = []
