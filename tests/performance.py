import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from luxonis_ml.data import *

"""
Randomly test performance of adding, updating, and creating versions

Run with LUXONISML_LEVEL=DEBUG

Run once to test ADD and a second time to test UPDATE
"""

CLASSES = ["person", "cat", "dog", "chair", "mouse", "apple", "pear", "orange"]
MAX_BOXES = 30
MAX_KEYPOINTS = 30
NUM_KEYPOINTS = 20
TEAM_ID = "a2fcd460-226e-11ee-be56-0242ac120002"
TEAM_NAME = "perf_test"
DATASET_NAME = "perf"


def _generate_random_image():
    return np.random.randint(0, 255, (400, 400, 3))


def _generate_random_class():
    num_classes = np.random.choice(range(1, len(CLASSES) + 1))
    return list(np.random.choice(CLASSES, num_classes, replace=False))


def _generate_random_boxes():
    num_boxes = np.random.choice(range(MAX_BOXES + 1))
    if num_boxes == 0:
        return None
    else:
        boxes = []
        for _ in range(num_boxes):
            lbl = np.random.choice(CLASSES)
            x = np.random.uniform(0, 0.5)
            y = np.random.uniform(0, 0.5)
            w = np.random.uniform(0, 0.5)
            h = np.random.uniform(0, 0.5)
            boxes.append([lbl, x, y, w, h])
        return boxes


def _generate_random_segmentation():
    return np.random.randint(0, len(CLASSES), (400, 400))


def _generate_random_keypoints():
    num_kp = np.random.choice(range(MAX_KEYPOINTS + 1))
    if num_kp == 0:
        return None
    else:
        keypoints = []
        for _ in range(num_kp):
            lbl = np.random.choice(CLASSES)
            points = []
            for _ in range(NUM_KEYPOINTS):
                x = np.random.uniform(0, 1.0)
                y = np.random.uniform(0, 1.0)
                points.append([x, y])
            keypoints.append([lbl, points])
        return keypoints


def main(num):
    direc = "../data/performance"
    if not os.path.exists(direc) or len(os.listdir(direc)) < args.num:
        print("Initializing media...")
        os.makedirs(direc, exist_ok=True)
        for i in tqdm(range(num)):
            img = _generate_random_image()
            cv2.imwrite(str(Path(direc) / f"{i}.jpg"), img)

    print("Generating annotations...")
    additions = []
    for i in tqdm(range(num)):
        addition = {"image": {"filepath": str(Path(direc) / f"{i}.jpg")}}
        addition["image"]["class"] = _generate_random_class()
        addition["image"]["boxes"] = _generate_random_boxes()
        addition["image"]["segmentation"] = _generate_random_segmentation()
        addition["image"]["keypoints"] = _generate_random_keypoints()
        additions.append(addition)

    dataset_id = LuxonisDataset.create(
        dataset_name=DATASET_NAME
    )

    with LuxonisDataset(DATASET_NAME) as dataset:
        dataset.create_source(
            "image",
            custom_components=[
                LDFComponent("image", HType.IMAGE, IType.BGR),
            ],
        )

        dataset.set_classes(CLASSES)

        dataset.add(additions)
        dataset.create_version(note="performance test")

    print(f"Results for {dataset_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num", type=int, help="Number of samples", required=True
    )
    args = parser.parse_args()

    main(args.num)
