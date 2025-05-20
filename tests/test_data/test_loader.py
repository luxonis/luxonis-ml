import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from luxonis_ml.data import (
    BucketStorage,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
)
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.enums import DatasetType
from luxonis_ml.typing import Params
from luxonis_ml.utils import LuxonisFileSystem

from .utils import create_image

AUGMENTATIONS_CONFIG: list[Params] = [  # type: ignore[reportAssignmentType]
    {
        "name": "Mosaic4",
        "params": {"p": 1, "out_width": 512, "out_height": 512},
    },
    {
        "name": "Rotate",
        "params": {
            "limit": 10,
            "p": 0.3,
            "border_mode": 0,
            "value": [0, 0, 0],
        },
    },
    {
        "name": "Perspective",
        "params": {
            "scale": [0.02, 0.05],
            "keep_size": True,
            "pad_mode": 0,
            "pad_val": 0,
            "mask_pad_val": 0,
            "fit_output": False,
            "interpolation": 1,
            "always_apply": False,
            "p": 0.3,
        },
    },
    {
        "name": "Affine",
        "params": {
            "scale": 1.0,
            "translate_percent": 0.0,
            "rotate": 0,
            "shear": 5,
            "interpolation": 1,
            "mask_interpolation": 0,
            "cval": 0,
            "cval_mask": 0,
            "mode": 0,
            "fit_output": False,
            "keep_ratio": False,
            "rotate_method": "largest_box",
            "always_apply": False,
            "p": 0.3,
        },
    },
    {
        "name": "MixUp",
        "params": {
            "p": 0.3,
        },
    },
    {
        "name": "VerticalSymetricKeypointsFlip",
        "params": {
            "p": 0.3,
            "keypoint_pairs": [
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (11, 11),
                (12, 12),
                (13, 13),
                (14, 14),
                (15, 15),
                (16, 16),
            ],
        },
    },
    {
        "name": "HorizontalSymetricKeypointsFlip",
        "params": {
            "p": 0.4,
            "keypoint_pairs": [
                (0, 0),
                (1, 2),
                (3, 4),
                (5, 6),
                (7, 8),
                (9, 10),
                (11, 12),
                (13, 14),
                (15, 16),
            ],
        },
    },
]


@contextmanager
def set_seed(seed: int):
    np_state = np.random.get_state()
    random_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    yield
    np.random.set_state(np_state)
    random.setstate(random_state)


def create_loader(
    storage_url: str,
    tempdir: Path,
    **kwargs,
) -> LuxonisLoader:
    with set_seed(42):
        dataset = LuxonisParser(
            f"{storage_url}/COCO_people_subset.zip",
            dataset_name="_augmentation_reproducibility",
            save_dir=tempdir,
            dataset_type=DatasetType.COCO,
            delete_local=True,
        ).parse()
    return LuxonisLoader(
        dataset,
        height=512,
        width=512,
        view="train",
        seed=42,
        **kwargs,
    )


def load_annotations(annotation_name: str) -> list[dict[str, Any]]:
    dest_dir = Path("./tests/data/")
    local_path = dest_dir / annotation_name

    if not local_path.exists():
        remote_path = f"gs://luxonis-test-bucket/luxonis-ml-test-data/test-augmentation-data/{annotation_name}"
        local_path = LuxonisFileSystem.download(remote_path, dest=dest_dir)

    with open(local_path) as f:
        return json.load(f)


def test_edge_cases(tempdir: Path):
    keypoints_before = np.array(
        [
            [
                0.41100962,
                0.63641827,
                2.0,
                0.49552885,
                0.61091346,
                2.0,
                0.50201923,
                0.63555288,
                2.0,
                0.41879808,
                0.66302885,
                2.0,
            ],
            [
                0.86091346,
                0.76336538,
                2.0,
                0.99495192,
                0.71382212,
                2.0,
                0.97978365,
                0.75096154,
                0.0,
                0.87086538,
                0.7909375,
                2.0,
            ],
            [
                0.77483173,
                0.23288462,
                2.0,
                0.90862981,
                0.16939904,
                2.0,
                0.91990385,
                0.19620192,
                2.0,
                0.78622596,
                0.26213942,
                2.0,
            ],
            [
                0.07838942,
                0.40788462,
                2.0,
                0.12990385,
                0.59675481,
                2.0,
                0.10711538,
                0.60266827,
                2.0,
                0.05271635,
                0.41540865,
                2.0,
            ],
            [
                0.03276442,
                0.38091346,
                2.0,
                0.08644231,
                0.5690625,
                2.0,
                0.06149038,
                0.57569712,
                2.0,
                0.00709135,
                0.3884375,
                2.0,
            ],
            [
                0.0,
                0.39269231,
                0.0,
                0.01310096,
                0.5846875,
                2.0,
                0.0,
                0.58747596,
                0.0,
                0.0,
                0.40019231,
                0.0,
            ],
            [
                0.06036058,
                0.23735577,
                2.0,
                0.0,
                0.25528846,
                0.0,
                0.0,
                0.23185096,
                0.0,
                0.05483173,
                0.21627404,
                2.0,
            ],
            [
                0.09098558,
                0.72336538,
                2.0,
                0.14283654,
                0.90288462,
                2.0,
                0.12247596,
                0.91052885,
                2.0,
                0.06740385,
                0.72949519,
                2.0,
            ],
            [
                0.04668269,
                0.18875,
                2.0,
                0.0,
                0.20668269,
                0.0,
                0.0,
                0.18322115,
                0.0,
                0.04115385,
                0.16766827,
                2.0,
            ],
            [
                0.02704327,
                0.14060096,
                2.0,
                0.0,
                0.15853365,
                0.0,
                0.0,
                0.13507212,
                0.0,
                0.02149038,
                0.11951923,
                0.0,
            ],
            [
                0.41819712,
                0.90716346,
                2.0,
                0.45608173,
                0.89471154,
                2.0,
                0.47170673,
                0.94043269,
                2.0,
                0.43572115,
                0.95435096,
                2.0,
            ],
        ]
    )

    bboxes_before = np.array(
        [
            [0.0, 0.41100962, 0.61091346, 0.09100962, 0.05211538],
            [0.0, 0.86091346, 0.71382212, 0.13403846, 0.07711538],
            [0.0, 0.77483173, 0.16939904, 0.14507212, 0.09274038],
            [0.0, 0.05271635, 0.40788462, 0.0771875, 0.19478365],
            [0.0, 0.00709135, 0.38091346, 0.07935096, 0.19478365],
            [0.0, 0.0, 0.39269231, 0.01310096, 0.19478365],
            [0.0, 0.0, 0.21627404, 0.06036058, 0.03901442],
            [0.0, 0.06740385, 0.72336538, 0.07543269, 0.18716346],
            [0.0, 0.0, 0.16766827, 0.04668269, 0.03901442],
            [0.0, 0.0, 0.11951923, 0.02704327, 0.03901442],
            [0.0, 0.41819712, 0.89471154, 0.05350962, 0.05963942],
        ]
    )

    def generator() -> DatasetIterator:
        num_samples = len(keypoints_before)
        for i in range(num_samples):
            img = create_image(i, tempdir)
            keypoints_list = keypoints_before[i].tolist()
            bbox_list = bboxes_before[i].tolist()

            keypoints_split = [
                [x, y, int(visibility)]
                for x, y, visibility in (
                    keypoints_list[j : j + 3]
                    for j in range(0, len(keypoints_list), 3)
                )
            ]

            yield {
                "file": img,
                "annotation": {
                    "class": "dog",
                    "boundingbox": {
                        "x": bbox_list[1],
                        "y": bbox_list[2],
                        "w": bbox_list[3],
                        "h": bbox_list[4],
                    },
                    "keypoints": {"keypoints": keypoints_split},
                },
            }

    dataset = LuxonisDataset(
        "test_edge_cases",
        delete_local=True,
        delete_remote=True,
        bucket_storage=BucketStorage.LOCAL,
    ).add(generator())

    dataset.make_splits(ratios=(1, 0, 0))

    augmentation_config: list[Params] = [
        {
            "name": "Mosaic4",
            "params": {"p": 1, "out_width": 512, "out_height": 512},
        },
        {
            "name": "Rotate",
            "params": {
                "limit": 10,
                "p": 1,
                "border_mode": 0,
                "value": [0, 0, 0],
            },
        },
        {
            "name": "Perspective",
            "params": {
                "scale": [0.02, 0.05],
                "keep_size": True,
                "pad_mode": 0,
                "pad_val": 0,
                "mask_pad_val": 0,
                "fit_output": False,
                "interpolation": 1,
                "always_apply": False,
                "p": 1,
            },
        },
        {
            "name": "Affine",
            "params": {
                "scale": 1.0,
                "translate_percent": 0.0,
                "rotate": 0,
                "shear": 5,
                "interpolation": 1,
                "mask_interpolation": 0,
                "cval": 0,
                "cval_mask": 0,
                "mode": 0,
                "fit_output": False,
                "keep_ratio": False,
                "rotate_method": "largest_box",
                "always_apply": False,
                "p": 1,
            },
        },
    ]

    loader = LuxonisLoader(
        dataset,
        view="train",
        augmentation_config=augmentation_config,
        height=512,
        width=512,
    )
    for _ in range(20):
        for batch in loader:
            keypoints = batch[1]["/keypoints"]
            bboxes = batch[1]["/boundingbox"]
            keypoints = keypoints.reshape(-1, 3)
            for kp in keypoints:
                x, y, v = kp
                assert not (x == 0 and y == 0 and v in {1, 2}), (
                    f"Invalid keypoint detected: {kp}"
                )
                assert 0 <= x <= 1, f"Keypoint x out of bounds: {kp}"
                assert 0 <= y <= 1, f"Keypoint y out of bounds: {kp}"

            for bbox in bboxes:
                _, x_min, y_min, width, height = bbox

                x_max = x_min + width
                y_max = y_min + height

                assert 0 <= x_min <= 1, f"BBox x_min out of bounds: {bbox}"
                assert 0 <= y_min <= 1, f"BBox y_min out of bounds: {bbox}"
                assert 0 <= x_max <= 1, f"BBox x_max out of bounds: {bbox}"
                assert 0 <= y_max <= 1, f"BBox y_max out of bounds: {bbox}"

                bbox_area = width * height
                assert bbox_area >= 0.0004, (
                    f"BBox area too small: {bbox}, area={bbox_area}"
                )


def test_dataset_reproducibility(storage_url: str, tempdir: Path):
    loader1 = create_loader(
        storage_url, tempdir, augmentation_config=AUGMENTATIONS_CONFIG
    )
    run1 = [ann for _, ann in loader1]

    loader2 = create_loader(
        storage_url, tempdir, augmentation_config=AUGMENTATIONS_CONFIG
    )
    run2 = [ann for _, ann in loader2]

    assert all(
        a1.keys() == a2.keys()
        and all(
            np.array_equal(a1[k], a2[k])
            if isinstance(a1[k], np.ndarray)
            else a1[k] == a2[k]
            for k in a1
        )
        for a1, a2 in zip(run1, run2, strict=True)
    )


def test_augmentation_reproducibility(storage_url: str, tempdir: Path):
    def mask_to_rle(mask: np.ndarray) -> list[int]:
        pixels = mask.flatten()
        rle = []
        prev_pixel = pixels[0]
        count = 0

        for pixel in pixels:
            if pixel == prev_pixel:
                count += 1
            else:
                rle.append(count)
                count = 1
                prev_pixel = pixel
        rle.append(count)

        return rle

    def rle_to_mask(counts: list[int], height: int, width: int) -> np.ndarray:
        flat = np.repeat(np.arange(len(counts)) % 2, counts)
        return flat.reshape((height, width), order="F").astype(bool)

    def convert_annotation(ann: dict[str, Any]) -> dict[str, Any]:
        def round_nested_list(
            data: list[Any] | float, decimals: int = 3
        ) -> list[Any] | float:
            if isinstance(data, list):
                return [round_nested_list(elem, decimals) for elem in data]
            if isinstance(data, float):
                return round(data, decimals)
            return data

        return {
            "classification": round_nested_list(
                ann["/classification"].tolist()
                if isinstance(ann["/classification"], np.ndarray)
                else ann["/classification"]
            ),
            "bounding_box": round_nested_list(
                ann["/boundingbox"].tolist()
                if isinstance(ann["/boundingbox"], np.ndarray)
                else ann["/boundingbox"]
            ),
            "segmentation": mask_to_rle(ann["/segmentation"])
            if isinstance(ann["/segmentation"], np.ndarray)
            else ann["/segmentation"],
            "keypoints": round_nested_list(
                ann["/keypoints"].tolist()
                if isinstance(ann["/keypoints"], np.ndarray)
                else ann["/keypoints"]
            ),
        }

    loader_aug = create_loader(
        storage_url, tempdir, augmentation_config=AUGMENTATIONS_CONFIG
    )
    new_aug_annotations = [convert_annotation(ann) for _, ann in loader_aug]

    original_aug_annotations = load_annotations(
        "test_augmentation_reproducibility_labels.json"
    )

    for orig_ann, new_ann in zip(
        original_aug_annotations, new_aug_annotations, strict=True
    ):
        assert orig_ann["classification"] == new_ann["classification"]
        assert orig_ann["bounding_box"] == new_ann["bounding_box"]
        assert orig_ann["keypoints"] == new_ann["keypoints"]
        orig_mask = rle_to_mask(orig_ann["segmentation"], 512, 512)
        new_mask = rle_to_mask(new_ann["segmentation"], 512, 512)
        diff = np.count_nonzero(orig_mask != new_mask)
        assert diff <= 50


def test_colorspace(storage_url: str, tempdir: Path):
    norm_3d = [
        {
            "name": "Normalize",
            "params": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "p": 1,
            },
        },
    ]
    norm_1d = [
        {
            "name": "Normalize",
            "params": {
                "mean": [0.5],
                "std": [0.5],
                "p": 1,
            },
        },
    ]
    loader = create_loader(storage_url, tempdir, augmentation_config=norm_3d)
    rgb_img, _ = next(iter(loader))
    assert len(rgb_img.shape) == 3
    assert rgb_img.shape[2] == 3
    loader = create_loader(
        storage_url, tempdir, color_space="BGR", augmentation_config=norm_3d
    )
    bgr_img, _ = next(iter(loader))
    assert len(bgr_img.shape) == 3
    assert bgr_img.shape[2] == 3
    assert np.array_equal(rgb_img, bgr_img[:, :, ::-1])
    loader = create_loader(
        storage_url, tempdir, color_space="GRAY", augmentation_config=norm_1d
    )
    gray_img, _ = next(iter(loader))
    assert len(gray_img.shape) == 3
    assert gray_img.shape[2] == 1
