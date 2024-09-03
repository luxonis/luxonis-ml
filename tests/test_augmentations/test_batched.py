import numpy as np

from luxonis_ml.data import Augmentations, LabelType


def get_img():
    return np.zeros((320, 320, 3), dtype=np.uint8)


def get_labels():
    return {
        LabelType.CLASSIFICATION: np.array([1.0]),
        LabelType.BOUNDINGBOX: np.array(
            [
                [0.0, 0.57, 0.30, 0.17, 0.25],
                [0.0, 0.39, 0.27, 0.20, 0.10],
            ]
        ),
        LabelType.KEYPOINTS: np.array(
            [
                [0.0, 0.69, 0.37, 2.0, 0.0, 0.0, 0.0, 0.68, 0.36, 2.0],
                [0.0, 0.51, 0.33, 2.0, 0.0, 0.0, 0.0, 0.50, 0.32, 2.0],
            ]
        ),
        LabelType.SEGMENTATION: np.zeros((1, 320, 320)),
    }


def test_mosaic4():
    config = {
        "name": "Mosaic4",
        "params": {
            "p": 1.0,
            "out_width": 640,
            "out_height": 640,
        },
    }
    augmentations = Augmentations([256, 256], [config])
    data = [(get_img(), get_labels()) for _ in range(4)]
    augmentations(data, nk=3, ns=1)


def test_mixup():
    config = {
        "name": "MixUp",
        "params": {
            "p": 1.0,
        },
    }
    augmentations = Augmentations([256, 256], [config])
    data = [(get_img(), get_labels()) for _ in range(2)]
    augmentations(data, nk=3, ns=1)
