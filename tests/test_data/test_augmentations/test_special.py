import numpy as np

from luxonis_ml.data import AlbumentationsEngine


def test_metadata_no_boxes():
    config = [
        {
            "name": "Defocus",
            "params": {"p": 1.0},
        },
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        },
    ]
    augmentations = AlbumentationsEngine(
        256, 256, {"/metadata/id": "metadata/id"}, {"/metadata/id": 0}, config
    )
    _, labels = augmentations.apply(
        [
            (np.zeros((3, 256, 256)), {"/metadata/id": np.array([i])})
            for i in range(4)
        ]
    )
    assert labels["/metadata/id"].tolist() == [0, 1, 2, 3]
