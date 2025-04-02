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


def test_skip_augmentations():
    config = [
        {
            "name": "Perspective",
        },
        {
            "name": "Flip",
        },
        {
            "name": "HorizontalFlip",
        },
        {
            "name": "VerticalFlip",
        },
        {
            "name": "Rotate",
        },
        {
            "name": "Mosaic4",
            "params": {"out_width": 640, "out_height": 640},
        },
    ]
    targets = {
        "/boundingbox": "boundingbox",
        "/classification": "classification",
        "/keypoints": "keypoints",
        "/instance_segmentation": "instance_segmentation",
        "/segmentation": "segmentation",
    }
    n_classes = {
        "/boundingbox": 1,
        "/classification": 1,
        "/keypoints": 1,
        "/instance_segmentation": 1,
        "/segmentation": 1,
    }
    augmentations = AlbumentationsEngine(256, 256, targets, n_classes, config)

    spatial_transform_names = next(
        (
            [t.__class__.__name__ for t in cell.cell_contents.transforms]
            for cell in augmentations.spatial_transform.__closure__
            if hasattr(cell.cell_contents, "transforms")
        ),
        [],
    )

    batched_transform_names = [
        t.__class__.__name__ for t in augmentations.batch_transform.transforms
    ]

    assert spatial_transform_names == ["Rotate"]
    assert batched_transform_names == ["Mosaic4"]
