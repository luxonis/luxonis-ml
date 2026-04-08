import albumentations as A
import numpy as np
import pytest

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
        256,
        256,
        {"/metadata/id": "metadata/id"},
        {"/metadata/id": 0},
        ["image"],
        config,
    )
    _, labels = augmentations.apply(
        [
            (
                {"image": np.zeros((3, 256, 256))},
                {"/metadata/id": np.array([i])},
            )
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
    source_names = ["image"]
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )

    spatial_transform_names = next(
        (
            [t.__class__.__name__ for t in cell.cell_contents.transforms]
            for cell in augmentations.spatial_transform.__closure__  # type: ignore
            if hasattr(cell.cell_contents, "transforms")
        ),
        [],
    )

    batched_transform_names = [
        t.__class__.__name__ for t in augmentations.batch_transform.transforms
    ]
    assert spatial_transform_names == [
        "Perspective",
        "Lambda",
        "HorizontalFlip",
        "Lambda",
        "VerticalFlip",
        "Lambda",
        "Rotate",
        "Lambda",
    ]
    assert batched_transform_names == ["Mosaic4"]


def test_use_for_resizing_wraps_probabilistic_resize_in_oneof():
    augmentations = AlbumentationsEngine(
        256,
        256,
        {"task/boundingbox": "boundingbox"},
        {"task/boundingbox": 1},
        ["image"],
        [
            {
                "name": "AtLeastOneBBoxRandomCrop",
                "params": {
                    "height": 32,
                    "width": 32,
                    "erosion_factor": 0.0,
                    "p": 0.3,
                },
                "use_for_resizing": True,
            }
        ],
        keep_aspect_ratio=False,
    )

    resize_ops = next(
        (
            cell.cell_contents.transforms
            for cell in augmentations.resize_transform.__closure__  # type: ignore
            if hasattr(cell.cell_contents, "transforms")
        ),
        [],
    )

    resize_op = resize_ops[0]
    assert isinstance(resize_op, A.OneOf)
    assert len(resize_op.transforms) == 2
    assert isinstance(resize_op.transforms[0], A.AtLeastOneBBoxRandomCrop)
    assert isinstance(resize_op.transforms[1], A.Resize)
    assert resize_op.transforms[0].height == 256
    assert resize_op.transforms[0].width == 256
    assert resize_op.transforms[0].p == pytest.approx(0.3)
    assert resize_op.transforms[1].p == pytest.approx(0.7)
    assert resize_op.transforms[1].height == 256
    assert resize_op.transforms[1].width == 256


def test_use_for_resizing_falls_back_to_default_resize():
    # The probabilistic AtLeastOneBBoxRandomCrop with use_for_resizing should still preserve
    # the required final image size through the default resize fallback.
    augmentations = AlbumentationsEngine(
        256,
        256,
        {"task/boundingbox": "boundingbox"},
        {"task/boundingbox": 1},
        ["image"],
        [
            {
                "name": "AtLeastOneBBoxRandomCrop",
                "params": {
                    "height": 32,
                    "width": 32,
                    "erosion_factor": 0.0,
                    "p": 0.0,
                },
                "use_for_resizing": True,
            }
        ],
        keep_aspect_ratio=False,
        seed=42,
    )

    images, _ = augmentations.apply(
        [
            (
                {"image": np.full((100, 200, 3), 255, dtype=np.uint8)},
                {"task/boundingbox": np.array([[0.0, 0.5, 0.5, 0.1, 0.1]])},
            )
        ]
    )

    assert images["image"].shape == (256, 256, 3)
