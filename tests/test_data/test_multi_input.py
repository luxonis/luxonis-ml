from pathlib import Path

import pytest

from luxonis_ml.data import DatasetIterator, LuxonisLoader

from .utils import create_dataset, create_image


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_multi_input(dataset_name: str, tempdir: Path):
    def generator() -> DatasetIterator:
        for i in range(4):
            img1 = create_image(i, tempdir)
            img2 = create_image(i + 4, tempdir)
            img3 = create_image(i + 8, tempdir)
            yield {
                "files": {
                    "image1": img1,
                    "image2": img2,
                    "image3": img3,
                },
                "annotation": {
                    "class": "person",
                    "keypoints": {"keypoints": [[0.1, 0.1, 0], [0.2, 0.2, 1]]},
                    "boundingbox": {
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.2,
                        "h": 0.2,
                    },
                },
            }

    augs = [
        {"name": "Normalize"},
        {"name": "Defocus", "params": {"p": 1}},
        {
            "name": "Mosaic4",
            "params": {"out_width": 512, "out_height": 512, "p": 1},
        },
        {
            "name": "Affine",
            "params": {
                "p": 1,
                "scale": [0.8, 1.2],
                "rotate": [-10, 10],
                "translate": [0.1, 0.1],
            },
        },
    ]
    color_space = {"image1": "RGB", "image2": "BGR", "image3": "GRAY"}
    dataset = create_dataset(dataset_name, generator())
    loader = LuxonisLoader(
        dataset,
        height=512,
        width=512,
        augmentation_config=augs,
        color_space=color_space,
    )
    assert len(loader) == 4
    for img_dict, labels in loader:
        assert "image1" in img_dict
        assert "image2" in img_dict
        assert "image3" in img_dict

        assert img_dict["image1"].shape == (512, 512, 3)
        assert img_dict["image2"].shape == (512, 512, 3)
        assert img_dict["image3"].shape == (512, 512, 1)

        assert "/keypoints" in labels
        assert "/boundingbox" in labels
