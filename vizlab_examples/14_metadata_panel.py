"""Metadata sidebar: a "second window" of arbitrary JSON-like image info.

``with_panel`` (or ``Image.with_panel``) renders the image and appends a panel
beside it showing whatever metadata you pass — augmentations, source, tags,
filenames — as an indented key/value tree. It never occludes the image or labels.
"""

from _common import gradient, save

from luxonis_ml.vizlab import BBox, Classification, Image

METADATA = {
    "source": "coco/val2017/000000042.jpg",
    "split": "train",
    "image_id": 42,
    "augmentations": [
        "horizontal_flip",
        "random_resized_crop (0.8)",
        "gaussian_blur (sigma=0.5)",
    ],
    "tags": {
        "difficulty": "hard",
        "verified": True,
        "occlusion": 0.35,
    },
    "notes": "reviewed twice; bounding boxes tightened around occluded regions",
}


def main() -> None:
    """Render a labeled scene with a metadata sidebar."""
    img = Image(gradient(420, 320, hue=0.58))
    img.add(BBox((40, 60, 180, 240), label="person", score=0.97))
    img.add(BBox((180, 110, 200, 170), label="dog", score=0.86))
    img.add(Classification(tags=[("outdoor", 0.98)]))

    save(img.with_panel(METADATA, title="metadata"), "14_metadata_panel.png")


if __name__ == "__main__":
    main()
