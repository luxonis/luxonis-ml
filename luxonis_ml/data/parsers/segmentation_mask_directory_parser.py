import csv
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

from luxonis_ml.data import DatasetGenerator

from .luxonis_parser import LuxonisParser, ParserOutput


class SegmentationMaskDirectoryParser(LuxonisParser):
    def validate(self, dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if not split_path.exists():
                return False
            if not (split_path / "_classes.csv").exists():
                return False
            masks = list(split_path.glob("*_mask.*"))
            img_stems = [str(m).rstrip(f"_mask{m.suffix}") for m in masks]
            if not [str(m.stem) for m in masks] != img_stems:
                return False
        return True

    def from_dir(self, dataset_dir: str) -> None:
        """Parses directory with SegmentationMask annotations to LDF.

        Expected format::

            dataset_dir/
            ├── train/
            │   ├── img1.jpg
            │   ├── img1_mask.png
            │   ├── ...
            │   └── _classes.csv
            ├── valid/
            └── test/

        C{_classes.csv} contains mappings between pixel value and class name.

        This is the default format returned when using U{Roboflow <https://roboflow.com/>}.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "train"),
            seg_dir=osp.join(dataset_dir, "train"),
            classes_path=osp.join(dataset_dir, "train", "_classes.csv"),
        )
        added_val_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "valid"),
            seg_dir=osp.join(dataset_dir, "valid"),
            classes_path=osp.join(dataset_dir, "valid", "_classes.csv"),
        )
        added_test_imgs = self.from_format(
            image_dir=osp.join(dataset_dir, "test"),
            seg_dir=osp.join(dataset_dir, "test"),
            classes_path=osp.join(dataset_dir, "test", "_classes.csv"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    def _from_format(
        self, image_dir: str, seg_dir: str, classes_path: str
    ) -> ParserOutput:
        """Parses annotations with SegmentationMask format to LDF.

        Annotations include classification and segmentation.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type seg_dir: str
        @param seg_dir: Path to directory with segmentation mask
        @type classes_path: str
        @param classes_path: Path to CSV file with class names
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images
        """
        with open(classes_path) as f:
            reader = csv.reader(f, delimiter=",")

            class_names = {}
            for i, row in enumerate(reader):
                if i == 0:
                    idx_pixel_val = row.index("Pixel Value")

                    # NOTE: space prefix included
                    idx_class = row.index(" Class")
                else:
                    class_names[int(row[idx_pixel_val])] = row[idx_class]

        def generator() -> DatasetGenerator:
            images = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
            for image_path in images:
                mask_path = image_path.replace(".jpg", "_mask.png")
                mask_path = osp.abspath(osp.join(seg_dir, mask_path))
                path = osp.abspath(osp.join(image_dir, image_path))
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                ids = np.unique(mask)
                for id in ids:
                    class_name = class_names[id]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

                    curr_seg_mask = np.zeros_like(mask)
                    curr_seg_mask[mask == id] = 1
                    curr_seg_mask = np.asfortranarray(
                        curr_seg_mask
                    )  # pycocotools requirement
                    curr_rle = mask_util.encode(curr_seg_mask)
                    value = (
                        curr_rle["size"][0],
                        curr_rle["size"][1],
                        curr_rle["counts"],
                    )
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "segmentation",
                        "value": value,
                    }

        added_images = self._get_added_images(generator)
        return generator, list(class_names.values()), {}, added_images
