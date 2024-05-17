from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class SegmentationMaskDirectoryParser(BaseParser):
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

    This is one of the formats that can be generated by
    U{Roboflow <https://roboflow.com/>}.
    """

    @staticmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        if not split_path.exists():
            return None
        if not (split_path / "_classes.csv").exists():
            return None
        masks = list(split_path.glob("*_mask.*"))
        for mask_path in masks:
            img_path = split_path / f"{mask_path.stem[:-5]}.jpg"
            if not img_path.exists():
                return None
        return {
            "image_dir": split_path,
            "seg_dir": split_path,
            "classes_path": split_path / "_classes.csv",
        }

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if SegmentationMaskDirectoryParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(self, dataset_dir: Path) -> Tuple[List[str], List[str], List[str]]:
        added_train_imgs = self._parse_split(
            image_dir=dataset_dir / "train",
            seg_dir=dataset_dir / "train",
            classes_path=dataset_dir / "train" / "_classes.csv",
        )
        added_val_imgs = self._parse_split(
            image_dir=dataset_dir / "valid",
            seg_dir=dataset_dir / "valid",
            classes_path=dataset_dir / "valid" / "_classes.csv",
        )
        added_test_imgs = self._parse_split(
            image_dir=dataset_dir / "test",
            seg_dir=dataset_dir / "test",
            classes_path=dataset_dir / "test" / "_classes.csv",
        )
        return added_train_imgs, added_val_imgs, added_test_imgs

    def from_split(
        self, image_dir: Path, seg_dir: Path, classes_path: Path
    ) -> ParserOutput:
        """Parses annotations with SegmentationMask format to LDF.

        Annotations include classification and segmentation.

        @type image_dir: Path
        @param image_dir: Path to directory with images
        @type seg_dir: Path
        @param seg_dir: Path to directory with segmentation mask
        @type classes_path: Path
        @param classes_path: Path to CSV file with class names
        @rtype: L{ParserOutput}
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images
        """
        df = pd.read_csv(classes_path)

        idx_pixel_val = "Pixel Value"
        idx_class = " Class"  # NOTE: space prefix included

        class_names = pd.Series(df[idx_class].values, index=df[idx_pixel_val]).to_dict()

        def generator() -> DatasetIterator:
            for mask_path in seg_dir.glob("*_mask.*"):
                image_path = next(image_dir.glob(f"{mask_path.stem[:-5]}.*"))
                file = str(image_path.absolute())
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                ids = np.unique(mask)
                for id in ids:
                    class_name = class_names[id]
                    yield {
                        "file": file,
                        "annotation": {
                            "class": class_name,
                        },
                    }

                    curr_seg_mask = np.zeros_like(mask)
                    curr_seg_mask[mask == id] = 1
                    curr_seg_mask = np.asfortranarray(
                        curr_seg_mask
                    )  # pycocotools requirement
                    curr_rle = mask_util.encode(curr_seg_mask)
                    yield {
                        "file": file,
                        "annotation": {
                            "class": class_name,
                            "width": curr_rle["size"][0],
                            "height": curr_rle["size"][1],
                            "counts": curr_rle["counts"],
                        },
                    }

        added_images = self._get_added_images(generator())
        return generator(), list(class_names.values()), {}, added_images
