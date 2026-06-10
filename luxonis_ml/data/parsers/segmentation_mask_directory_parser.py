from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class SegmentationMaskDirectoryParser(BaseParser):
    """Parse a directory with segmentation mask annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── img1.jpg
        │   ├── img1_mask.png
        │   ├── ...
        │   └── _classes.csv
        ├── valid/
        └── test/

    ``_classes.csv`` maps pixel values to class names.

    This is one of the formats that Roboflow can generate.
    """

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
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

    def from_dir(
        self, dataset_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
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
        """Parse segmentation mask annotations into LDF records.

        Annotations include classification and segmentation.

        Args:
            image_dir: Directory with images.
            seg_dir: Directory with segmentation masks.
            classes_path: CSV file with class names.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        """
        # NOTE: space prefix included
        idx_class = " Class"

        df = pl.read_csv(classes_path).filter(pl.col(idx_class).is_not_null())
        class_names = df[idx_class].to_list()

        def generator() -> DatasetIterator:
            for mask_path in seg_dir.glob("*_mask.*"):
                image_path = next(image_dir.glob(f"{mask_path.stem[:-5]}.*"))
                file = str(image_path.absolute().resolve())
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                ids = np.unique(mask)
                for id in ids:
                    class_name = class_names[id]

                    curr_seg_mask = np.zeros_like(mask)
                    curr_seg_mask[mask == id] = 1
                    yield {
                        "file": file,
                        "annotation": {
                            "class": class_name,
                            "segmentation": {"mask": curr_seg_mask},
                        },
                    }

        added_images = self._get_added_images(generator())
        return generator(), {}, added_images
