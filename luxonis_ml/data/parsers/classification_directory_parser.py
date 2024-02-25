import os
import os.path as osp
from pathlib import Path

from luxonis_ml.data import DatasetGenerator

from .luxonis_parser import LuxonisParser, ParserOutput


class ClassificationDirectoryParser(LuxonisParser):
    def validate(self, dataset_dir: Path) -> bool:
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if not split_path.exists():
                return False
            classes = [d for d in split_path.iterdir() if d.is_dir()]
            if not classes:
                return False
        return True

    def from_dir(self, dataset_dir: str) -> None:
        """Parses directory with ClassificationDirectory annotations to LDF.

        Expected format::

            dataset_dir/
            ├── train/
            │   ├── class1/
            │   │   ├── img1.jpg
            │   │   ├── img2.jpg
            │   │   └── ...
            │   ├── class2/
            │   └── ...
            ├── valid/
            └── test/

        This is the default format returned when using U{Roboflow <https://roboflow.com/>}.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_format(
            class_dir=osp.join(dataset_dir, "train"),
        )
        added_val_imgs = self.from_format(
            class_dir=osp.join(dataset_dir, "valid"),
        )
        added_test_imgs = self.from_format(
            class_dir=osp.join(dataset_dir, "test"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    def _from_format(self, class_dir: str) -> ParserOutput:
        """Parses annotations from classification directory format to LDF. Annotations
        include classification.

        @type class_dir: str
        @param class_dir: Path to top level directory
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        class_names = [
            d for d in os.listdir(class_dir) if osp.isdir(osp.join(class_dir, d))
        ]

        def generator() -> DatasetGenerator:
            for class_name in class_names:
                images = os.listdir(osp.join(class_dir, class_name))
                for img_path in images:
                    path = osp.join(osp.abspath(class_dir), class_name, img_path)
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

        added_images = self._get_added_images(generator)

        return generator, class_names, {}, added_images
