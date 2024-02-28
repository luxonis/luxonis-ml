from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from luxonis_ml.data import (
    DatasetGeneratorFunction,
    LuxonisDataset,
)

ParserOutput = Tuple[DatasetGeneratorFunction, List[str], Dict[str, Dict], List[str]]
"""Type alias for parser output.

Contains a function to create the annotation generator, list of classes names, skeleton
dictionary for keypoints and list of added images.
"""


class BaseParser(ABC):
    @staticmethod
    @abstractmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        """Validates if the split is in expected format.

        @type split_path: Path
        @param split_path: Path to split directory.
        @rtype: Optional[Dict[str, Any]]
        @return: Dictionary with kwargs to pass to L{from_split} method or C{None} if
            the split is not in expected format.
        """
        pass

    @staticmethod
    @abstractmethod
    def validate(dataset_dir: Path) -> bool:
        """Validates if the dataset is in expected format.

        @type dataset_dir: Path
        @param dataset_dir: Path to source dataset directory.
        @rtype: bool
        @return: If the dataset is in expected format.
        """
        pass

    @abstractmethod
    def from_dir(
        self, dataset: LuxonisDataset, dataset_dir: Path, **kwargs
    ) -> Tuple[List[str], List[str], List[str]]:
        """Parses all present data in LuxonisDataset format.

        @type dataset_dir: str
        @param dataset_dir: Path to source dataset directory.
        @type parser_kwargs: Dict[str, Any]
        @param parser_kwargs: Additional kwargs for specific parser function.
        @rtype: LuxonisDataset
        @return: Output LDF with all images and annotations parsed.
        """
        pass

    @abstractmethod
    def from_split(self, **kwargs) -> ParserOutput:
        pass

    def _parse_split(self, dataset: LuxonisDataset, **kwargs) -> List[str]:
        generator, class_names, skeletons, added_images = self.from_split(**kwargs)
        dataset.set_classes(class_names)
        dataset.set_skeletons(skeletons)
        dataset.add(generator)

        return added_images

    def parse_split(
        self,
        dataset: LuxonisDataset,
        split: Optional[Literal["train", "val", "test"]] = None,
        random_split: bool = False,
        split_ratios: Optional[Tuple[float, float, float]] = None,
        **kwargs,
    ) -> LuxonisDataset:
        added_images = self._parse_split(dataset, **kwargs)
        if split:
            dataset.make_splits(definitions={split: added_images})
        elif random_split:
            split_ratios = split_ratios or (0.8, 0.1, 0.1)
            dataset.make_splits(split_ratios)
        return dataset

    def parse_dir(
        self, dataset: LuxonisDataset, dataset_dir: Path, **kwargs
    ) -> LuxonisDataset:
        train, test, val = self.from_dir(dataset, dataset_dir, **kwargs)

        dataset.make_splits(
            definitions={
                "train": train,
                "val": val,
                "test": test,
            }
        )
        return dataset

    @staticmethod
    def _get_added_images(generator: DatasetGeneratorFunction) -> List[str]:
        """Returns list of unique images added by the generator function.

        @type generator: L{DatasetGeneratorFunction}
        @param generator: Generator function
        @rtype: List[str]
        @return: List of added images by generator function
        """
        return list(set(item["file"] for item in generator()))

    @staticmethod
    def _compare_stem_files(list1: Iterable[Path], list2: Iterable[Path]) -> bool:
        set1 = set(Path(f).stem for f in list1)
        set2 = set(Path(f).stem for f in list2)
        return len(set1) > 0 and set1 == set2

    @staticmethod
    def _list_images(image_dir: Path) -> List[Path]:
        cv2_supported_image_formats = {
            ".bmp",
            ".dib",
            ".jpeg",
            ".jpg",
            ".jpe",
            ".jp2",
            ".png",
            ".WebP",
            ".webp",
            ".pbm",
            ".pgm",
            ".ppm",
            ".pxm",
            ".pnm",
            ".sr",
            ".ras",
            ".tiff",
            ".tif",
            ".exr",
            ".hdr",
            ".pic",
        }
        return [
            img
            for img in image_dir.glob("*")
            if img.suffix in cv2_supported_image_formats
        ]
