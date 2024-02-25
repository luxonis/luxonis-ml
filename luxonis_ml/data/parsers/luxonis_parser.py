import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from luxonis_ml.data import (
    DatasetGeneratorFunction,
    LuxonisDataset,
)

ParserOutput = Tuple[DatasetGeneratorFunction, List[str], Dict[str, Dict], List[str]]
"""Type alias for parser output.

Contains a function to create the annotation generator, list of classes names, skeleton
dictionary for keypoints and list of added images.
"""


class LuxonisParser(ABC):
    def __init__(self, **ldf_kwargs):
        """A parser class used for parsing common dataset formats to LDF.

        @type ldf_kwargs: Dict[str, Any]
        @param ldf_kwargs: Init parameters for L{LuxonisDataset}.
        """

        self.logger = logging.getLogger(__name__)
        self.dataset_exists = LuxonisDataset.exists(
            dataset_name=ldf_kwargs["dataset_name"]
        )
        self.dataset = LuxonisDataset(**ldf_kwargs)

    @abstractmethod
    def validate(self, dataset_dir: Path) -> bool:
        """Validates if the dataset is in expected format.

        @type dataset_dir: str
        @param dataset_dir: Path to source dataset directory.
        @rtype: bool
        @return: If the dataset is in expected format.
        """
        pass

    @abstractmethod
    def from_dir(self, dataset_dir: Path, **parser_kwargs) -> None:
        """Parses all present data in LuxonisDataset format. Check under selected parser
        function for expected directory structure.

        @type dataset_dir: str
        @param dataset_dir: Path to source dataset directory.
        @type parser_kwargs: Dict[str, Any]
        @param parser_kwargs: Additional kwargs for specific parser function.
        @rtype: LuxonisDataset
        @return: Output LDF with all images and annotations parsed.
        """
        pass

    @abstractmethod
    def _from_format(self, *args, **kwargs) -> ParserOutput:
        pass

    def from_format(self, *args, **kwargs):
        generator, class_names, skeletons, added_images = self._from_format(
            *args, **kwargs
        )
        self.dataset.set_classes(class_names)
        self.dataset.set_skeletons(skeletons)
        self.dataset.add(generator)

        return added_images

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
        return set(Path(f).stem for f in list1) == set(Path(f).stem for f in list2)

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
