import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from luxonis_ml.data import BaseDataset, DatasetIterator
from luxonis_ml.data.datasets.annotation import DatasetRecord
from luxonis_ml.enums.enums import DatasetType

ParserOutput = Tuple[DatasetIterator, Dict[str, Dict], List[Path]]


class BaseParser(ABC):
    def __init__(
        self,
        dataset: BaseDataset,
        dataset_type: DatasetType,
        task_name: Optional[Union[str, Dict[str, str]]],
    ):
        """
        @type dataset: BaseDataset
        @param dataset: Dataset to add the parsed data to.
        @type dataset_type: DatasetType
        @param dataset_type: Type of the dataset.
        @type task_name: Optional[Union[str, Dict[str, str]]]
        @param task_name: Optional task name(s) for the dataset.
            Can be either a single string, in which case all the records
            added to the dataset will use this value as `task_name`, or
            a dictionary with class names as keys and task names as values.
            In the latter case, the task name for a record with a given
            class name will be taken from the dictionary.
        """
        self.dataset = dataset
        self.dataset_type = dataset_type
        if isinstance(task_name, str):
            self.task_name = defaultdict(lambda: task_name)
        else:
            self.task_name = task_name

    @staticmethod
    @abstractmethod
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        """Validates if a split subdirectory is in an expected format.
        If so, returns kwargs to pass to L{from_split} method.

        @type split_path: Path
        @param split_path: Path to split directory.
        @rtype: Optional[Dict[str, Any]]
        @return: Dictionary with kwargs to pass to L{from_split} method
            or C{None} if the split is not in the expected format.
        """
        ...

    @staticmethod
    @abstractmethod
    def validate(dataset_dir: Path) -> bool:
        """Validates if the dataset is in an expected format.

        @type dataset_dir: Path
        @param dataset_dir: Path to source dataset directory.
        @rtype: bool
        @return: If the dataset is in the expected format.
        """
        ...

    @abstractmethod
    def from_dir(
        self, dataset_dir: Path, **kwargs
    ) -> Tuple[List[str], List[str], List[str]]:
        """Parses all present data to L{LuxonisDataset} format.

        @type dataset_dir: str
        @param dataset_dir: Path to source dataset directory.
        @type kwargs: Any
        @param kwargs: Additional arguments for a specific parser
            implementation.
        @rtype: Tuple[List[str], List[str], List[str]]
        @return: Tuple with added images for C{train}, C{val} and
            C{test} splits.
        """
        ...

    @abstractmethod
    def from_split(self, **kwargs) -> ParserOutput:
        """Parses a data in a split subdirectory to L{LuxonisDataset}
        format.

        @type kwargs: Dict[str, Any]
        @param kwargs: Additional kwargs for specific parser implementation.
            Should work together with L{validate_split} method like:

                >>> from_split(**validate_split(split_path))

        @rtype: ParserOutput
        @return: C{LDF} generator, list of class names,
            skeleton dictionary for keypoints and list of added images.
        """
        ...

    def _parse_split(self, **kwargs) -> List[Path]:
        """Parses data in a split subdirectory.

        @type kwargs: Dict[str, Any]
        @param kwargs: Additional kwargs for specific parser
            implementation.
        @rtype: List[str]
        @return: List of added images.
        """
        old_cwd = Path.cwd()
        try:
            generator, skeletons, added_images = self.from_split(**kwargs)
            self.dataset.add(self._wrap_generator(generator))
            if skeletons:
                for skeleton in skeletons.values():
                    self.dataset.set_skeletons(
                        skeleton.get("labels"),
                        skeleton.get("edges"),
                    )
            return added_images
        finally:
            os.chdir(old_cwd)

    def parse_split(
        self,
        split: Optional[str] = None,
        random_split: bool = False,
        split_ratios: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> BaseDataset:
        """Parses data in a split subdirectory to L{LuxonisDataset}
        format.

        @type split: Optional[str]
        @param split: As what split the data will be added to LDF. If
            set, C{split_ratios} and C{random_split} are ignored.
        @type random_split: bool
        @param random_split: If random splits should be made. If
            C{True}, C{split_ratios} are used.
        @type split_ratios: Optional[Tuple[float, float, float]]
        @param split_ratios: Ratios for random splits. Only used if
            C{random_split} is C{True}. Defaults to C{(0.8, 0.1, 0.1)}.
        @type kwargs: Dict[str, Any]
        @param kwargs: Additional C{kwargs} for specific parser
            implementation.
        @rtype: LuxonisDataset
        @return: C{LDF} with all the images and annotations parsed.
        """
        added_images = self._parse_split(**kwargs)
        if split is not None:
            self.dataset.make_splits({split: added_images})
        elif random_split:
            self.dataset.make_splits(split_ratios)
        return self.dataset

    def parse_dir(self, dataset_dir: Path, **kwargs) -> BaseDataset:
        """Parses entire dataset directory to L{LuxonisDataset} format.

        @type dataset_dir: str
        @param dataset_dir: Path to source dataset directory.
        @type kwargs: Dict[str, Any]
        @param kwargs: Additional C{kwargs} for specific parser
            implementation.
        @rtype: LuxonisDataset
        @return: C{LDF} with all the images and annotations parsed.
        """
        train, val, test = self.from_dir(dataset_dir, **kwargs)

        self.dataset.make_splits({"train": train, "val": val, "test": test})
        return self.dataset

    @staticmethod
    def _get_added_images(generator: DatasetIterator) -> List[Path]:
        """Returns list of unique images added by the generator
        function.

        @type generator: L{DatasetGenerator}
        @param generator: Generator function
        @rtype: List[str]
        @return: List of added images by generator function
        """
        return list(
            {
                Path(item["file"] if isinstance(item, dict) else item.file)
                for item in generator
            }
        )

    @staticmethod
    def _compare_stem_files(
        list1: Iterable[Path], list2: Iterable[Path]
    ) -> bool:
        """Compares sets of files by their stem.

        Example:

            >>> BaseParser._compare_stem_files([Path("a.jpg"), Path("b.jpg")],
            ...                                [Path("a.xml"), Path("b.xml")])
            True
            >>> BaseParser._compare_stem_files([Path("a.jpg")], [Path("b.txt")])
            False

        @type list1: Iterable[Path]
        @param list1: First list of files
        @type list2: Iterable[Path]
        @param list2: Second list of files
        @rtype: bool
        @return: If the two sets of files are equal when compared by their stems.
            If the sets are empty, returns C{False}.
        """
        set1 = {Path(f).stem for f in list1}
        set2 = {Path(f).stem for f in list2}
        return len(set1) > 0 and set1 == set2

    @staticmethod
    def _list_images(image_dir: Path) -> List[Path]:
        """Returns list of all images in the directory supported by
        opencv.

        @type image_dir: Path
        @param image_dir: Path to directory with images
        @rtype: List[Path]
        @return: List of images in the directory
        """
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

    def _wrap_generator(self, generator: DatasetIterator) -> DatasetIterator:
        """Adds task to the generator.

        @type generator: DatasetIterator
        @param generator: Generator function
        @rtype: DatasetIterator
        @return: Generator function with added task
        """

        for item in generator:
            if isinstance(item, dict):
                item = DatasetRecord(**item)

            if self.task_name is not None:
                if item.annotation is None:
                    for task_name in set(self.task_name.values()):
                        yield item.model_copy(
                            update={"task_name": task_name}, deep=True
                        )
                else:
                    class_name = item.annotation.class_name
                    if class_name is not None:
                        try:
                            task_name = self.task_name[class_name]
                        except KeyError:
                            raise ValueError(
                                f"Class '{class_name}' not found in task names."
                            ) from None

                        item.task_name = self.task_name[class_name]
                    yield item
            else:
                yield item
