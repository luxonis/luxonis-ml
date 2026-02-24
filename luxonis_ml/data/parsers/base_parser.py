import inspect
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from luxonis_ml.data import BaseDataset, DatasetIterator
from luxonis_ml.data.datasets.annotation import DatasetRecord
from luxonis_ml.enums.enums import DatasetType
from luxonis_ml.typing import PathType

if TYPE_CHECKING:
    from luxonis_ml.data import LuxonisDataset

ParserOutput = tuple[DatasetIterator, dict[str, dict], list[Path]]


class BaseParser(ABC):
    SPLIT_NAMES: tuple[str, ...] = ("train", "valid", "test")

    def __init__(
        self,
        dataset: BaseDataset,
        dataset_type: DatasetType,
        task_name: str | dict[str, str] | None,
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
        self.initial_class_ordering: dict[str, list[str]] | None = None
        if isinstance(task_name, str):
            self.task_name = defaultdict(lambda: task_name)
        else:
            self.task_name = task_name

    def _set_initial_class_ordering(self, ordered_classes: list[str]) -> None:
        """Sets initial_class_ordering from an ordered list of class
        names, grouping by task name if there is one configured.

        @type ordered_classes: list[str]
        @param ordered_classes: Class names in the initial order
        """
        if self.task_name is not None:
            tasks_to_classes: dict[str, list[str]] = {}
            for class_name in ordered_classes:
                tasks_to_classes.setdefault(
                    self.task_name[class_name], []
                ).append(class_name)
            self.initial_class_ordering = tasks_to_classes
        else:
            self.initial_class_ordering = {"": ordered_classes}

    @staticmethod
    @abstractmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        """Validates if a split subdirectory is in an expected format.
        If so, returns kwargs to pass to L{from_split} method.

        @type split_path: Path
        @param split_path: Path to split directory.
        @rtype: Optional[Dict[str, Any]]
        @return: Dictionary with kwargs to pass to L{from_split} method
            or C{None} if the split is not in the expected format.
        """
        ...

    @classmethod
    def validate(cls, dataset_dir: Path) -> bool:
        """Validates if the dataset is in an expected format.

        @type dataset_dir: Path
        @param dataset_dir: Path to source dataset directory.
        @rtype: bool
        @return: If the dataset is in the expected format.
        """
        splits = [
            d.name
            for d in dataset_dir.iterdir()
            if d.is_dir() and d.name in cls.SPLIT_NAMES
        ]
        if len(splits) == 0:
            return False

        return all(cls.validate_split(dataset_dir / split) for split in splits)

    @abstractmethod
    def from_dir(
        self, dataset_dir: Path, **kwargs
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Parses all present data to L{LuxonisDataset} format.

        @type dataset_dir: Path
        @param dataset_dir: Path to source dataset directory.
        @type kwargs: Any
        @param kwargs: Additional arguments for a specific parser
            implementation.
        @rtype: Tuple[List[Path], List[Path], List[Path]]
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

    def _parse_split(self, **kwargs) -> list[Path]:
        """Parses data in a split subdirectory.

        @type kwargs: Dict[str, Any]
        @param kwargs: Additional kwargs for specific parser
            implementation.
        @rtype: List[str]
        @return: List of added images.
        """
        generator, skeletons, added_images = self.from_split(**kwargs)
        self.dataset.add(
            self._wrap_generator(generator),
            initial_class_ordering=self.initial_class_ordering,
        )
        if skeletons:
            for skeleton in skeletons.values():
                self.dataset.set_skeletons(
                    skeleton.get("labels"),
                    skeleton.get("edges"),
                )
        return added_images

    def _apply_counts_to_pool(
        self,
        images: Sequence[PathType],
        split_ratios: dict[str, int],
    ) -> dict[str, Sequence[PathType]]:
        """Distributes images across splits based on counts.

        When total requested exceeds available, fills splits by priority
        (most requested first).

        @type images: Sequence[PathType]
        @param images: List of images to distribute.
        @type split_ratios: Dict[str, int]
        @param split_ratios: Counts for each split.
        @rtype: Dict[str, Sequence[PathType]]
        @return: Dictionary mapping split names to their assigned
            images.
        """
        total_requested = sum(split_ratios.values())
        available = len(images)

        shuffled = list(images)
        random.shuffle(shuffled)

        if total_requested > available:
            logger.warning(
                f"Requested {total_requested} total samples, "
                f"but only {available} available. "
                "Filling splits by priority (most requested first)."
            )
            sorted_splits = sorted(
                ["train", "val", "test"],
                key=lambda s: split_ratios[s],
                reverse=True,
            )

            sampled: dict[str, Sequence[PathType]] = {}
            remaining = available
            offset = 0
            for split_name in sorted_splits:
                count = min(split_ratios[split_name], remaining)
                sampled[split_name] = shuffled[offset : offset + count]
                offset += count
                remaining -= count
            return sampled

        # Enough samples: distribute in order
        sampled = {}
        offset = 0
        for split_name in ["train", "val", "test"]:
            count = split_ratios[split_name]
            sampled[split_name] = shuffled[offset : offset + count]
            offset += count
        return sampled

    def _sample_from_splits(
        self,
        original_splits: dict[str, Sequence[PathType]],
        split_ratios: dict[str, int],
    ) -> dict[str, Sequence[PathType]]:
        """Samples from each original split independently.

        @type original_splits: Dict[str, Sequence[PathType]]
        @param original_splits: Original split assignments.
        @type split_ratios: Dict[str, int]
        @param split_ratios: Requested counts for each split.
        @rtype: Dict[str, Sequence[PathType]]
        @return: Dictionary mapping split names to sampled images.
        """
        sampled: dict[str, Sequence[PathType]] = {}
        for split_name in ["train", "val", "test"]:
            requested = split_ratios[split_name]
            available = original_splits[split_name]
            available_count = len(available)

            if requested == 0:
                sampled[split_name] = []
            elif requested >= available_count:
                if requested > available_count:
                    logger.warning(
                        f"Requested {requested} samples for '{split_name}' split, "
                        f"but only {available_count} available. "
                        f"Using all {available_count} samples."
                    )
                sampled[split_name] = list(available)
            else:
                sampled[split_name] = random.sample(list(available), requested)
        return sampled

    def parse_split(
        self,
        split: str | None = None,
        random_split: bool = False,
        split_ratios: dict[str, float | int] | None = None,
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
        @type split_ratios: Optional[Dict[str, Union[float, int]]]
        @param split_ratios: Ratios or counts for splits. Only used if
            C{random_split} is C{True}. If floats, treated as ratios. If
            ints, treated as counts. Defaults to C{(0.8, 0.1, 0.1)}.
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
            is_counts = split_ratios is not None and all(
                isinstance(v, int) for v in split_ratios.values()
            )
            if is_counts:
                sampled = self._apply_counts_to_pool(
                    added_images,
                    split_ratios,  # type: ignore[arg-type]
                )
                self.dataset.make_splits(sampled)
                self._remove_unsplit_records()
            else:
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
        # Skip train directory check for parsers that use images/labels
        # subdirectory structure (YoloV6, YOLOv8) instead of train/valid/test
        # at root level
        skip_train_check = self.__class__.__name__ in (
            "YoloV6Parser",
            "YOLOv8Parser",
        )
        if not skip_train_check:
            train_dir = dataset_dir / "train"
            if not train_dir.exists() or not train_dir.is_dir():
                existing_dirs = [
                    d.name for d in dataset_dir.iterdir() if d.is_dir()
                ]
                raise ValueError(
                    f"Train split not found in dataset. "
                    f"Expected a 'train' directory but found: {existing_dirs}."
                )

        split_ratios = kwargs.pop("split_ratios", None)
        is_counts = split_ratios is not None and all(
            isinstance(v, int) for v in split_ratios.values()
        )

        # Disable automatic val-to-test splitting when using explicit counts
        if is_counts and "split_val_to_test" not in kwargs:
            sig = inspect.signature(self.from_dir)
            if "split_val_to_test" in sig.parameters:
                kwargs["split_val_to_test"] = False

        train, val, test = self.from_dir(dataset_dir, **kwargs)
        original_splits: dict[str, Sequence[PathType]] = {
            "train": train,
            "val": val,
            "test": test,
        }

        if split_ratios is None:
            self.dataset.make_splits(original_splits)
        elif is_counts:
            sampled = self._apply_counts_to_splits(
                original_splits,
                split_ratios,  # type: ignore[arg-type]
            )
            self.dataset.make_splits(sampled)
            self._remove_unsplit_records()
        else:
            logger.warning(
                "Using percentage-based split ratios will redistribute "
                "and shuffle all samples across splits. Original split "
                "boundaries will not be preserved."
            )
            self.dataset.make_splits(split_ratios)

        return self.dataset

    def _apply_counts_to_splits(
        self,
        original_splits: dict[str, Sequence[PathType]],
        split_ratios: dict[str, int],
    ) -> dict[str, Sequence[PathType]]:
        """Applies count-based split ratios to pre-existing splits.

        Samples from each original split independently. If more samples
        are requested than available in a split, all available samples
        from that split are used.

        @type original_splits: Dict[str, Sequence[PathType]]
        @param original_splits: Original split assignments.
        @type split_ratios: Dict[str, int]
        @param split_ratios: Requested counts for each split.
        @rtype: Dict[str, Sequence[PathType]]
        @return: Dictionary mapping split names to assigned images.
        """
        return self._sample_from_splits(original_splits, split_ratios)

    def _remove_unsplit_records(self) -> None:
        """Removes records from the dataset that are not assigned to any
        split."""
        # Cast to LuxonisDataset to access internal methods
        dataset: LuxonisDataset = self.dataset  # type: ignore[assignment]

        splits = dataset.get_splits()
        if splits is None:
            return

        # Get all group_ids that are in any split
        all_split_group_ids: set[str] = set()
        for group_ids in splits.values():
            all_split_group_ids.update(group_ids)

        if not all_split_group_ids:
            return

        df = dataset._load_df_offline(lazy=True)
        if df is None:
            return

        # Filter to keep only records whose group_id is in a split
        df = df.filter(pl.col("group_id").is_in(list(all_split_group_ids)))
        dataset._save_df_offline(df.collect())

    @staticmethod
    def _get_added_images(generator: DatasetIterator) -> list[Path]:
        """Returns list of unique images added by the generator
        function.

        @type generator: L{DatasetGenerator}
        @param generator: Generator function
        @rtype: List[str]
        @return: List of added images by generator function
        """
        return list(
            {
                Path(v)
                for item in generator
                for v in (
                    [item["file"]]
                    if isinstance(item, dict) and "file" in item
                    else item["files"].values()
                    if isinstance(item, dict) and "files" in item
                    else [item.file]
                    if isinstance(item, DatasetRecord)
                    else []
                )
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
    def _list_images(image_dir: Path) -> list[Path]:
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
