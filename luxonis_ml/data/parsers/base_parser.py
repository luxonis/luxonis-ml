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
from luxonis_ml.data.utils.enums import ParserIssue, ParserIssueMessage
from luxonis_ml.enums.enums import DatasetType
from luxonis_ml.typing import PathType

if TYPE_CHECKING:
    from luxonis_ml.data import LuxonisDataset

ParserOutput = tuple[DatasetIterator, dict[str, dict], list[Path]]


class BaseParser(ABC):
    """Base class for dataset-format parsers.

    Attributes:
        SPLIT_NAMES: Canonical split directory names checked by
            ``validate``.
        dataset: Dataset being populated by the parser.
        dataset_type: Dataset format handled by the parser.
        task_name: Optional task naming rule. When a string is provided,
            every parsed record receives that task name. When a mapping is
            provided, class names are mapped to task names.

    """

    SPLIT_NAMES: tuple[str, ...] = ("train", "valid", "test")
    CANONICAL_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

    def __init__(
        self,
        dataset: BaseDataset,
        dataset_type: DatasetType,
        task_name: str | dict[str, str] | None,
    ):
        """Create a parser for a target dataset.

        Args:
            dataset: Dataset to populate with parsed records.
            dataset_type: Source dataset format.
            task_name: Optional task naming rule. A string is used for all
                records. A mapping uses class names as keys and task names
                as values.

        """
        self.dataset = dataset
        self.dataset_type = dataset_type
        if isinstance(task_name, str):
            self.task_name = defaultdict(lambda: task_name)
        else:
            self.task_name = task_name
        self._parser_issue_messages: list[ParserIssueMessage] = []
        self._seen_parser_issue_messages: set[ParserIssueMessage] = set()

    def reset_parser_issue_messages(self) -> None:
        """Clear collected parser issue messages."""
        self._parser_issue_messages.clear()
        self._seen_parser_issue_messages.clear()

    def get_parser_issue_messages(self) -> list[ParserIssueMessage]:
        """Return parser issue messages collected during the last
        parse.
        """
        return list(self._parser_issue_messages)

    @staticmethod
    @abstractmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        """Validate whether a split directory has the expected format.

        Args:
            split_path: Path to a split directory.

        Returns:
            Keyword arguments for ``from_split``, or ``None`` if the split
            is not in the expected format.

        """
        ...

    @classmethod
    def validate(cls, dataset_dir: Path) -> bool:
        """Validate whether the dataset directory has the expected
        format.

        Args:
            dataset_dir: Source dataset directory.

        Returns:
            Whether the dataset is in the expected format.

        """
        splits = [
            d.name
            for d in dataset_dir.iterdir()
            if d.is_dir() and d.name in cls.SPLIT_NAMES
        ]
        if len(splits) == 0:
            return False

        return all(cls.validate_split(dataset_dir / split) for split in splits)

    @classmethod
    def _canonicalize_split_name(cls, split_name: str) -> str:
        """All current parsers use ``train`` and ``test`` split names
        whereas validation splits can vary in name between ``val`` ``valid``
        and ``validation``.

        This maps ``valid`` -> ``val`` and ``validation`` -> val
        """
        if split_name in {"valid", "validation"}:
            return "val"
        return split_name

    @classmethod
    def discover_dir_splits(
        cls, dataset_dir: Path
    ) -> dict[str, dict[str, Any]]:
        """Return present and valid split directories keyed by their
        canonical split names.
        """
        discovered: dict[str, dict[str, Any]] = {}
        for split_name in cls.SPLIT_NAMES:
            split_kwargs = cls.validate_split(dataset_dir / split_name)
            if split_kwargs is None:
                continue
            discovered[cls._canonicalize_split_name(split_name)] = split_kwargs
        return discovered

    @abstractmethod
    def from_dir(
        self, dataset_dir: Path, **kwargs
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Parse all data in a source dataset directory.

        Args:
            dataset_dir: Source dataset directory.
            kwargs: Additional parser-specific arguments.

        Returns:
            Added images for the train, validation, and test splits.

        """
        ...

    @abstractmethod
    def from_split(self, **kwargs) -> ParserOutput:
        """Parse data from one split subdirectory.

        Args:
            kwargs: Parser-specific arguments, usually produced by
                ``validate_split``.

        Example:
            .. python::

                split_kwargs = parser.validate_split(split_path)
                if split_kwargs is not None:
                    parser.from_split(**split_kwargs)

        Returns:
            LDF generator, skeleton metadata, and added images.

        """
        ...

    def _parse_split(self, **kwargs) -> list[Path]:
        """Parse data in one split subdirectory.

        Args:
            kwargs: Parser-specific arguments.

        Returns:
            Added images.

        """
        generator, skeletons, added_images = self.from_split(**kwargs)
        self.dataset.add(self._wrap_generator(generator))
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
        """Distribute images across splits based on requested counts.

        When total requested exceeds available, fills splits by priority
        (most requested first).

        Args:
            images: Images to distribute.
            split_ratios: Requested counts for each split.

        Returns:
            Split names mapped to assigned images.

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
        """Sample from each original split independently.

        Args:
            original_splits: Existing split assignments.
            split_ratios: Requested counts for each split.

        Returns:
            Split names mapped to sampled images.

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
        """Parse one split subdirectory into the target dataset.

        Args:
            split: Optional split name to assign to parsed data. When set,
                ``split_ratios`` and ``random_split`` are ignored.
            random_split: Whether to generate random splits using
                ``split_ratios``.
            split_ratios: Optional ratios or counts. Float values are
                treated as ratios; integer values are treated as counts.
            kwargs: Parser-specific arguments.

        Returns:
            Dataset with parsed images and annotations.

        """
        self.reset_parser_issue_messages()
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
        """Parse an entire dataset directory into the target dataset.

        Args:
            dataset_dir: Source dataset directory.
            kwargs: Parser-specific arguments.

        Returns:
            Dataset with parsed images and annotations.

        Raises:
            ValueError: If a parser that expects top-level splits cannot
                find a ``train`` directory.

        """
        self.reset_parser_issue_messages()
        split_ratios = kwargs.pop("split_ratios", None)
        is_counts = split_ratios is not None and all(
            isinstance(v, int) for v in split_ratios.values()
        )

        # Disable automatic val-to-test splitting when using explicit counts
        if is_counts and "split_val_to_test" not in kwargs:
            sig = inspect.signature(self._parse_available_splits)
            if "split_val_to_test" in sig.parameters:
                kwargs["split_val_to_test"] = False

        split_definitions = self._parse_available_splits(dataset_dir, **kwargs)
        if not split_definitions:
            existing_dirs = [
                d.name for d in dataset_dir.iterdir() if d.is_dir()
            ]
            raise ValueError(
                "No valid split directories found in dataset. "
                f"Found directories: {existing_dirs}."
            )
        if all(
            len(split_images) == 0
            for split_images in split_definitions.values()
        ):
            raise ValueError(
                "No samples were parsed from the discovered split directories."
            )

        original_splits: dict[str, Sequence[PathType]] = {
            split_name: split_definitions.get(split_name, [])
            for split_name in self.CANONICAL_SPLIT_NAMES
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

    def _parse_available_splits(
        self, dataset_dir: Path, **kwargs
    ) -> dict[str, list[Path]]:
        split_definitions: dict[str, list[Path]] = {}
        for split_name, split_kwargs in self.discover_dir_splits(
            dataset_dir
        ).items():
            split_definitions[split_name] = self._parse_split(
                **split_kwargs, **kwargs
            )
        return split_definitions

    def _apply_counts_to_splits(
        self,
        original_splits: dict[str, Sequence[PathType]],
        split_ratios: dict[str, int],
    ) -> dict[str, Sequence[PathType]]:
        """Apply count-based split requests to existing splits.

        Samples from each original split independently. If more samples
        are requested than available in a split, all available samples
        from that split are used.

        Args:
            original_splits: Existing split assignments.
            split_ratios: Requested counts for each split.

        Returns:
            Split names mapped to assigned images.

        """
        return self._sample_from_splits(original_splits, split_ratios)

    def _remove_unsplit_records(self) -> None:
        """Remove records that are not assigned to any split."""
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
        """Return unique images yielded by a dataset generator.

        Args:
            generator: Dataset record generator.

        Returns:
            Unique added image paths.

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

    def _warn_skipped_annotation(
        self,
        parser_issue: ParserIssue,
        reason: str,
        *,
        source: PathType | None = None,
        image: PathType | None = None,
        annotation_id: str | int | None = None,
    ) -> None:
        message = ParserIssueMessage(
            parser_issue=parser_issue,
            reason=reason,
            source=source,
            image=image,
            annotation_id=annotation_id,
        )
        if message in self._seen_parser_issue_messages:
            return

        self._seen_parser_issue_messages.add(message)
        self._parser_issue_messages.append(message)

        details = []
        if annotation_id is not None:
            details.append(f"annotation_id={annotation_id}")
        if source is not None:
            details.append(f"source={source}")
        if image is not None:
            details.append(f"image={image}")

        suffix = f" ({', '.join(details)})" if details else ""
        logger.warning(f"Skipping annotation: {reason}{suffix}")

    @staticmethod
    def _compare_stem_files(
        list1: Iterable[Path], list2: Iterable[Path]
    ) -> bool:
        """Compare sets of files by stem.

        Example:
            >>> BaseParser._compare_stem_files([Path("a.jpg"), Path("b.jpg")],
            ...                                [Path("a.xml"), Path("b.xml")])
            True
            >>> BaseParser._compare_stem_files([Path("a.jpg")], [Path("b.txt")])
            False

        Args:
            list1: First files to compare.
            list2: Second files to compare.

        Returns:
            Whether the non-empty file stem sets are equal.

        """
        set1 = {Path(f).stem for f in list1}
        set2 = {Path(f).stem for f in list2}
        return len(set1) > 0 and set1 == set2

    @staticmethod
    def _list_images(image_dir: Path) -> list[Path]:
        """List OpenCV-supported images in a directory.

        Args:
            image_dir: Directory with images.

        Returns:
            Supported image paths.

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
        """Add configured task names to generated records.

        Args:
            generator: Dataset record generator.

        Returns:
            Generator that yields records with task names applied.

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
