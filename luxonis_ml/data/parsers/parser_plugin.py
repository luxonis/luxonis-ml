import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, ClassVar, overload

from loguru import logger

from luxonis_ml.data.datasets.annotation import DatasetRecord
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.utils.enums import ParserIssue, ParserIssueMessage
from luxonis_ml.typing import PathType
from luxonis_ml.utils import Registry

PARSERS_REGISTRY: Registry[type["ParserPlugin"]] = Registry(name="parsers")


@dataclass
class ParsedDataset:
    """Data-only result produced by a parser plugin.

    ``files`` and ``splits`` must be complete before ``records`` is consumed.
    This lets dataset importers select count-based subsets without relying on
    implementation-specific dataset internals.
    """

    records: DatasetIterator
    skeletons: dict[str, dict[str, Any]]
    files: list[Path]
    splits: dict[str, list[Path]] | None = None


@dataclass
class ParseIssueCollector:
    """Collect and report non-fatal issues encountered during import."""

    full_warnings: bool = False
    warning_limit: int = 10
    _messages: list[ParserIssueMessage] = field(default_factory=list)
    _seen: set[ParserIssueMessage] = field(default_factory=set)
    _logged_warnings: int = 0
    _suppressed_warnings: int = 0
    _counts_by_reason: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    @property
    def messages(self) -> list[ParserIssueMessage]:
        """Return a defensive copy of collected issues."""
        return list(self._messages)

    def warn(
        self,
        parser_issue: ParserIssue,
        reason: str,
        *,
        source: PathType | None = None,
        image: PathType | None = None,
        annotation_id: str | int | None = None,
    ) -> None:
        """Record a skipped annotation and log it according to the cap."""
        message = ParserIssueMessage(
            parser_issue=parser_issue,
            reason=reason,
            source=source,
            image=image,
            annotation_id=annotation_id,
        )
        if message in self._seen:
            return

        self._seen.add(message)
        self._messages.append(message)
        self._counts_by_reason[reason] += 1

        details = []
        if annotation_id is not None:
            details.append(f"annotation_id={annotation_id}")
        if source is not None:
            details.append(f"source={source}")
        if image is not None:
            details.append(f"image={image}")
        suffix = f" ({', '.join(details)})" if details else ""

        if self.full_warnings or self._logged_warnings < self.warning_limit:
            logger.warning(f"Skipping annotation: {reason}{suffix}")
            self._logged_warnings += 1
        else:
            self._suppressed_warnings += 1

    def log_summary(self) -> None:
        """Log the summary for warnings hidden by the configured cap."""
        if self.full_warnings or self._suppressed_warnings == 0:
            return

        logger.warning(
            "Skipped logging "
            f"{self._suppressed_warnings} additional warnings. "
            "Enable the `--log-all-warnings` flag to see the full list."
        )
        for reason, count in sorted(
            self._counts_by_reason.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            logger.warning(f"Skipped annotations: {reason} ({count} records)")


class ParserPlugin(ABC):
    """Recognize and parse one or more external dataset formats.

    Plugins produce records and metadata only. Dataset construction, mutation,
    task mapping, split assignment, and issue retention belong to
    `BaseDataset.import_dataset`.
    """

    dataset_types: ClassVar[tuple[str, ...]] = ()

    def __init__(self, issues: ParseIssueCollector) -> None:
        self._issues = issues

    @classmethod
    @abstractmethod
    def supports(cls, source: Path) -> bool:
        """Return whether this plugin recognizes ``source``."""
        ...

    @abstractmethod
    def parse(
        self,
        source: Path,
        *,
        dataset_type: str,
        **kwargs: Any,
    ) -> ParsedDataset:
        """Parse ``source`` without mutating a dataset."""
        ...

    def _warn_skipped_annotation(
        self,
        parser_issue: ParserIssue,
        reason: str,
        *,
        source: PathType | None = None,
        image: PathType | None = None,
        annotation_id: str | int | None = None,
    ) -> None:
        self._issues.warn(
            parser_issue,
            reason,
            source=source,
            image=image,
            annotation_id=annotation_id,
        )

    @staticmethod
    def _get_added_images(generator: DatasetIterator) -> list[Path]:
        return list(
            dict.fromkeys(
                Path(value)
                for item in generator
                for value in _record_files(item)
            )
        )

    @staticmethod
    def _compare_stem_files(
        list1: Iterable[Path], list2: Iterable[Path]
    ) -> bool:
        set1 = {Path(file).stem for file in list1}
        set2 = {Path(file).stem for file in list2}
        return bool(set1) and set1 == set2

    @staticmethod
    def _list_images(image_dir: Path) -> list[Path]:
        supported_formats = {
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
            image
            for image in image_dir.glob("*")
            if image.suffix in supported_formats
        ]


class SplitParserPlugin(ParserPlugin):
    """Data-only plugin helper for formats organized into named splits."""

    split_names: ClassVar[tuple[str, ...]] = ("train", "valid", "test")

    @staticmethod
    @abstractmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        """Return parse arguments when ``split_path`` is recognized."""
        ...

    @abstractmethod
    def _parse_split(self, **kwargs: Any) -> ParsedDataset:
        """Parse one recognized input split."""
        ...

    @staticmethod
    def _canonicalize_split_name(split_name: str) -> str:
        return "val" if split_name in {"valid", "validation"} else split_name

    @classmethod
    def discover_splits(cls, source: Path) -> dict[str, dict[str, Any]]:
        if not source.is_dir():
            return {}

        discovered: dict[str, dict[str, Any]] = {}
        for split_name in cls.split_names:
            split_kwargs = cls.validate_split(source / split_name)
            if split_kwargs is not None:
                discovered[cls._canonicalize_split_name(split_name)] = (
                    split_kwargs
                )
        return discovered

    @classmethod
    def supports(cls, source: Path) -> bool:
        return source.is_dir() and (
            bool(cls.discover_splits(source))
            or cls.validate_split(source) is not None
        )

    def parse(
        self,
        source: Path,
        *,
        dataset_type: str,
        **kwargs: Any,
    ) -> ParsedDataset:
        del dataset_type
        split_inputs = self.discover_splits(source)
        if not split_inputs:
            split_kwargs = self.validate_split(source)
            if split_kwargs is None:
                raise ValueError(
                    f"Dataset {source} is not in the expected format for "
                    f"{type(self).__name__}."
                )
            return self._parse_split(**split_kwargs, **kwargs)

        outputs = {
            split_name: self._parse_split(**split_kwargs, **kwargs)
            for split_name, split_kwargs in split_inputs.items()
        }
        return combine_split_outputs(outputs)


def combine_split_outputs(
    outputs: dict[str, ParsedDataset],
) -> ParsedDataset:
    """Combine per-split parser outputs into one streaming result."""
    files = list(
        dict.fromkeys(
            file for output in outputs.values() for file in output.files
        )
    )
    skeletons: dict[str, dict[str, Any]] = {}
    for output in outputs.values():
        skeletons.update(output.skeletons)

    return ParsedDataset(
        records=chain.from_iterable(
            output.records for output in outputs.values()
        ),
        skeletons=skeletons,
        files=files,
        splits={
            split_name: outputs[split_name].files
            if split_name in outputs
            else []
            for split_name in ("train", "val", "test")
        },
    )


@overload
def register_parser_plugin(
    plugin: None = None,
    *,
    force: bool = False,
) -> Callable[[type[ParserPlugin]], type[ParserPlugin]]: ...


@overload
def register_parser_plugin(
    plugin: type[ParserPlugin],
    *,
    force: bool = False,
) -> type[ParserPlugin]: ...


def register_parser_plugin(
    plugin: type[ParserPlugin] | None = None,
    *,
    force: bool = False,
) -> type[ParserPlugin] | Callable[[type[ParserPlugin]], type[ParserPlugin]]:
    """Register a parser under each of its dataset type identifiers."""

    def register(cls: type[ParserPlugin]) -> type[ParserPlugin]:
        if not cls.dataset_types:
            raise ValueError("Parser plugins must declare `dataset_types`.")
        for dataset_type in cls.dataset_types:
            PARSERS_REGISTRY.register(
                module=cls,
                name=dataset_type,
                force=force,
            )
        return cls

    if plugin is None:
        return register
    return register(plugin)


def get_parser_plugin(
    source: Path,
    dataset_type: str | None,
) -> tuple[type[ParserPlugin], str]:
    """Resolve an explicit parser type or auto-detect one."""
    if dataset_type is not None:
        plugin = PARSERS_REGISTRY.get(dataset_type)
        if not plugin.supports(source):
            raise ValueError(
                f"Dataset {source} is not in the expected format for the "
                f"{dataset_type} parser."
            )
        return plugin, dataset_type

    matches: list[type[ParserPlugin]] = []
    for plugin in dict.fromkeys(PARSERS_REGISTRY.values()):
        if plugin.supports(source):
            matches.append(plugin)

    if not matches:
        raise ValueError(
            f"Dataset {source} is not in expected format for any registered "
            "parser."
        )
    if len(matches) > 1:
        matched_names = {plugin.dataset_types[0] for plugin in matches}
        if matched_names == {"yolov6", "yolov8"}:
            yolov6 = next(
                plugin
                for plugin in matches
                if plugin.dataset_types[0] == "yolov6"
            )
            if issubclass(yolov6, SplitParserPlugin):
                discovered = yolov6.discover_splits(source)
                if "train" in discovered and len(discovered) >= 2:
                    return yolov6, "yolov6"
            raise ValueError(
                "Dataset layout is compatible with multiple parsers when "
                "only a subset of splits is present. This layout is "
                "ambiguous between YOLOv6 and YOLOv8. Please specify "
                "dataset_type."
            )
        matched_types = ", ".join(
            plugin.dataset_types[0] for plugin in matches
        )
        raise ValueError(
            "Dataset layout is compatible with multiple parsers: "
            f"{matched_types}. Please specify `dataset_type`."
        )

    plugin = matches[0]
    return plugin, plugin.dataset_types[0]


def _record_files(item: dict | DatasetRecord) -> Iterator[PathType]:
    if isinstance(item, DatasetRecord):
        yield from item.files.values()
    elif "file" in item:
        yield item["file"]
    elif "files" in item:
        yield from item["files"].values()


def apply_counts_to_pool(
    images: Sequence[PathType],
    split_ratios: dict[str, int],
) -> dict[str, Sequence[PathType]]:
    """Distribute count requests across a single image pool."""
    total_requested = sum(split_ratios.values())
    shuffled = list(images)
    random.shuffle(shuffled)

    if total_requested > len(shuffled):
        logger.warning(
            f"Requested {total_requested} total samples, but only "
            f"{len(shuffled)} available. Filling splits by priority "
            "(most requested first)."
        )
        sorted_splits = sorted(
            ("train", "val", "test"),
            key=lambda split: split_ratios[split],
            reverse=True,
        )
        sampled: dict[str, Sequence[PathType]] = {}
        offset = 0
        for split_name in sorted_splits:
            count = min(
                split_ratios[split_name],
                len(shuffled) - offset,
            )
            sampled[split_name] = shuffled[offset : offset + count]
            offset += count
        return sampled

    sampled = {}
    offset = 0
    for split_name in ("train", "val", "test"):
        count = split_ratios[split_name]
        sampled[split_name] = shuffled[offset : offset + count]
        offset += count
    return sampled


def apply_counts_to_splits(
    original_splits: Mapping[str, Sequence[PathType]],
    split_ratios: dict[str, int],
) -> dict[str, Sequence[PathType]]:
    """Sample requested counts independently from original splits."""
    sampled: dict[str, Sequence[PathType]] = {}
    for split_name in ("train", "val", "test"):
        requested = split_ratios[split_name]
        available = list(original_splits.get(split_name, ()))
        if requested == 0:
            sampled[split_name] = []
        elif requested >= len(available):
            if requested > len(available):
                logger.warning(
                    f"Requested {requested} samples for '{split_name}' "
                    f"split, but only {len(available)} available. Using all "
                    f"{len(available)} samples."
                )
            sampled[split_name] = available
        else:
            sampled[split_name] = random.sample(available, requested)
    return sampled
