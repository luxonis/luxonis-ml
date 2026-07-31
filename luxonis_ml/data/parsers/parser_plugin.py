import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TypeAlias, overload

from loguru import logger

from luxonis_ml.data.datasets.annotation import DatasetRecord
from luxonis_ml.data.datasets.base_dataset import (
    DatasetIterator,
)
from luxonis_ml.data.utils.enums import ParserIssue, ParserIssueMessage
from luxonis_ml.typing import PathType
from luxonis_ml.utils import Registry

PARSERS_REGISTRY: Registry[type["ParserPlugin"]] = Registry(name="parsers")


#: A record together with the split it belongs to, or ``None`` when the
#: source carries no split information.
SplitRecord: TypeAlias = tuple[str | None, "dict | DatasetRecord"]


@dataclass(frozen=True)
class Layout:
    """What a parser recognized in a source.

    Detection is what discovers a layout, and parsing is handed the result
    instead of rediscovering it, so a source is inspected once per import.

    Attributes:
        splits: Parse arguments for each split, keyed by canonical split
            name. A source that is not organized into splits is a single
            entry keyed ``None``.

    """

    splits: dict[str | None, dict[str, Any]]

    @property
    def split_names(self) -> list[str]:
        """The named splits, excluding a source parsed as a whole."""
        return [name for name in self.splits if name is not None]


@dataclass
class ParseResult:
    """Data-only result produced by a parser plugin.

    ``records`` is a single-pass iterator: a parser walks its source once
    and tags each record with the split it belongs to, rather than
    publishing a file list a caller has to be given up front.

    Attributes:
        records: Split name and record, streamed in one pass.
        skeletons: Keypoint skeleton metadata keyed by task name. Complete
            once ``records`` is exhausted; a parser that already knows its
            skeletons may fill it before streaming starts.

    """

    records: Iterator[SplitRecord]
    skeletons: dict[str, dict[str, Any]]


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
        #: Skeleton metadata, filled whenever a parse learns it. Read by
        #: the importer once the records are exhausted, so a parser that
        #: only discovers its skeletons while streaming may fill it late.
        self._skeletons: dict[str, dict[str, Any]] = {}

    @classmethod
    @abstractmethod
    def detect(cls, source: Path) -> Layout | None:
        """Return the layout of ``source``, or ``None`` if unrecognized.

        Whatever recognizing the source revealed - which splits it has,
        where their images and annotations live - belongs in the returned
        layout, because it is handed back to `parse` instead of being
        discovered a second time.
        """
        ...

    @abstractmethod
    def parse(
        self,
        source: Path,
        layout: Layout,
        **kwargs: Any,
    ) -> ParseResult:
        """Parse ``source`` without mutating a dataset."""
        ...

    def enumerate_files(
        self,
        source: Path,
        layout: Layout,
        **kwargs: Any,
    ) -> dict[str | None, list[Path]] | None:
        """List the files of each split without parsing annotations.

        Only count-based `split_ratios` need the files up front, to pick a
        subset before anything is imported. Returning ``None`` says this
        parser cannot answer more cheaply than parsing, and the importer
        falls back to a throwaway parse - for that one case, not for
        every import.

        Args:
            source: Root of the dataset.
            layout: Layout returned by `detect`.
            kwargs: Format-specific parse arguments.

        Returns:
            Files per split, or ``None`` if the parser cannot enumerate
            them cheaply.

        """
        del source, layout, kwargs
        return None

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
    def _compare_stem_files(
        list1: Iterable[Path], list2: Iterable[Path]
    ) -> bool:
        set1 = {Path(file).stem for file in list1}
        set2 = {Path(file).stem for file in list2}
        return bool(set1) and set1 == set2

    @staticmethod
    def _list_images(image_dir: Path) -> list[Path]:
        """List the images of a directory that OpenCV can read.

        The suffix is matched case-insensitively, so a source naming its
        images `.JPG` - which the exact-case match this replaced skipped
        without a word - is imported like any other.
        """
        supported_formats = {
            ".bmp",
            ".dib",
            ".jpeg",
            ".jpg",
            ".jpe",
            ".jp2",
            ".png",
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
            if image.suffix.lower() in supported_formats
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
    def _split_records(self, **kwargs: Any) -> DatasetIterator:
        """Stream the records of one recognized input split."""
        ...

    def _split_files(self, **kwargs: Any) -> list[Path] | None:
        """List the files of one split without parsing its annotations.

        Override where a split's files are a directory listing or an
        index the parser reads anyway. Returning ``None`` means only a
        parse can answer.
        """
        del kwargs
        return None

    @staticmethod
    def _canonicalize_split_name(split_name: str) -> str:
        return "val" if split_name in {"valid", "validation"} else split_name

    @classmethod
    def detect(cls, source: Path) -> Layout | None:
        if not source.is_dir():
            return None

        discovered: dict[str | None, dict[str, Any]] = {}
        for split_name in cls.split_names:
            split_kwargs = cls.validate_split(source / split_name)
            if split_kwargs is not None:
                discovered[cls._canonicalize_split_name(split_name)] = (
                    split_kwargs
                )
        if discovered:
            return Layout(discovered)

        split_kwargs = cls.validate_split(source)
        if split_kwargs is None:
            return None
        return Layout({None: split_kwargs})

    def parse(
        self,
        source: Path,
        layout: Layout,
        **kwargs: Any,
    ) -> ParseResult:
        del source

        def records() -> Iterator[SplitRecord]:
            for split_name, split_kwargs in layout.splits.items():
                for record in self._split_records(**split_kwargs, **kwargs):
                    yield split_name, record

        return ParseResult(records(), self._skeletons)

    def enumerate_files(
        self,
        source: Path,
        layout: Layout,
        **kwargs: Any,
    ) -> dict[str | None, list[Path]] | None:
        del source
        enumerated: dict[str | None, list[Path]] = {}
        for split_name, split_kwargs in layout.splits.items():
            files = self._split_files(**split_kwargs, **kwargs)
            if files is None:
                return None
            enumerated[split_name] = files
        return enumerated


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
        # Checked up front so a collision on a later type does not leave the
        # earlier ones registered.
        if not force:
            taken = [
                dataset_type
                for dataset_type in cls.dataset_types
                if dataset_type in PARSERS_REGISTRY
            ]
            if taken:
                raise KeyError(
                    f"Parser types {taken} are already registered in the "
                    f"`{PARSERS_REGISTRY.name}` registry. Pass `force=True` "
                    "to override them."
                )
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
) -> tuple[type[ParserPlugin], str, Layout]:
    """Resolve an explicit parser type or auto-detect one.

    Returns the plugin, its dataset type, and the layout detection found,
    so that parsing does not have to inspect the source again.
    """
    if dataset_type is not None:
        plugin = PARSERS_REGISTRY.get(dataset_type)
        layout = plugin.detect(source)
        if layout is None:
            raise ValueError(
                f"Dataset {source} is not in the expected format for the "
                f"{dataset_type} parser."
            )
        return plugin, dataset_type, layout

    matches: list[tuple[type[ParserPlugin], Layout]] = []
    for plugin in dict.fromkeys(PARSERS_REGISTRY.values()):
        layout = plugin.detect(source)
        if layout is not None:
            matches.append((plugin, layout))

    if not matches:
        raise ValueError(
            f"Dataset {source} is not in expected format for any registered "
            "parser."
        )
    if len(matches) > 1:
        best_match = _resolve_by_split_coverage(matches)
        if best_match is None:
            matched_types = ", ".join(
                plugin.dataset_types[0] for plugin, _ in matches
            )
            raise ValueError(
                "Dataset layout is compatible with multiple parsers: "
                f"{matched_types}. Please specify `dataset_type`."
            )
        return best_match[0], best_match[0].dataset_types[0], best_match[1]

    plugin, layout = matches[0]
    return plugin, plugin.dataset_types[0], layout


def _resolve_by_split_coverage(
    matches: Sequence[tuple[type[ParserPlugin], Layout]],
) -> tuple[type[ParserPlugin], Layout] | None:
    """Return the plugin that recognized the most splits, if unique.

    Layouts differing only in how splits are named — ``images/valid`` for
    YOLOv6 against ``images/val`` for Ultralytics YOLOv8 — are recognized by
    several plugins, but only the matching one recognizes every split that is
    actually present. Returns ``None`` when the layout stays ambiguous.

    The counts come from the layouts detection already produced, so
    resolving an ambiguity costs no further look at the source.
    """
    counts = [
        (len(layout.split_names), plugin, layout) for plugin, layout in matches
    ]

    best_count = max(count for count, _, _ in counts)
    best = [
        (plugin, layout)
        for count, plugin, layout in counts
        if count == best_count
    ]
    if best_count == 0 or len(best) != 1:
        return None
    return best[0]


def apply_counts_to_pool(
    images: Sequence[PathType],
    split_ratios: dict[str, int],
) -> dict[str, Sequence[PathType]]:
    """Distribute count requests across a single image pool.

    Splits missing from ``split_ratios`` are treated as :math:`0`.
    """
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
            key=lambda split: split_ratios.get(split, 0),
            reverse=True,
        )
        sampled: dict[str, Sequence[PathType]] = {}
        offset = 0
        for split_name in sorted_splits:
            count = min(
                split_ratios.get(split_name, 0),
                len(shuffled) - offset,
            )
            sampled[split_name] = shuffled[offset : offset + count]
            offset += count
        return sampled

    sampled = {}
    offset = 0
    for split_name in ("train", "val", "test"):
        count = split_ratios.get(split_name, 0)
        sampled[split_name] = shuffled[offset : offset + count]
        offset += count
    return sampled


def apply_counts_to_splits(
    original_splits: Mapping[str, Sequence[PathType]],
    split_ratios: dict[str, int],
) -> dict[str, Sequence[PathType]]:
    """Sample requested counts independently from original splits.

    Splits missing from ``split_ratios`` are treated as :math:`0`.
    """
    sampled: dict[str, Sequence[PathType]] = {}
    for split_name in ("train", "val", "test"):
        requested = split_ratios.get(split_name, 0)
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
