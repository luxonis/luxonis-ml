"""Sample-selection rules shared by the dataset visualization commands."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from luxonis_ml.ldf import DatasetRecord, Detection
from luxonis_ml.typing import Params, ParamValue, TaskType

InspectionAnnotationType: TypeAlias = TaskType | Literal["metadata"]
"""Annotation families accepted by the inspector's type filter.

Every `TaskType`, plus ``"metadata"`` for the whole ``metadata/<key>`` family.
"""

NameFilterMode: TypeAlias = Literal["include", "exclude"]
"""Whether named tasks or classes are required or rejected."""


@dataclass(frozen=True, slots=True)
class SampleFilterConfig:
    """Shared flat CLI options for selecting dataset samples.

    Repeated task, class, and annotation-type values are alternatives within
    their category. Different categories and repeated metadata expressions are
    combined with ``and``. A selected sample retains all matching-task
    annotations for visual context.

    Attributes:
        task_name: Complete task names to include or exclude.
        task_name_mode: Include only or exclude the named tasks.
        class_name: Class names to include or exclude when selecting samples.
        class_name_mode: Include samples containing a named class or exclude
            samples containing any named class.
        annotation_type: Annotation families of which at least one must occur.
        metadata_filter: Exact metadata predicates. Each occurrence takes
            separate ``path value`` tokens; dotted paths address nested sample
            metadata.
        min_confidence: Minimum numeric ``score`` or ``confidence`` metadata on
            at least one detection.
        min_instances: Minimum number of spatial instances.
        max_instances: Maximum number of spatial instances.
        unlabeled_only: Select samples without annotations.
        search: Case-insensitive substring matched against filenames, task and
            class names, and sample or detection metadata.

    """

    task_name: list[str] | None = None
    task_name_mode: NameFilterMode = "include"
    class_name: list[str] | None = None
    class_name_mode: NameFilterMode = "include"
    annotation_type: list[InspectionAnnotationType] | None = None
    metadata_filter: list[tuple[str, str]] | None = None
    min_confidence: float | None = None
    min_instances: int | None = None
    max_instances: int | None = None
    unlabeled_only: bool = False
    search: str | None = None

    @property
    def task_filter(self) -> frozenset[str] | None:
        """Deduplicated task scope, or ``None`` when every task is selected.

        The set alone does not say whether the named tasks are wanted or
        rejected. Use `accepts_task` to apply ``task_name_mode``.
        """
        return frozenset(self.task_name) if self.task_name else None

    def accepts_task(self, task_name: str) -> bool:
        """Whether one task passes the include or exclude task filter."""
        names = self.task_filter
        if names is None:
            return True
        return (task_name in names) == (self.task_name_mode == "include")

    def validate(
        self,
        *,
        available_tasks: Iterable[str] = (),
        available_classes: Iterable[str] = (),
    ) -> None:
        """Reject requested task or class names outside the given scope."""
        _reject_unknown_names(
            "task name", self.task_name or (), available_tasks
        )
        _reject_unknown_names(
            "class name",
            self._requested_classes(),
            (name.strip() for name in available_classes),
        )

    def query(self) -> "InspectionQuery":
        """Build the immutable matcher consumed by inspect and compare."""
        search = self.search.strip() if self.search else ""
        return InspectionQuery(
            class_names=frozenset(self._requested_classes()),
            class_name_mode=self.class_name_mode,
            annotation_types=frozenset(self.annotation_type or []),
            metadata=tuple(
                MetadataPredicate.from_pair(path, expected)
                for path, expected in self.metadata_filter or []
            ),
            min_confidence=self.min_confidence,
            min_instances=self.min_instances,
            max_instances=self.max_instances,
            unlabeled_only=self.unlabeled_only,
            search=search or None,
        )

    def _requested_classes(self) -> tuple[str, ...]:
        """Class names trimmed the way stored class names are."""
        return tuple(name.strip() for name in self.class_name or [])


@dataclass(frozen=True, slots=True)
class MetadataPredicate:
    """One exact metadata comparison supplied as separate path and value tokens.

    Dotted paths address nested sample metadata. A single-segment path also
    matches per-detection metadata, including nested sub-detections.
    """

    path: tuple[str, ...]
    expected: str

    @classmethod
    def from_pair(cls, path: str, expected: str) -> "MetadataPredicate":
        """Build a predicate from the CLI's separate path and value tokens."""
        parts = tuple(part.strip() for part in path.split(".") if part.strip())
        if not parts:
            raise ValueError("Metadata filter paths cannot be empty.")
        return cls(path=parts, expected=expected.strip())

    def matches(
        self,
        sample_metadata: Params,
        detections: Sequence[Detection],
    ) -> bool:
        """Whether sample or detection metadata contains the expected value."""
        found, value = _metadata_path(sample_metadata, self.path)
        if found and _metadata_equal(value, self.expected):
            return True
        if len(self.path) != 1:
            return False
        key = self.path[0]
        return any(
            key in detection.metadata
            and _metadata_equal(detection.metadata[key], self.expected)
            for detection in detections
        )


@dataclass(frozen=True, slots=True)
class InspectionQuery:
    """Conjunctive sample filters for dataset inspection and comparison.

    Repeated class names and annotation types are alternatives within their
    category; different categories are combined with ``and``. A filter selects
    a whole sample. It does not remove the other detections from that sample,
    so the visual context stays complete.
    """

    class_names: frozenset[str] = frozenset()
    class_name_mode: NameFilterMode = "include"
    annotation_types: frozenset[InspectionAnnotationType] = frozenset()
    metadata: tuple[MetadataPredicate, ...] = ()
    min_confidence: float | None = None
    min_instances: int | None = None
    max_instances: int | None = None
    unlabeled_only: bool = False
    search: str | None = None

    def __post_init__(self) -> None:
        """Validate bounds and incompatible filters."""
        if self.min_confidence is not None and not (
            0.0 <= self.min_confidence <= 1.0
        ):
            raise ValueError("--min-confidence must be between 0 and 1.")
        if self.min_instances is not None and self.min_instances < 0:
            raise ValueError("--min-instances must be non-negative.")
        if self.max_instances is not None and self.max_instances < 0:
            raise ValueError("--max-instances must be non-negative.")
        if (
            self.min_instances is not None
            and self.max_instances is not None
            and self.min_instances > self.max_instances
        ):
            raise ValueError(
                "--min-instances cannot be greater than --max-instances."
            )
        if self.unlabeled_only and (
            (self.class_names and self.class_name_mode == "include")
            or self.annotation_types
            or self.min_confidence is not None
            or (self.min_instances is not None and self.min_instances > 0)
        ):
            raise ValueError(
                "--unlabeled-only cannot be combined with an inclusive class "
                "filter, annotation type, confidence, or a positive "
                "minimum-instance filter."
            )

    @property
    def active(self) -> bool:
        """Whether the query contains at least one sample-level filter."""
        return bool(
            self.class_names
            or self.annotation_types
            or self.metadata
            or self.min_confidence is not None
            or self.min_instances is not None
            or self.max_instances is not None
            or self.unlabeled_only
            or self.search
        )

    def matches(
        self,
        records: Mapping[str, DatasetRecord],
        sample_metadata: Params,
        *,
        extra_annotation_types: frozenset[
            InspectionAnnotationType
        ] = frozenset(),
    ) -> bool:
        """Return whether one converted loader sample satisfies this query."""
        detections = list(_record_detections(records.values()))
        if self.unlabeled_only and (
            extra_annotation_types or _has_annotations(detections)
        ):
            return False
        if self.class_names:
            matches_classes = _matches_classes(detections, self.class_names)
            if matches_classes != (self.class_name_mode == "include"):
                return False
        if self.annotation_types and not _matches_annotation_types(
            detections,
            self.annotation_types,
            extra_annotation_types,
        ):
            return False
        if not _matches_instance_bounds(
            detections, self.min_instances, self.max_instances
        ):
            return False
        if self.min_confidence is not None and not _matches_confidence(
            detections, self.min_confidence
        ):
            return False
        if any(
            not predicate.matches(sample_metadata, detections)
            for predicate in self.metadata
        ):
            return False
        return not self.search or _matches_search(
            self.search, records, detections, sample_metadata
        )


def _reject_unknown_names(
    kind: str, requested: Iterable[str], available: Iterable[str]
) -> None:
    known = frozenset(available)
    unknown = [name for name in dict.fromkeys(requested) if name not in known]
    if not unknown:
        return
    missing = ", ".join(repr(name) for name in unknown)
    scope = ", ".join(repr(name) for name in sorted(known)) or "(none)"
    raise ValueError(
        f"Unknown {kind}(s): {missing}. Available {kind}s: {scope}."
    )


def _record_detections(
    records: Iterable[DatasetRecord],
) -> Iterator[Detection]:
    """Yield every top-level and nested detection of a record collection."""
    for record in records:
        if record.annotation is not None:
            yield from _detection_tree(record.annotation)


def _detection_tree(detection: Detection) -> Iterator[Detection]:
    """Yield a detection followed by all of its nested sub-detections."""
    yield detection
    for child in detection.sub_detections.values():
        yield from _detection_tree(child)


def _has_annotations(detections: Sequence[Detection]) -> bool:
    return any(detection.get_task_types() for detection in detections)


def _matches_classes(
    detections: Sequence[Detection], requested: frozenset[str]
) -> bool:
    return any(
        detection.class_name is not None
        and detection.class_name.strip() in requested
        for detection in detections
    )


def _matches_annotation_types(
    detections: Sequence[Detection],
    requested: frozenset[InspectionAnnotationType],
    extra: frozenset[InspectionAnnotationType],
) -> bool:
    """Whether any detection carries a requested annotation family.

    ``"metadata"`` stands for the whole family, because
    `Detection.get_task_types` reports one ``metadata/<key>`` type per key.
    """
    present = {
        task_type
        for detection in detections
        for task_type in detection.get_task_types()
    }
    present.update(extra)
    if present & (requested - {"metadata"}):
        return True
    return "metadata" in requested and any(
        task_type.startswith("metadata/") for task_type in present
    )


def _matches_instance_bounds(
    detections: Sequence[Detection],
    minimum: int | None,
    maximum: int | None,
) -> bool:
    count = sum(
        detection.boundingbox is not None
        or detection.keypoints is not None
        or detection.instance_segmentation is not None
        for detection in detections
    )
    return (minimum is None or count >= minimum) and (
        maximum is None or count <= maximum
    )


def _matches_confidence(
    detections: Sequence[Detection], minimum: float
) -> bool:
    scores = (_detection_confidence(detection) for detection in detections)
    return any(score is not None and score >= minimum for score in scores)


def _detection_confidence(detection: Detection) -> float | None:
    for key in ("score", "confidence"):
        score = _as_number(detection.metadata.get(key))
        if score is not None:
            return score
    return None


def _matches_search(
    query: str,
    records: Mapping[str, DatasetRecord],
    detections: Sequence[Detection],
    sample_metadata: Params,
) -> bool:
    needle = query.casefold()
    return any(
        needle in text.casefold()
        for text in _searchable_text(records, detections, sample_metadata)
    )


def _searchable_text(
    records: Mapping[str, DatasetRecord],
    detections: Sequence[Detection],
    sample_metadata: Params,
) -> Iterator[str]:
    """Yield the task names, class names, and metadata the search covers."""
    yield from records
    yield from _metadata_strings(sample_metadata)
    for detection in detections:
        if detection.class_name is not None:
            yield detection.class_name
        for key, value in detection.metadata.items():
            yield key
            yield str(value)


def _metadata_strings(value: ParamValue) -> Iterator[str]:
    """Flatten recursive metadata keys and scalar values into search text."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _metadata_strings(child)
        return
    if isinstance(value, Sequence) and not isinstance(value, str):
        for child in value:
            yield from _metadata_strings(child)
        return
    yield str(value)


def _metadata_path(
    metadata: Params, path: tuple[str, ...]
) -> tuple[bool, ParamValue]:
    """Resolve a dotted path without conflating a missing value with ``None``."""
    current: ParamValue = metadata
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _metadata_equal(value: ParamValue, expected: str) -> bool:
    """Compare a scalar metadata value to its command-line representation."""
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, str)
    ):
        return False
    number = _as_number(value)
    if number is None:
        return str(value).casefold() == expected.casefold()
    # Numbers must compare numerically, not by repr: a stored 0.5 has to match
    # ``--metadata-filter score 0.50``, and a stored 1.0 has to match
    # ``... count 1``.
    try:
        return number == float(expected)
    except ValueError:
        return False


def _as_number(value: ParamValue) -> float | None:
    """Read a metadata value as a number, refusing ``bool``.

    ``bool`` subclasses ``int``, but a filter compares ``True`` to the word
    "true", never to the number 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
