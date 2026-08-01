"""Typed sample-selection rules shared by dataset visualization commands."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from luxonis_ml.ldf import DatasetRecord, Detection
from luxonis_ml.typing import Params, ParamValue, PrimitiveType

InspectionAnnotationType: TypeAlias = Literal[
    "array",
    "boundingbox",
    "classification",
    "instance_segmentation",
    "keypoints",
    "metadata",
    "segmentation",
]
"""Annotation families accepted by the inspector's type filter."""

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
        """Deduplicated task scope, or ``None`` when every task is selected."""
        names = tuple(dict.fromkeys(self.task_name or []))
        return frozenset(names) if names else None

    def validate(
        self,
        *,
        available_tasks: Iterable[str] = (),
        available_classes: Iterable[str] = (),
    ) -> None:
        """Reject requested task or class names absent from the supplied scope."""
        tasks = tuple(dict.fromkeys(available_tasks))
        requested_tasks = tuple(dict.fromkeys(self.task_name or []))
        unknown_tasks = [task for task in requested_tasks if task not in tasks]
        if unknown_tasks:
            unknown = ", ".join(repr(task) for task in unknown_tasks)
            available = ", ".join(repr(task) for task in tasks)
            raise ValueError(
                f"Unknown task name(s): {unknown}. "
                f"Available task names: {available or '(none)'}."
            )

        classes = frozenset(name.strip() for name in available_classes)
        requested_classes = tuple(dict.fromkeys(self.class_name or []))
        unknown_classes = [
            name for name in requested_classes if name not in classes
        ]
        if unknown_classes:
            unknown = ", ".join(repr(name) for name in unknown_classes)
            available = ", ".join(repr(name) for name in sorted(classes))
            raise ValueError(
                f"Unknown class name(s): {unknown}. "
                f"Available class names: {available or '(none)'}."
            )

    def query(self) -> "InspectionQuery":
        """Build the immutable matcher consumed by inspect and compare."""
        search = self.search.strip() if self.search else ""
        return InspectionQuery(
            class_names=frozenset(dict.fromkeys(self.class_name or [])),
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
    category; different categories are combined with ``and``. Filters select
    whole samples rather than pruning matching detections from a selected
    sample, preserving visual context.
    """

    class_names: frozenset[str] = field(default_factory=frozenset)
    class_name_mode: NameFilterMode = "include"
    annotation_types: frozenset[InspectionAnnotationType] = field(
        default_factory=frozenset
    )
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
        for name, value in (
            ("--min-instances", self.min_instances),
            ("--max-instances", self.max_instances),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative.")
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


def _record_detections(
    records: Iterable[DatasetRecord],
) -> Iterator[Detection]:
    """Yield every top-level and nested detection from a record collection."""
    for record in records:
        if record.annotation is not None:
            yield from _detection_tree(record.annotation)


def _detection_tree(detection: Detection) -> Iterator[Detection]:
    """Yield a detection followed by all nested sub-detections."""
    yield detection
    for child in detection.sub_detections.values():
        yield from _detection_tree(child)


def _has_annotations(detections: Sequence[Detection]) -> bool:
    """Whether any detection carries a label, array, or metadata value."""
    return any(detection.get_task_types() for detection in detections)


def _matches_classes(
    detections: Sequence[Detection], requested: frozenset[str]
) -> bool:
    """Whether any detection belongs to a requested class."""
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
    """Whether any detection carries a requested annotation family."""
    present = {
        task_type
        for detection in detections
        for task_type in detection.get_task_types()
    }
    present.update(extra)
    return any(
        (
            any(task_type.startswith("metadata/") for task_type in present)
            if requested_type == "metadata"
            else requested_type in present
        )
        for requested_type in requested
    )


def _matches_instance_bounds(
    detections: Sequence[Detection],
    minimum: int | None,
    maximum: int | None,
) -> bool:
    """Whether the number of spatial instances falls within both bounds."""
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
    """Whether any detection has ``score``/``confidence`` metadata above cutoff."""
    return any(
        confidence is not None and confidence >= minimum
        for detection in detections
        for confidence in (_detection_confidence(detection),)
    )


def _detection_confidence(detection: Detection) -> float | None:
    """Read a numeric confidence from conventional detection metadata keys."""
    for key in ("score", "confidence"):
        value = detection.metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _matches_search(
    query: str,
    records: Mapping[str, DatasetRecord],
    detections: Sequence[Detection],
    sample_metadata: Params,
) -> bool:
    """Case-insensitive substring search over identity, labels, and metadata."""
    needle = query.casefold()
    values = [*records]
    values.extend(
        detection.class_name
        for detection in detections
        if detection.class_name is not None
    )
    values.extend(_params_strings(sample_metadata))
    for detection in detections:
        values.extend(str(key) for key in detection.metadata)
        values.extend(str(value) for value in detection.metadata.values())
    return any(needle in value.casefold() for value in values)


def _params_strings(params: Params) -> Iterator[str]:
    """Flatten a top-level metadata mapping into searchable text."""
    for key, value in params.items():
        yield key
        yield from _metadata_strings(value)


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
    current: Mapping[str, ParamValue] | Mapping[PrimitiveType, ParamValue] = (
        metadata
    )
    value: ParamValue = None
    for index, part in enumerate(path):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        value = current[part]
        if index == len(path) - 1:
            return True, value
        if not isinstance(value, Mapping):
            return False, None
        current = value
    return False, None


def _metadata_equal(value: ParamValue, expected: str) -> bool:
    """Compare a scalar metadata value to its command-line representation."""
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, str)
    ):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Numbers must compare numerically, not by repr: a stored 0.5 has to
        # match ``--metadata-filter score 0.50``, and a stored 1.0 has to match
        # ``... count 1``.
        try:
            return float(value) == float(expected)
        except ValueError:
            return False
    return str(value).casefold() == expected.casefold()
