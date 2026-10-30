"""Dataset metadata, as stored in ``metadata.json``."""

from collections.abc import Callable, Iterable, Mapping
from typing import Literal, TypeVar

from pydantic import Field
from typing_extensions import TypedDict

from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.typing import BaseModelExtraForbid

from .source import LuxonisSource

T = TypeVar("T")


class Skeletons(TypedDict):
    """Keypoint skeleton metadata.

    Attributes:
        labels: Keypoint names in index order.
        edges: Keypoint graph edges as :math:`0`-based index pairs.

    """

    labels: list[str]
    edges: list[tuple[int, int]]


class Metadata(BaseModelExtraForbid):
    """Stored metadata for a Luxonis Data Format dataset.

    Attributes:
        source: Dataset source description.
        ldf_version: Luxonis Data Format version.
        classes: Class-index mappings per task.
        tasks: Task types per task name.
        skeletons: Keypoint skeleton definitions per task.
        categorical_encodings: Integer encodings for categorical metadata.
        metadata_types: Metadata value types per metadata task.
        parent_dataset: Optional identifier of the source dataset this
            dataset was derived from.

    """

    source: LuxonisSource | None
    ldf_version: str = str(LDF_VERSION)
    classes: dict[str, dict[str, int]] = Field(default_factory=dict)
    tasks: dict[str, list[str]] = Field(default_factory=dict)
    skeletons: dict[str, Skeletons] = Field(default_factory=dict)
    categorical_encodings: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )
    metadata_types: dict[str, Literal["float", "int", "str", "Category"]] = (
        Field(default_factory=dict)
    )
    parent_dataset: str | None = None

    def set_classes(
        self, classes: list[str] | dict[str, int], task: str
    ) -> None:
        """Set the classes of a task.

        Args:
            classes: Class names, or class IDs keyed by class name. Class
                names get IDs in case-insensitive alphabetical order, and
                ``"background"`` always gets ID :math:`0`.
            task: Task to update.

        """
        if isinstance(classes, list):
            self.classes[task] = {
                class_name: i
                for i, class_name in enumerate(self._sort_classes(classes))
            }
        else:
            self.classes[task] = classes

    def merge_with(self, other: "Metadata") -> "Metadata":
        """Merge two metadata objects together.

        Args:
            other: Metadata object to merge into this one.

        Returns:
            New metadata object containing merged classes, tasks,
            skeletons, categorical encodings, metadata types, and source
            information.

        Raises:
            ValueError: If the two metadata objects use different LDF
                versions.

        """
        if self.ldf_version != other.ldf_version:  # pragma: no cover
            raise ValueError(
                "Cannot merge metadata with different LDF versions"
            )

        if self.source is None:
            merged_source = other.source
        elif other.source is None:
            merged_source = self.source
        else:
            merged_source = self.source.merge_with(other.source)

        return Metadata(
            ldf_version=self.ldf_version,
            source=merged_source,
            classes=_merge_dicts(
                self.classes, other.classes, self._merge_class_maps
            ),
            tasks=_merge_dicts(
                self.tasks, other.tasks, lambda a, b: list(set(a + b))
            ),
            skeletons={**self.skeletons, **other.skeletons},
            categorical_encodings=_merge_dicts(
                self.categorical_encodings,
                other.categorical_encodings,
                lambda a, b: {**a, **b},
            ),
            metadata_types={**self.metadata_types, **other.metadata_types},
        )

    def _merge_class_maps(
        self, a: dict[str, int], b: dict[str, int]
    ) -> dict[str, int]:
        """Index the union of two class sets in sorted order."""
        all_classes = self._sort_classes(set(a) | set(b))
        return {class_name: i for i, class_name in enumerate(all_classes)}

    def _sort_classes(self, classes: Iterable[str]) -> list[str]:
        return sorted(
            classes,
            key=lambda x: (0, "") if x == "background" else (1, x.lower()),
        )


def _merge_dicts(
    a: Mapping[str, T], b: Mapping[str, T], combine: Callable[[T, T], T]
) -> dict[str, T]:
    """Union of two mappings; a key in both gets ``combine(a[k], b[k])``."""
    merged = {**a, **b}
    for key in a.keys() & b.keys():
        merged[key] = combine(a[key], b[key])
    return merged
