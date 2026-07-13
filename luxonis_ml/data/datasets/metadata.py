from collections.abc import Iterable
from typing import Literal

from typing_extensions import TypedDict

from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.typing import BaseModelExtraForbid

from .source import LuxonisSource


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
    classes: dict[str, dict[str, int]] = {}
    tasks: dict[str, list[str]] = {}
    skeletons: dict[str, Skeletons] = {}
    categorical_encodings: dict[str, dict[str, int]] = {}
    metadata_types: dict[str, Literal["float", "int", "str", "Category"]] = {}
    parent_dataset: str | None = None

    def set_classes(
        self, classes: list[str] | dict[str, int], task: str
    ) -> None:
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

        merged_classes = {}
        for key in set(self.classes) | set(other.classes):
            if key in self.classes and key in other.classes:
                all_classes = self._sort_classes(
                    set(self.classes[key]) | set(other.classes[key])
                )
                merged_classes[key] = {
                    class_name: i for i, class_name in enumerate(all_classes)
                }
            elif key in self.classes:
                merged_classes[key] = self.classes[key]
            else:
                merged_classes[key] = other.classes[key]

        merged_tasks = {}
        for key in set(self.tasks) | set(other.tasks):
            if key in self.tasks and key in other.tasks:
                merged_tasks[key] = list(
                    set(self.tasks[key] + other.tasks[key])
                )
            elif key in self.tasks:
                merged_tasks[key] = self.tasks[key]
            else:
                merged_tasks[key] = other.tasks[key]

        merged_skeletons = {**self.skeletons, **other.skeletons}

        merged_categorical_encodings = {}
        for key in set(self.categorical_encodings) | set(
            other.categorical_encodings
        ):
            if (
                key in self.categorical_encodings
                and key in other.categorical_encodings
            ):
                merged_inner = self.categorical_encodings[key].copy()
                merged_inner.update(other.categorical_encodings[key])
                merged_categorical_encodings[key] = merged_inner
            elif key in self.categorical_encodings:
                merged_categorical_encodings[key] = self.categorical_encodings[
                    key
                ]
            else:
                merged_categorical_encodings[key] = (
                    other.categorical_encodings[key]
                )

        merged_metadata_types = {**self.metadata_types, **other.metadata_types}
        if self.source is None and other.source is not None:
            merged_source = other.source
        elif self.source is not None and other.source is None:
            merged_source = self.source
        elif self.source is not None and other.source is not None:
            merged_source = self.source.merge_with(other.source)
        else:
            merged_source = None

        return Metadata(
            ldf_version=self.ldf_version,
            source=merged_source,
            classes=merged_classes,
            tasks=merged_tasks,
            skeletons=merged_skeletons,
            categorical_encodings=merged_categorical_encodings,
            metadata_types=merged_metadata_types,  # type: ignore
        )

    def _sort_classes(self, classes: Iterable[str]) -> list[str]:
        return sorted(
            classes,
            key=lambda x: (0, "") if x == "background" else (1, x.lower()),
        )
