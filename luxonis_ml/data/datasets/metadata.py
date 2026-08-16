from collections.abc import Iterable
from typing import Literal

from loguru import logger
from pydantic import AliasChoices, Field

from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.ldf import KeypointMetadata
from luxonis_ml.typing import BaseModelExtraForbid

from .source import LuxonisSource


class Metadata(BaseModelExtraForbid):
    """Stored metadata for a Luxonis Data Format dataset.

    Attributes:
        source: Dataset source description.
        ldf_version: Luxonis Data Format version.
        classes: Class-index mappings per task.
        tasks: Task types per task name.
        keypoint_metadata: Keypoint definitions per task. A dataset written
            before LDF 2.2 stores them under ``"skeletons"``, which is
            still accepted.
        categorical_encodings: Integer encodings for categorical metadata.
        metadata_types: Metadata value types per metadata task.
        parent_dataset: Optional identifier of the source dataset this
            dataset was derived from.

    """

    source: LuxonisSource | None
    ldf_version: str = str(LDF_VERSION)
    classes: dict[str, dict[str, int]] = {}
    tasks: dict[str, list[str]] = {}
    keypoint_metadata: dict[str, KeypointMetadata] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("keypoint_metadata", "skeletons"),
    )
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
            New metadata object containing merged classes, tasks, keypoint
            metadata, categorical encodings, metadata types, and source
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

        mine, theirs = self.keypoint_metadata, other.keypoint_metadata
        merged_keypoint_metadata = {**mine, **theirs}
        for task in set(mine) & set(theirs):
            merged_keypoint_metadata[task] = _merge_keypoint_metadata(
                mine[task], theirs[task], task
            )

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
            keypoint_metadata=merged_keypoint_metadata,
            categorical_encodings=merged_categorical_encodings,
            metadata_types=merged_metadata_types,  # type: ignore
        )

    def _sort_classes(self, classes: Iterable[str]) -> list[str]:
        return sorted(
            classes,
            key=lambda x: (0, "") if x == "background" else (1, x.lower()),
        )


def _merge_keypoint_metadata(
    mine: KeypointMetadata, theirs: KeypointMetadata, task: str
) -> KeypointMetadata:
    """Merge the keypoint metadata of one task from two datasets.

    The dataset that is merged in wins. A field it leaves empty comes
    from the other dataset. `edges`, `flip_pairs` and `sigmas` hold
    indices into `labels`, so they only carry over if both datasets list
    the same labels in the same order.

    Args:
        mine: Keypoint metadata of the target dataset.
        theirs: Keypoint metadata of the dataset that is merged in.
        task: Name of the task, used in the warnings.

    Returns:
        The merged keypoint metadata.

    """
    if mine.labels != theirs.labels:
        logger.warning(
            f"Task '{task}' has different keypoint labels in the two "
            "datasets being merged. Keeping the ones from the dataset "
            "being merged in. They now describe the keypoints of the "
            "other dataset too, which gave them a different order."
        )
        return theirs

    conflicts = [
        field
        for field in KeypointMetadata.model_fields
        if getattr(mine, field)
        and getattr(theirs, field)
        and getattr(mine, field) != getattr(theirs, field)
    ]
    if conflicts:
        logger.warning(
            f"Task '{task}' has a different {', '.join(conflicts)} in the "
            "two datasets being merged. Keeping the one from the dataset "
            "being merged in."
        )
    return theirs.model_copy(
        update={
            field: getattr(mine, field)
            for field in KeypointMetadata.model_fields
            if not getattr(theirs, field)
        }
    )
