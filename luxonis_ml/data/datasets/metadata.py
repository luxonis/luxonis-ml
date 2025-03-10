from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

from typing_extensions import TypedDict

from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.utils.pydantic_utils import BaseModelExtraForbid

from .source import LuxonisSource


class Skeletons(TypedDict):
    labels: List[str]
    edges: List[Tuple[int, int]]


class Metadata(BaseModelExtraForbid):
    source: Optional[LuxonisSource]
    ldf_version: str = str(LDF_VERSION)
    classes: Dict[str, Dict[str, int]] = {}
    tasks: Dict[str, List[str]] = {}
    skeletons: Dict[str, Skeletons] = {}
    categorical_encodings: Dict[str, Dict[str, int]] = {}
    metadata_types: Dict[str, Literal["float", "int", "str", "Category"]] = {}
    parent_dataset: Optional[str] = None

    def set_classes(
        self, classes: Union[List[str], Dict[str, int]], task: str
    ) -> None:
        if isinstance(classes, list):
            self.classes[task] = {
                class_name: i
                for i, class_name in enumerate(self._sort_classes(classes))
            }
        else:
            self.classes[task] = classes

    def merge_with(self, other: "Metadata") -> "Metadata":
        """Merge two metadata objects together."""
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

    def _sort_classes(self, classes: Iterable[str]) -> List[str]:
        return sorted(
            classes,
            key=lambda x: (0, "") if x == "background" else (1, x.lower()),
        )
