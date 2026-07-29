import warnings
from pathlib import Path
from typing import Any, Generic, TypeVar, overload

from luxonis_ml.data.datasets import (
    DATASETS_REGISTRY,
    BaseDataset,
    LuxonisDataset,
)
from luxonis_ml.data.utils.enums import ParserIssueMessage
from luxonis_ml.enums import DatasetType

T = TypeVar("T", str, None)


class LuxonisParser(Generic[T]):
    """Deprecated compatibility wrapper for dataset-owned imports.

    Use `LuxonisDataset.import_dataset` instead. Source acquisition, parser
    detection, and dataset construction are intentionally deferred until
    `parse` so this class remains a thin argument adapter.
    """

    def __init__(
        self,
        dataset_dir: str,
        *,
        dataset_name: str | None = None,
        save_dir: Path | str | None = None,
        dataset_plugin: T = None,
        dataset_type: DatasetType | str | None = None,
        task_name: str | dict[str, str] | None = None,
        full_warnings: bool = False,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "`LuxonisParser` is deprecated and will be removed in a future "
            "release. Use `LuxonisDataset.import_dataset(...)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._dataset_dir = dataset_dir
        self._dataset_name = dataset_name
        self._save_dir = save_dir
        self._dataset_plugin = dataset_plugin
        self._dataset_type = dataset_type
        self._task_name = task_name
        self._full_warnings = full_warnings
        self._dataset_kwargs = kwargs
        self._dataset: BaseDataset | None = None

    @overload
    def parse(self: "LuxonisParser[str]", **kwargs: Any) -> BaseDataset: ...

    @overload
    def parse(
        self: "LuxonisParser[None]", **kwargs: Any
    ) -> LuxonisDataset: ...

    def parse(self, **kwargs: Any) -> BaseDataset:
        """Import the configured source through the dataset-owned API."""
        dataset_class: type[BaseDataset]
        if self._dataset_plugin is None:
            dataset_class = LuxonisDataset
        else:
            dataset_class = DATASETS_REGISTRY.get(self._dataset_plugin)

        split = kwargs.pop("split", None)
        random_split = kwargs.pop("random_split", True)
        split_ratios = kwargs.pop("split_ratios", None)
        self._dataset = dataset_class.import_dataset(
            self._dataset_dir,
            dataset_name=self._dataset_name,
            save_dir=self._save_dir,
            dataset_type=self._dataset_type,
            task_name=self._task_name,
            full_warnings=self._full_warnings,
            split=split,
            random_split=random_split,
            split_ratios=split_ratios,
            parser_kwargs=kwargs,
            **self._dataset_kwargs,
        )
        return self._dataset

    def get_parser_issue_messages(self) -> list[ParserIssueMessage]:
        """Return issues collected during the most recent parse."""
        return self._get_parser_issue_messages()

    def _get_parser_issue_messages(self) -> list[ParserIssueMessage]:
        if self._dataset is None:
            return []
        return self._dataset.get_parser_issue_messages()
