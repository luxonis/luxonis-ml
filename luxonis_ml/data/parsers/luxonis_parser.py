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

from .parser_plugin import get_parser_plugin
from .source import prepare_source

T = TypeVar("T", str, None)


class LuxonisParser(Generic[T]):
    """Deprecated compatibility wrapper for dataset-owned imports.

    Use `LuxonisDataset.import_dataset` instead. The source is acquired and
    its format resolved during construction, so unsupported sources are
    rejected there and repeated `parse` calls neither download nor extract the
    source again.
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
        self._dataset_dir, derived_name = prepare_source(dataset_dir, save_dir)
        _plugin_type, self._dataset_type, _layout = get_parser_plugin(
            self._dataset_dir,
            dataset_type.value
            if isinstance(dataset_type, DatasetType)
            else dataset_type,
        )
        self._dataset_name = (
            dataset_name
            or derived_name.replace(" ", "_").split(".", maxsplit=1)[0]
        )
        self._dataset_class: type[BaseDataset] = (
            LuxonisDataset
            if dataset_plugin is None
            else DATASETS_REGISTRY.get(dataset_plugin)
        )
        self._save_dir = save_dir
        self._task_name = task_name
        self._full_warnings = full_warnings
        self._dataset_kwargs = kwargs
        self._dataset: BaseDataset | None = None
        self._issues: list[ParserIssueMessage] = []

    @overload
    def parse(self: "LuxonisParser[str]", **kwargs: Any) -> BaseDataset: ...

    @overload
    def parse(
        self: "LuxonisParser[None]", **kwargs: Any
    ) -> LuxonisDataset: ...

    def parse(self, **kwargs: Any) -> BaseDataset:
        """Import the configured source through the dataset-owned API."""
        split = kwargs.pop("split", None)
        random_split = kwargs.pop("random_split", True)
        split_ratios = kwargs.pop("split_ratios", None)

        # Cleared rather than left holding the previous parse: a failed
        # call must not answer `get_parser_issue_messages` with what an
        # earlier one collected.
        self._dataset = None
        self._issues.clear()
        self._dataset = self._dataset_class.import_dataset(
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
            _issue_sink=self._issues,
            **self._dataset_kwargs,
        )
        return self._dataset

    def get_parser_issue_messages(self) -> list[ParserIssueMessage]:
        """Return issues collected during the most recent parse."""
        return self._get_parser_issue_messages()

    def _get_parser_issue_messages(self) -> list[ParserIssueMessage]:
        if self._dataset is not None:
            return self._dataset.get_parser_issue_messages()
        # A parse that failed never returned a dataset to read these
        # from, and the issues collected before it gave up are exactly
        # what a caller handling that failure is asking about.
        return list(self._issues)
