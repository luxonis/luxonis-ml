import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from itertools import chain
from pathlib import Path
from typing import Any, TypeAlias, cast

from loguru import logger
from semver.version import Version
from typing_extensions import Self

from luxonis_ml.data.datasets.annotation import DatasetRecord
from luxonis_ml.data.datasets.source import LuxonisSource
from luxonis_ml.data.utils.enums import ParserIssueMessage
from luxonis_ml.enums import DatasetType
from luxonis_ml.typing import PathType
from luxonis_ml.utils import AutoRegisterMeta, Registry

DATASETS_REGISTRY: Registry[type["BaseDataset"]] = Registry(name="datasets")


DatasetIterator: TypeAlias = Iterator[dict | DatasetRecord]


def _prepare_import_records(
    records: Iterator[tuple[str | None, dict | DatasetRecord]],
    *,
    task_name: str | dict[str, str] | None,
    selected_files: set[Path] | None,
    split_files: dict[str | None, dict[Path, None]],
) -> DatasetIterator:
    """Turn parser output into records ready for `BaseDataset.add`.

    Files are collected per split as the records stream past, which is why
    a parser no longer has to publish a file list before it starts: the
    only caller that needs one is `make_splits`, and that runs after every
    record has been added.

    Args:
        records: Split name and record, as produced by a parser.
        task_name: Optional task name, or class-to-task mapping.
        selected_files: When given, only records naming one of these files
            are kept.
        split_files: Filled with the ordered, unique files of each split.

    Yields:
        Records to add to the dataset.

    """
    if task_name is None:
        task_names = None
        unannotated_task_names: set[str] = set()
    elif isinstance(task_name, str):
        # A `defaultdict` is only populated on lookup, so the task names used
        # for records without annotations have to be collected up front.
        task_names = defaultdict(lambda: task_name)
        unannotated_task_names = {task_name}
    else:
        task_names = task_name
        unannotated_task_names = set(task_name.values())

    for split_name, raw_record in records:
        record = (
            DatasetRecord(**raw_record)
            if isinstance(raw_record, dict)
            else raw_record
        )
        if selected_files is not None and not any(
            Path(file).absolute() in selected_files
            for file in record.files.values()
        ):
            continue

        # A `dict` rather than a `set` so the files of a split keep the
        # order the parser emitted them in, which is the order the old
        # file-list pass produced.
        files_of_split = split_files.setdefault(split_name, {})
        for file in record.files.values():
            files_of_split[file] = None

        if task_names is None:
            yield record
            continue

        if record.annotation is None:
            for name in unannotated_task_names:
                yield record.model_copy(
                    update={"task_name": name},
                    deep=True,
                )
            continue

        class_name = record.annotation.class_name
        if class_name is not None:
            try:
                name = task_names[class_name]
            except KeyError:
                raise ValueError(
                    f"Class '{class_name}' not found in task names."
                ) from None
            # Only `task_name` changes, so a shallow copy is enough and avoids
            # duplicating masks and polygons for every record.
            record = record.model_copy(update={"task_name": name})
        yield record


def _record_files(item: dict | DatasetRecord) -> Iterator[PathType]:
    """Yield the files a record names."""
    if isinstance(item, DatasetRecord):
        yield from item.files.values()
    elif "file" in item:
        yield item["file"]
    elif "files" in item:
        yield from item["files"].values()


def _peek(records: DatasetIterator) -> DatasetIterator:
    """Return ``records``, raising if it yields nothing.

    Pulling the first record before anything is written keeps an empty
    source failing before a dataset is populated, which is what the old
    up-front file list used to guarantee.
    """
    first = next(records, None)
    if first is None:
        raise ValueError("No samples were parsed from the source.")
    return chain([first], records)


def _enumerate_by_parsing(
    plugin: Any,
    source: Path,
    layout: Any,
    parser_kwargs: dict[str, Any],
) -> dict[str | None, list[Path]]:
    """Collect each split's files by parsing and discarding the records.

    The fallback for a parser that cannot enumerate its files without
    parsing them. It costs a full extra parse, which is why it runs only
    for count-based `split_ratios`.
    """
    collected: dict[str | None, dict[Path, None]] = {}
    result = plugin.parse(source, layout, **parser_kwargs)
    for split_name, raw_record in result.records:
        files_of_split = collected.setdefault(split_name, {})
        for file in _record_files(raw_record):
            files_of_split[Path(file)] = None
    return {name: list(files) for name, files in collected.items()}


class BaseDataset(
    ABC, metaclass=AutoRegisterMeta, registry=DATASETS_REGISTRY, register=False
):
    """Base class for datasets in the Luxonis MLOps ecosystem."""

    _parser_issue_messages: list[ParserIssueMessage] = []

    @classmethod
    def import_dataset(
        cls,
        source: PathType,
        *,
        dataset_name: str | None = None,
        save_dir: Path | str | None = None,
        dataset_type: DatasetType | str | None = None,
        task_name: str | dict[str, str] | None = None,
        full_warnings: bool = False,
        split: str | None = None,
        random_split: bool = True,
        split_ratios: dict[str, float | int] | None = None,
        parser_kwargs: Mapping[str, Any] | None = None,
        **dataset_kwargs: Any,
    ) -> Self:
        """Import an external dataset using a registered parser plugin.

        The method is inherited by concrete dataset implementations and
        returns an instance of the class on which it is called.

        Args:
            source: Local path, supported remote URL, Roboflow reference, or
                Ultralytics Platform reference.
            dataset_name: Name of the dataset to create. By default, derive
                it from ``source``.
            save_dir: Directory used for downloaded or extracted sources.
            dataset_type: Registered parser type. When omitted, detect it.
            task_name: Optional task name, or class-to-task mapping.
            full_warnings: Whether to log every skipped annotation warning.
            split: Optional split to assign to every imported file.
            random_split: Whether a source without original splits should be
                split automatically.
            split_ratios: Ratios or counts for train, validation, and test.
                Splits omitted from a count-based mapping default to
                :math:`0`.
            parser_kwargs: Format-specific parser keyword arguments.
            dataset_kwargs: Arguments passed to the dataset constructor.

        Returns:
            Newly created and populated dataset.

        """
        from luxonis_ml.data.parsers.parser_plugin import (
            ParseIssueCollector,
            apply_counts_to_pool,
            apply_counts_to_splits,
            get_parser_plugin,
        )
        from luxonis_ml.data.parsers.source import prepare_source

        source_path, derived_name = prepare_source(source, save_dir)
        type_name = (
            dataset_type.value
            if isinstance(dataset_type, DatasetType)
            else dataset_type
        )
        plugin_type, _selected_type, layout = get_parser_plugin(
            source_path,
            type_name,
        )

        # Resolved before the dataset is created so that invalid arguments do
        # not leave an empty dataset behind.
        is_counts = split_ratios is not None and all(
            isinstance(value, int) for value in split_ratios.values()
        )
        count_ratios: dict[str, int] | None = None
        if is_counts:
            assert split_ratios is not None
            count_ratios = {
                name: int(split_ratios.get(name, 0))
                for name in ("train", "val", "test")
            }
            if sum(count_ratios.values()) == 0:
                raise ValueError(
                    "Count-based `split_ratios` must request at least one "
                    f"sample, got {split_ratios}."
                )

        resolved_name = (
            dataset_name
            or derived_name.replace(" ", "_").split(".", maxsplit=1)[0]
        )
        if not resolved_name:
            raise ValueError(
                f"Could not derive a dataset name from source '{source}'. "
                "Pass `dataset_name` explicitly."
            )
        dataset = cast(Any, cls)(
            dataset_name=resolved_name,
            **dataset_kwargs,
        )
        issues = ParseIssueCollector(full_warnings=full_warnings)
        plugin = plugin_type(issues)

        try:
            resolved_parser_kwargs = dict(parser_kwargs or {})
            if (
                split is None
                and is_counts
                and "split_val_to_test" not in resolved_parser_kwargs
                and "split_val_to_test"
                in inspect.signature(plugin.parse).parameters
            ):
                resolved_parser_kwargs["split_val_to_test"] = False

            has_original_splits = bool(layout.split_names)

            selected_splits: dict[str, Sequence[PathType]] | None = None
            selected_files: set[Path] | None = None
            if (
                split is None
                and count_ratios is not None
                and (has_original_splits or random_split)
            ):
                # The only feature that has to know the files before a
                # record is added. A parser that cannot enumerate them
                # cheaply pays a throwaway parse here, and only here.
                enumerated = plugin.enumerate_files(
                    source_path, layout, **resolved_parser_kwargs
                )
                if enumerated is None:
                    enumerated = _enumerate_by_parsing(
                        plugin, source_path, layout, resolved_parser_kwargs
                    )
                if has_original_splits:
                    selected_splits = apply_counts_to_splits(
                        {
                            name: files
                            for name, files in enumerated.items()
                            if name is not None
                        },
                        count_ratios,
                    )
                else:
                    selected_splits = apply_counts_to_pool(
                        [
                            file
                            for files in enumerated.values()
                            for file in files
                        ],
                        count_ratios,
                    )
                selected_files = {
                    Path(file).absolute()
                    for files in selected_splits.values()
                    for file in files
                }

            parsed = plugin.parse(
                source_path,
                layout,
                **resolved_parser_kwargs,
            )
            split_files: dict[str | None, dict[Path, None]] = {}
            records = _prepare_import_records(
                parsed.records,
                task_name=task_name,
                selected_files=selected_files,
                split_files=split_files,
            )
            dataset.add(_peek(records))

            parsed_splits = {
                name: list(files)
                for name, files in split_files.items()
                if name is not None
            }
            all_files = [
                file for files in split_files.values() for file in files
            ]

            for skeleton in parsed.skeletons.values():
                dataset.set_skeletons(
                    skeleton.get("labels"),
                    skeleton.get("edges"),
                )

            if split is not None:
                dataset.make_splits({split: all_files})
            elif selected_splits is not None:
                dataset.make_splits(selected_splits)
            elif split_ratios is not None:
                if has_original_splits or random_split:
                    if has_original_splits and not is_counts:
                        logger.warning(
                            "Using percentage-based split ratios will "
                            "redistribute and shuffle all samples across "
                            "splits. Original split boundaries will not be "
                            "preserved."
                        )
                    dataset.make_splits(split_ratios)
            elif has_original_splits:
                # A source with original splits defines all three, even the
                # ones it left empty: a train-only dataset still reports an
                # empty `val` and `test` rather than omitting them.
                original_splits: dict[str, Sequence[PathType]] = {
                    name: parsed_splits.get(name, [])
                    for name in ("train", "val", "test")
                }
                dataset.make_splits(original_splits)
            elif random_split:
                dataset.make_splits(None)

            logger.info("Dataset imported successfully.")
        except BaseException:
            # A parser streams its records, so a source it cannot finish
            # reading fails part-way through `add`, once some of it is
            # already written. Without this the caller is left with a
            # registered, half-populated dataset that looks importable.
            # Nothing here can be recovered by retrying, so the dataset
            # is removed and the original error propagates.
            with suppress(Exception):
                dataset.delete_dataset(delete_local=True)
            raise
        else:
            return dataset
        finally:
            issues.log_summary()
            dataset._parser_issue_messages = issues.messages

    def get_parser_issue_messages(self) -> list[ParserIssueMessage]:
        """Return issues collected during the most recent import."""
        return list(self._parser_issue_messages)

    @property
    @abstractmethod
    def identifier(self) -> str:
        """The unique identifier for the dataset."""
        ...

    @property
    @abstractmethod
    def version(self) -> Version:
        """The version of the underlying LDF."""
        ...

    @abstractmethod
    def set_tasks(self, tasks: dict[str, list[str]]) -> None:
        """Set dataset tasks.

        Args:
            tasks: Mapping from task names to task types.

        """
        ...

    @abstractmethod
    def get_tasks(self) -> dict[str, list[str]]:
        """Return task names and task types.

        Returns:
            Task types keyed by task name.

        """
        ...

    @abstractmethod
    def set_classes(
        self,
        classes: list[str] | dict[str, int],
        task: str | None = None,
    ) -> None:
        """Set classes for one or more tasks.

        Args:
            classes: Class names, or class IDs keyed by class name. If
                class names are provided, IDs are assigned
                alphabetically starting from :math:`0`. A class named
                ``"background"`` is always assigned ID :math:`0`.
            task: Optional task to update. If omitted, all tasks are
                updated.

        """
        ...

    @abstractmethod
    def get_classes(self) -> dict[str, dict[str, int]]:
        """Get class names and IDs per task.

        Returns:
            Mapping from class names to class IDs grouped by task name:

            .. python::

                {
                    "color": {"red": 0, "green": 1, "blue": 2},
                    "brand": {"audi": 0, "bmw": 1, "mercedes": 2},
                }

        """
        ...

    @abstractmethod
    def get_source_names(self) -> list[str]:
        """Return input source names for the dataset.

        Returns:
            Source names used to identify input data.

        """
        ...

    @abstractmethod
    def update_source(self, source: LuxonisSource) -> None:
        """Update the dataset source definition.

        Args:
            source: Source definition to store.

        """
        ...

    @abstractmethod
    def set_skeletons(
        self,
        labels: list[str] | None = None,
        edges: list[tuple[int, int]] | None = None,
        task: str | None = None,
    ) -> None:
        """Set keypoint skeleton semantics for tasks that use keypoints.

        For example:

        .. python::

            dataset.set_skeletons(
                labels=["right hand", "right shoulder", ...],
                edges=[[0, 1], [4, 5], ...]
            )

        Args:
            labels: Optional keypoint names.
            edges: Optional edges between keypoints.
            task: Optional task to update. If omitted, all keypoint tasks
                are updated.

        Raises:
            ValueError: If neither ``labels`` nor ``edges`` are provided.

        """
        ...

    @abstractmethod
    def get_skeletons(
        self,
    ) -> dict[str, tuple[list[str], list[tuple[int, int]]]]:
        """Return keypoint skeletons for each task.

        Returns:
            Keypoint labels and edges keyed by task name.

        """
        ...

    @abstractmethod
    def add(
        self, generator: DatasetIterator, batch_size: int = 1_000_000
    ) -> None:
        """Write annotations to parquet files.

        Args:
            generator: Iterator yielding ``DatasetRecord`` instances or
                dictionaries that can be converted to ``DatasetRecord``.
            batch_size: Number of records to buffer before processing.
                Lower values reduce peak memory usage.

        """
        ...

    @abstractmethod
    def make_splits(
        self,
        splits: dict[str, Sequence[PathType]]
        | dict[str, float]
        | tuple[float, float, float]
        | None = None,
        *,
        ratios: dict[str, float] | tuple[float, float, float] | None = None,
        definitions: dict[str, list[PathType]] | None = None,
        replace_old_splits: bool = False,
    ) -> None:
        """Generate dataset splits.

        Args:
            splits: Split definitions or ratios. Accepts explicit
                filepath lists, split ratios keyed by split name, or a
                ``(train, val, test)`` ratio tuple.
            ratios: Optional deprecated split ratios. Use ``splits``
                instead.
            definitions: Optional deprecated filepath split definitions.
                Use ``splits`` instead.
            replace_old_splits: Whether to replace existing split
                assignments instead of adding only new files.

        """
        ...

    @abstractmethod
    def delete_dataset(self) -> None:
        """Delete local files belonging to the dataset."""
        ...

    @staticmethod
    @abstractmethod
    def exists(dataset_name: str) -> bool:
        """Check whether a dataset exists.

        Args:
            dataset_name: Dataset name to check.

        Returns:
            ``True`` if the dataset exists, ``False`` otherwise.

        """
        ...

    def get_n_classes(self) -> dict[str, int]:
        """Return number of classes per task.

        Returns:
            Mapping from task names to class counts.

        """
        return {
            task_name: len(classes)
            for task_name, classes in self.get_classes().items()
        }

    def get_class_names(self) -> dict[str, list[str]]:
        """Return class names per task.

        Returns:
            Class names keyed by task name:

            .. python::

                {
                    "vehicles": ["red", "green", "blue"],
                    "brands": ["audi", "bmw", "mercedes"],
                }

        """
        return {
            task_name: list(classes.keys())
            for task_name, classes in self.get_classes().items()
        }

    def get_task_names(self) -> list[str]:
        """Return task names for the dataset.

        This is equivalent to `get_tasks` but returns only the task names.

        Returns:
            Task names.

        """
        return list(self.get_tasks().keys())

    def get_n_keypoints(self) -> dict[str, int]:
        """Return the number of keypoints for each task.

        Returns:
            Number of keypoints keyed by task name.

        """
        skeletons = self.get_skeletons()
        return {task: len(skeletons[task][0]) for task in skeletons}
