"""`BaseDataset.import_dataset` behaviour that is not parser-specific."""

import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from luxonis_ml.data import (
    DatasetIterator,
    Layout,
    LuxonisDataset,
    LuxonisLoader,
    ParseResult,
    ParserPlugin,
    register_parser_plugin,
)
from luxonis_ml.data.datasets.annotation import (
    DatasetRecord,
    Detection,
)
from luxonis_ml.data.datasets.base_dataset import (
    BaseDataset,
    _delete_replaces_dataset,
    _prepare_import_records,
)
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.enums import DatasetType
from tests.test_data.parsers.helpers import _write_yolov8_split
from tests.test_data.utils import create_image


def test_split_parser_creates_default_splits(dataset_name: str, tempdir: Path):
    class_dir = tempdir / "flat_cls"
    image_dir = class_dir / "class_a"
    image_dir.mkdir(parents=True)
    create_image(0, image_dir)

    dataset = LuxonisDataset.import_dataset(
        str(class_dir),
        dataset_name=dataset_name,
        dataset_type=DatasetType.CLSDIR,
        delete_local=True,
    )
    try:
        splits = dataset.get_splits()
        assert splits is not None
        assert set(splits) == {"train", "val", "test"}
        assert sum(len(group_ids) for group_ids in splits.values()) == 1

        loader = LuxonisLoader(dataset)
        next(iter(loader))
    finally:
        dataset.delete_dataset(delete_local=True)


def test_count_split_filters_unselected_records(
    dataset_name: str,
    tempdir: Path,
):
    class_dir = tempdir / "counted_cls" / "class_a"
    class_dir.mkdir(parents=True)
    for index in range(5):
        create_image(index, class_dir)

    dataset = LuxonisDataset.import_dataset(
        str(class_dir.parent),
        dataset_name=dataset_name,
        dataset_type=DatasetType.CLSDIR,
        split_ratios={"train": 2, "val": 1, "test": 1},
        delete_local=True,
    )
    try:
        assert len(dataset) == 4
        splits = dataset.get_splits()
        assert splits is not None
        assert {name: len(ids) for name, ids in splits.items()} == {
            "train": 2,
            "val": 1,
            "test": 1,
        }
    finally:
        dataset.delete_dataset(delete_local=True)


def _collect_dataset_records(records: DatasetIterator) -> list[DatasetRecord]:
    """Collect records that `_prepare_import_records` has already parsed."""
    collected = []
    for record in records:
        assert isinstance(record, DatasetRecord)
        collected.append(record)
    return collected


def test_prepare_import_records_keeps_unannotated_records_in_any_order(
    tempdir: Path,
):
    """Records without annotations must survive a string ``task_name``.

    Regression: a string ``task_name`` was wrapped in an empty
    ``defaultdict``, which only materializes keys on lookup, so the fan-out
    set for annotation-less records was empty until some annotated record had
    already been seen. Background images were silently dropped, and which ones
    depended on iteration order — hence both orderings are checked here.
    """
    unannotated = DatasetRecord(
        files={"image": create_image(0, tempdir)}, annotation=None
    )
    annotated = DatasetRecord(
        files={"image": create_image(1, tempdir)},
        annotation=Detection.model_validate({"class": "budgie"}),
    )

    for records in ([unannotated, annotated], [annotated, unannotated]):
        prepared = _collect_dataset_records(
            _prepare_import_records(
                iter([(None, record) for record in records]),
                task_name="birds",
                selected_files=None,
                split_files={},
            )
        )
        assert len(prepared) == 2
        assert {record.task_name for record in prepared} == {"birds"}

    # A class-to-task mapping instead fans an annotation-less record out over
    # every distinct task name, so it is not lost from any of them either.
    prepared = _collect_dataset_records(
        _prepare_import_records(
            iter([(None, record) for record in [unannotated]]),
            task_name={"budgie": "birds", "dog": "mammals"},
            selected_files=None,
            split_files={},
        )
    )
    assert {record.task_name for record in prepared} == {"birds", "mammals"}


def test_prepare_import_records_does_not_copy_annotations(tempdir: Path):
    """Assigning a task name must not duplicate annotation payloads.

    Regression: the task name was applied with
    ``model_copy(update=..., deep=True)``, deep-copying every polygon list and
    mask array once per record. Only ``task_name`` changes, so a shallow copy
    is enough; the annotation object is shared and the input record is left
    untouched.
    """
    record = DatasetRecord(
        files={"image": create_image(0, tempdir)},
        annotation=Detection.model_validate({"class": "budgie"}),
    )

    (prepared,) = _collect_dataset_records(
        _prepare_import_records(
            iter([(None, record) for record in [record]]),
            task_name="birds",
            selected_files=None,
            split_files={},
        )
    )

    assert prepared.task_name == "birds"
    assert prepared.annotation is record.annotation
    assert record.task_name != "birds"


@pytest.mark.parametrize(
    ("split_ratios", "expected_sizes"),
    [
        ({"train": 2}, {"train": 2, "val": 0, "test": 0}),
        ({"val": 1, "test": 1}, {"train": 0, "val": 1, "test": 1}),
    ],
)
def test_count_split_ratios_may_omit_splits(
    dataset_name: str,
    tempdir: Path,
    split_ratios: dict[str, float | int],
    expected_sizes: dict[str, int],
):
    """Count-based ``split_ratios`` may name only some of the splits.

    Regression: the count helpers indexed ``split_ratios["train"]``,
    ``["val"]`` and ``["test"]`` unconditionally, so a partial mapping raised a
    bare ``KeyError`` — after the dataset had already been created on disk.
    Percentage-based ratios always allowed partial mappings. Splits left out of
    the mapping are treated as :math:`0`.
    """
    dataset_dir = tempdir / "yolo_counts"
    _write_yolov8_split(dataset_dir / "train", range(4))
    _write_yolov8_split(dataset_dir / "valid", range(4, 6))
    _write_yolov8_split(dataset_dir / "test", range(6, 8))
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="yolov8",
        split_ratios=split_ratios,
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        splits = dataset.get_splits()
        assert splits is not None
        assert {
            name: len(group_ids) for name, group_ids in splits.items()
        } == expected_sizes
    finally:
        dataset.delete_dataset(delete_local=True)


@pytest.mark.parametrize(
    "split_ratios", [{"train": 0, "val": 0, "test": 0}, {}]
)
def test_zero_count_split_ratios_fail_before_creating_dataset(
    dataset_name: str,
    tempdir: Path,
    split_ratios: dict[str, float | int],
):
    """Counts selecting nothing must fail loudly and leave nothing behind.

    Regression: zero counts filtered out every record, so ``make_splits``
    raised ``FileNotFoundError: Dataset is empty`` — but only after the dataset
    had been created and registered, leaving an orphaned empty dataset on
    disk. An empty mapping hit the same path, because ``all()`` over no values
    classifies it as count-based.
    """
    dataset_dir = tempdir / "yolo_zero_counts"
    _write_yolov8_split(dataset_dir / "train", range(2))
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    with pytest.raises(ValueError, match="must request at least one sample"):
        LuxonisDataset.import_dataset(
            str(dataset_dir),
            dataset_name=dataset_name,
            dataset_type="yolov8",
            split_ratios=split_ratios,
            delete_local=True,
            save_dir=tempdir,
        )

    assert not LuxonisDataset.exists(dataset_name)


def test_prepare_import_records_collects_files_per_split(tempdir: Path):
    """Files are collected from the stream, not published up front.

    This is what replaced the `files` list every parser used to build with
    a second walk over its source: the importer needs the file list only
    for `make_splits`, which runs after every record has been added, so it
    can be accumulated as the records go past.
    """
    first = create_image(0, tempdir)
    second = create_image(1, tempdir)
    records = [
        ("train", DatasetRecord(files={"image": first}, annotation=None)),
        ("train", DatasetRecord(files={"image": first}, annotation=None)),
        ("val", DatasetRecord(files={"image": second}, annotation=None)),
    ]

    split_files: dict[str | None, dict[Path, None]] = {}
    stream = _prepare_import_records(
        iter(records),
        task_name=None,
        selected_files=None,
        split_files=split_files,
    )

    # Nothing is known before the stream is consumed, which is exactly the
    # constraint the old contract could not express.
    assert split_files == {}

    consumed = list(stream)
    assert len(consumed) == 3
    assert {name: list(files) for name, files in split_files.items()} == {
        "train": [first.absolute()],
        "val": [second.absolute()],
    }


def test_prepare_import_records_skips_files_outside_the_selection(
    tempdir: Path,
):
    """A count-based subset never reaches the dataset, nor the file list."""
    kept = create_image(2, tempdir)
    dropped = create_image(3, tempdir)
    records = [
        ("train", DatasetRecord(files={"image": kept}, annotation=None)),
        ("train", DatasetRecord(files={"image": dropped}, annotation=None)),
    ]

    split_files: dict[str | None, dict[Path, None]] = {}
    consumed = list(
        _prepare_import_records(
            iter(records),
            task_name=None,
            selected_files={kept.absolute()},
            split_files=split_files,
        )
    )

    assert len(consumed) == 1
    assert {name: list(files) for name, files in split_files.items()} == {
        "train": [kept.absolute()]
    }


def test_failed_import_leaves_no_dataset_behind(
    dataset_name: str, tempdir: Path
):
    """A source that fails part-way must not leave a dataset registered.

    Regression: parsers stream their records, so a source that cannot be
    read to the end fails from inside `add`, after some of it has already
    been written. The dataset was created before parsing began, so the
    caller was left with a registered, half-populated dataset that looked
    importable. Nothing about that state is recoverable, so the import
    removes it again.
    """
    dataset_dir = tempdir / "broken_clsdir"
    class_dir = dataset_dir / "train" / "bird"
    class_dir.mkdir(parents=True)
    create_image(0, class_dir)
    # Not an image: the parser fails on it once the records stream.
    (class_dir / "img_1.jpg").write_text("not an image")

    def failing_import() -> None:
        LuxonisDataset.import_dataset(
            str(dataset_dir),
            dataset_name=dataset_name,
            dataset_type="clsdir",
            delete_local=True,
        )

    with pytest.MonkeyPatch.context() as patch:
        # `clsdir` itself tolerates undecodable files, so the failure is
        # injected where any parser's records are turned into a dataset.
        patch.setattr(
            LuxonisDataset,
            "add",
            lambda self, generator, **kwargs: (_ for _ in ()).throw(
                ValueError("boom")
            ),
        )
        with pytest.raises(ValueError, match="boom"):
            failing_import()

    assert dataset_name not in LuxonisDataset.list_datasets()


def test_failed_import_keeps_a_pre_existing_dataset(
    dataset_name: str, tempdir: Path
):
    """A failed import must not delete a dataset it only opened.

    Regression: the failure handler deleted the dataset unconditionally, but
    the constructor opens an existing dataset of that name rather than
    replacing it. Importing into a name that was already in use therefore
    destroyed everything that had been imported before, and because the
    handler caught ``BaseException``, so did pressing Ctrl-C.
    """
    existing = LuxonisDataset(dataset_name, delete_local=True)
    existing.add(
        iter(
            [
                DatasetRecord(
                    files={"image": create_image(0, tempdir)},
                    annotation=Detection.model_validate({"class": "budgie"}),
                )
            ]
        )
    )
    existing.make_splits({"train": 1.0})
    assert len(existing) == 1

    dataset_dir = tempdir / "second_import"
    class_dir = dataset_dir / "train" / "bird"
    class_dir.mkdir(parents=True)
    create_image(1, class_dir)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                LuxonisDataset,
                "add",
                lambda self, generator, **kwargs: (_ for _ in ()).throw(
                    ValueError("boom")
                ),
            )
            with pytest.raises(ValueError, match="boom"):
                LuxonisDataset.import_dataset(
                    str(dataset_dir),
                    dataset_name=dataset_name,
                    dataset_type="clsdir",
                )

        assert LuxonisDataset.exists(dataset_name)
        assert len(LuxonisDataset(dataset_name)) == 1
    finally:
        if LuxonisDataset.exists(dataset_name):
            LuxonisDataset(dataset_name).delete_dataset(delete_local=True)


def test_interrupted_import_keeps_what_it_wrote(
    dataset_name: str, tempdir: Path
):
    """Ctrl-C must not delete the part of the import that succeeded.

    Regression: the failure handler caught ``BaseException``, so a
    ``KeyboardInterrupt`` was answered by deleting the dataset locally and
    remotely. Interrupting a long import is a request to stop, not a
    request to throw away everything already parsed and uploaded, and
    unlike a parse error it is not a state the caller has to be protected
    from. Only ``Exception`` triggers the cleanup.
    """
    dataset_dir = tempdir / "interrupted_import"
    class_dir = dataset_dir / "train" / "bird"
    class_dir.mkdir(parents=True)
    create_image(0, class_dir)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                LuxonisDataset,
                "add",
                lambda self, generator, **kwargs: (_ for _ in ()).throw(
                    KeyboardInterrupt()
                ),
            )
            with pytest.raises(KeyboardInterrupt):
                LuxonisDataset.import_dataset(
                    str(dataset_dir),
                    dataset_name=dataset_name,
                    dataset_type="clsdir",
                    delete_local=True,
                )

        assert LuxonisDataset.exists(dataset_name)
    finally:
        if LuxonisDataset.exists(dataset_name):
            LuxonisDataset(dataset_name).delete_dataset(delete_local=True)


def test_percentage_split_ratios_fail_before_creating_dataset(
    dataset_name: str, tempdir: Path
):
    """Ratios that do not sum to 1 must be rejected before anything runs.

    Regression: only count-based ratios were validated up front. A typo in
    percentage ratios was caught by ``make_splits``, which runs after every
    record has been written and every image uploaded - and the failure
    handler then deleted the whole freshly created dataset. The check is
    the one ``make_splits`` makes, moved to where nothing has been written
    yet.
    """
    dataset_dir = tempdir / "yolo_bad_ratios"
    _write_yolov8_split(dataset_dir / "train", range(2))
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    with pytest.MonkeyPatch.context() as patch:
        # Reaching `add` at all means the source was parsed and written
        # before the ratios were looked at, which is the regression.
        patch.setattr(
            LuxonisDataset,
            "add",
            lambda self, generator, **kwargs: pytest.fail(
                "the import ran before the ratios were validated"
            ),
        )
        with pytest.raises(ValueError, match=r"Ratios must sum to 1\.0"):
            LuxonisDataset.import_dataset(
                str(dataset_dir),
                dataset_name=dataset_name,
                dataset_type="yolov8",
                split_ratios={"train": 0.8, "val": 0.1, "test": 0.2},
                delete_local=True,
                save_dir=tempdir,
            )

    assert not LuxonisDataset.exists(dataset_name)


@pytest.mark.parametrize(
    ("dataset_kwargs", "replaces"),
    [
        ({}, False),
        ({"delete_local": True}, True),
        ({"delete_remote": True}, False),
        ({"bucket_storage": "gcs", "delete_local": True}, False),
        ({"bucket_storage": "gcs", "delete_remote": True}, True),
        ({"bucket_storage": BucketStorage.GCS, "delete_local": True}, False),
    ],
)
def test_only_a_delete_that_reaches_the_dataset_claims_it(
    dataset_kwargs: dict[str, Any], replaces: bool
):
    """``delete_local`` on a remote dataset does not make it this import's.

    Regression: the failure handler may only delete a dataset the import
    itself created, and either delete flag was taken as proof of that. For
    a remote dataset ``delete_local`` clears the local cache only - what
    the bucket holds is the dataset that was already there - so a
    part-way failure deleted a production dataset the import had merely
    appended to.
    """
    assert _delete_replaces_dataset(dataset_kwargs) is replaces


def test_failed_import_removes_a_replaced_dataset(
    dataset_name: str, tempdir: Path
):
    """``delete_local`` makes the dataset this import's to clean up again."""
    LuxonisDataset(dataset_name, delete_local=True)
    assert LuxonisDataset.exists(dataset_name)

    dataset_dir = tempdir / "replacing_import"
    class_dir = dataset_dir / "train" / "bird"
    class_dir.mkdir(parents=True)
    create_image(0, class_dir)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            LuxonisDataset,
            "add",
            lambda self, generator, **kwargs: (_ for _ in ()).throw(
                ValueError("boom")
            ),
        )
        with pytest.raises(ValueError, match="boom"):
            LuxonisDataset.import_dataset(
                str(dataset_dir),
                dataset_name=dataset_name,
                dataset_type="clsdir",
                delete_local=True,
            )

    assert dataset_name not in LuxonisDataset.list_datasets()


@pytest.mark.parametrize(
    ("split_ratios", "expected_sizes"),
    [
        (
            {"train": 0, "val": 0.5, "test": 0.5},
            {"train": 0, "val": 2, "test": 2},
        ),
        (
            {"train": 0.5, "val": 0.5, "test": 0},
            {"train": 2, "val": 2, "test": 0},
        ),
    ],
)
def test_split_ratios_mixing_ints_and_floats_are_percentages(
    dataset_name: str,
    tempdir: Path,
    split_ratios: dict[str, float | int],
    expected_sizes: dict[str, int],
):
    """A ratio of exactly ``0`` must not turn the mapping into counts.

    Regression: ``make_splits`` tells ratios from counts by the type of the
    first value alone, so ``{"train": 0, "val": 0.5, "test": 0.5}`` matched
    neither branch and silently fell back to the default 0.8/0.1/0.1 split —
    no error, and not even the "ratios must sum to 1.0" check.
    """
    class_dir = tempdir / "ratio_cls" / "class_a"
    class_dir.mkdir(parents=True)
    for index in range(4):
        create_image(index, class_dir)

    dataset = LuxonisDataset.import_dataset(
        str(class_dir.parent),
        dataset_name=dataset_name,
        dataset_type="clsdir",
        split_ratios=split_ratios,
        delete_local=True,
    )
    try:
        splits = dataset.get_splits()
        assert splits is not None
        assert {
            name: len(group_ids) for name, group_ids in splits.items()
        } == expected_sizes
    finally:
        dataset.delete_dataset(delete_local=True)


def test_explicit_split_ratios_survive_random_split_false(
    dataset_name: str, tempdir: Path
):
    """``split_ratios`` is an explicit request, not automatic splitting.

    Regression: with ``random_split=False`` and a source carrying no splits of
    its own, the ratios were dropped without a word and no splits were made at
    all, so the import "succeeded" into a dataset no loader could read.
    """
    class_dir = tempdir / "no_random_cls" / "class_a"
    class_dir.mkdir(parents=True)
    for index in range(4):
        create_image(index, class_dir)

    dataset = LuxonisDataset.import_dataset(
        str(class_dir.parent),
        dataset_name=dataset_name,
        dataset_type="clsdir",
        split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        random_split=False,
        delete_local=True,
    )
    try:
        splits = dataset.get_splits()
        assert splits is not None
        assert {
            name: len(group_ids) for name, group_ids in splits.items()
        } == {"train": 2, "val": 1, "test": 1}
    finally:
        dataset.delete_dataset(delete_local=True)


def test_count_split_ratios_reject_unknown_splits(
    dataset_name: str, tempdir: Path
):
    """Counts are read for train/val/test only, so anything else must raise.

    Regression: ``{"valid": 4}`` was accepted, read as all-zero counts for the
    three canonical splits, and the request was silently discarded. Percentage
    ratios do honour arbitrary split names, which made the silence worse.
    """
    class_dir = tempdir / "unknown_split_cls" / "class_a"
    class_dir.mkdir(parents=True)
    create_image(0, class_dir)

    with pytest.raises(ValueError, match="only supports the splits"):
        LuxonisDataset.import_dataset(
            str(class_dir.parent),
            dataset_name=dataset_name,
            dataset_type="clsdir",
            split_ratios={"valid": 1},
            delete_local=True,
        )

    assert not LuxonisDataset.exists(dataset_name)


def test_import_routes_each_skeleton_to_its_own_task(
    dataset_name: str, tempdir: Path
):
    """A source with several keypoint tasks keeps a skeleton for each.

    Regression: the skeletons were iterated by value, dropping the task each
    was keyed by, so every one of them was written with ``task=None`` — which
    `set_skeletons` fans out over *all* tasks. A source declaring both a
    17-point ``person`` and a 21-point ``hand`` skeleton ended up reporting
    whichever came last for both.
    """
    source = tempdir / "sample.twoskel"
    source.write_text("two skeletons")
    images = {
        class_name: create_image(index, tempdir)
        for index, class_name in enumerate(("person", "hand"))
    }

    class TwoSkeletonParser(ParserPlugin):
        dataset_types = ("test-two-skeletons",)

        @classmethod
        def detect(cls, source: Path) -> Layout | None:
            if source.suffix != ".twoskel":
                return None
            return Layout({None: {}})

        def parse(
            self, source: Path, layout: Layout, **kwargs: Any
        ) -> ParseResult:
            def records() -> Iterator[tuple[str | None, dict[str, Any]]]:
                for class_name, image in images.items():
                    yield (
                        None,
                        {
                            "file": str(image),
                            "annotation": {"class": class_name},
                        },
                    )

            return ParseResult(
                records(),
                {
                    "person": {"labels": ["head"], "edges": [(0, 0)]},
                    "hand": {"labels": ["thumb", "index"], "edges": [(0, 1)]},
                },
            )

    register_parser_plugin(TwoSkeletonParser, force=True)

    dataset = LuxonisDataset.import_dataset(
        str(source),
        dataset_name=dataset_name,
        dataset_type="test-two-skeletons",
        # Puts each class in a task of its own, which is what the skeletons
        # are keyed by.
        task_name={"person": "person", "hand": "hand"},
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        skeletons = dataset.get_skeletons()
        assert skeletons["person"][0] == ["head"]
        assert skeletons["hand"][0] == ["thumb", "index"]
    finally:
        dataset.delete_dataset(delete_local=True)


def test_failed_import_cleans_up_remote_storage_too(
    dataset_name: str, tempdir: Path
):
    """Cleanup must remove the bucket copy, not only the local one.

    Regression: the failure handler passed only ``delete_local=True``. For a
    remote dataset that skips the branch which deletes the bucket, so every
    batch of media already uploaded before the failure was orphaned there and
    the half-populated remote dataset re-synced on the next open. The keywords
    are also part of the abstract ``delete_dataset`` contract now — it used to
    declare none, so a conforming third-party dataset raised ``TypeError``
    into a ``suppress(Exception)`` and cleaned up nothing at all.
    """
    assert set(inspect.signature(BaseDataset.delete_dataset).parameters) == {
        "self",
        "delete_remote",
        "delete_local",
    }

    dataset_dir = tempdir / "remote_cleanup"
    class_dir = dataset_dir / "train" / "bird"
    class_dir.mkdir(parents=True)
    create_image(0, class_dir)

    calls: list[dict[str, Any]] = []

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            LuxonisDataset,
            "add",
            lambda self, generator, **kwargs: (_ for _ in ()).throw(
                ValueError("boom")
            ),
        )
        patch.setattr(
            LuxonisDataset,
            "delete_dataset",
            lambda self, **kwargs: calls.append(kwargs),
        )
        with pytest.raises(ValueError, match="boom"):
            # No `delete_local` here: that would make the constructor call
            # `delete_dataset` too, and only the cleanup call is of interest.
            LuxonisDataset.import_dataset(
                str(dataset_dir),
                dataset_name=dataset_name,
                dataset_type="clsdir",
            )

    assert calls == [{"delete_local": True, "delete_remote": True}]

    # The patched-out cleanup never ran, so do it here.
    if LuxonisDataset.exists(dataset_name):
        LuxonisDataset(dataset_name).delete_dataset(delete_local=True)


def test_prepare_import_records_fans_out_in_a_stable_order(tempdir: Path):
    """Two imports of one source must emit the same records in one order.

    Regression: the copies of an annotation-less record were emitted by
    iterating a ``set`` of task names, whose order depends on PYTHONHASHSEED,
    so the same source imported twice produced the same rows in a different
    sequence — which defeats comparing two imports byte for byte.
    """
    unannotated = DatasetRecord(
        files={"image": create_image(0, tempdir)}, annotation=None
    )
    # Names whose set order differs from their sorted order for most seeds.
    task_name = {
        "a": "zebra",
        "b": "ant",
        "c": "moth",
        "d": "bee",
        "e": "crow",
    }

    prepared = _collect_dataset_records(
        _prepare_import_records(
            iter([(None, unannotated)]),
            task_name=task_name,
            selected_files=None,
            split_files={},
        )
    )

    assert [record.task_name for record in prepared] == sorted(
        set(task_name.values())
    )
