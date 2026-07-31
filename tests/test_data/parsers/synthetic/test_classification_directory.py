"""Classification-directory parser."""

import inspect
import os
from collections import Counter
from pathlib import Path
from types import GeneratorType

import pytest
from loguru import logger

from luxonis_ml.data import (
    PARSERS_REGISTRY,
    Layout,
    LuxonisDataset,
    LuxonisParser,
    ParserPlugin,
)
from luxonis_ml.data.parsers import (
    ClassificationDirectoryParser,
)
from luxonis_ml.enums import DatasetType
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
    _split_records,
)
from tests.test_data.utils import create_image


def _classes(root: Path, counts: dict[str, int]) -> Path:
    """Write ``counts`` images into one directory per class name."""
    for class_name, count in counts.items():
        for index in range(count):
            _image(root / class_name / f"{class_name}_{index}.jpg")
    return root


def _detect(source: Path) -> Layout:
    """Return the layout an import would hand to `parse`."""
    layout = ClassificationDirectoryParser.detect(source)
    assert layout is not None
    return layout


def _count_listings(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Return a list that grows by every directory listed for images."""
    listed: list[Path] = []
    list_images = ParserPlugin._list_images

    def counting_list_images(image_dir: Path) -> list[Path]:
        listed.append(image_dir)
        return list_images(image_dir)

    monkeypatch.setattr(
        ParserPlugin, "_list_images", staticmethod(counting_list_images)
    )
    return listed


def test_classification_directory_does_not_claim_data_directory(
    tempdir: Path,
):
    data_dir = tempdir / "coco" / "test" / "data"
    data_dir.mkdir(parents=True)
    create_image(0, data_dir)

    plugin = PARSERS_REGISTRY.get(DatasetType.CLSDIR.value)
    assert plugin.detect(data_dir.parent.parent) is None


def test_partial_split_clsdir_is_preserved(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "clsdir_partial"
    split_dir = dataset_dir / "valid" / "budgie"
    split_dir.mkdir(parents=True)
    create_image(16, split_dir)

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 0
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 0
    dataset.delete_dataset(delete_local=True)


def test_partial_split_clsdir_explicit_type_uses_dir_mode(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "clsdir_partial_explicit"
    split_dir = dataset_dir / "test" / "finch"
    split_dir.mkdir(parents=True)
    create_image(16, split_dir)

    with pytest.warns(DeprecationWarning, match="LuxonisParser"):
        parser = LuxonisParser(
            str(dataset_dir),
            dataset_name=dataset_name,
            dataset_type="clsdir",
            delete_local=True,
            save_dir=tempdir,
        )
    dataset = parser.parse()

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 0
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 1
    assert parser.get_parser_issue_messages() == []
    dataset.delete_dataset(delete_local=True)


def test_clsdir_ignores_reserved_directory_names(
    dataset_name: str,
    tempdir: Path,
):
    """Reserved directory names must never be ingested as classes.

    Regression: ``validate_split`` skips directories belonging to other
    layouts (``data``, ``raw``, ``masks``, split names, ``images``,
    ``labels``), but the record walk listed every subdirectory, so a
    source validated on its real class folders and then gained a bogus
    ``data`` class. Both now share one reserved-name set.
    """
    dataset_dir = tempdir / "clsdir_reserved"
    for index, class_name in enumerate(("budgie", "parrot", "data")):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True)
        create_image(index, class_dir)

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="clsdir",
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        class_names = {
            name
            for names in dataset.get_class_names().values()
            for name in names
        }
        assert class_names == {"budgie", "parrot"}
    finally:
        dataset.delete_dataset(delete_local=True)


def test_clsdir_says_which_directories_it_did_not_import(tempdir: Path):
    """Skipping a reserved-named directory of images must be reported.

    Reserved names keep another layout's directories from becoming
    classes, but the same name is also a plausible class - `train` is a
    kind of vehicle, `masks` a kind of object. Those images used to be
    dropped from the import in silence, so the dataset came out short
    with nothing to explain it. Only a directory that actually holds
    images is worth a line.
    """
    dataset_dir = tempdir / "clsdir_reserved_warning"
    for index, class_name in enumerate(("budgie", "train")):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True)
        create_image(index, class_dir)
    # Holds no images, so it is skipped either way and stays quiet.
    (dataset_dir / "masks").mkdir()

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )
    try:
        records = _records(
            _plugin(ClassificationDirectoryParser)._split_records(dataset_dir)
        )
    finally:
        logger.remove(sink_id)

    assert {record["annotation"]["class"] for record in records} == {"budgie"}
    assert [message for message in messages if "Not importing" in message] == [
        f"Not importing '{dataset_dir / 'train'}' as a class: its name "
        "belongs to another dataset layout. Rename the directory if it "
        "really is a class."
    ]


def test_classification_directory_validation(tmp_path: Path):
    parser = _plugin(ClassificationDirectoryParser)
    assert parser.validate_split(tmp_path / "missing") is None

    reserved = tmp_path / "reserved"
    for name in (
        "train",
        "valid",
        "test",
        "val",
        "validation",
        "images",
        "labels",
        "data",
        "raw",
        "masks",
    ):
        (reserved / name).mkdir(parents=True)
    assert parser.validate_split(reserved) is None

    class_dir = tmp_path / "classes"
    _image(class_dir / "bird" / "bird.jpg")
    (class_dir / "unexpected.txt").write_text("x")
    assert parser.validate_split(class_dir) is None
    (class_dir / "unexpected.txt").unlink()
    (class_dir / "info.json").write_text("{}")
    assert parser.validate_split(class_dir) == {"class_dir": class_dir}


def test_clsdir_tags_records_with_their_split(tmp_path: Path):
    """Each record must carry the split its class directory sits in.

    Which split a record belongs to used to be implied by the per-split
    file lists the parser published up front. It now rides along with the
    record, and ``valid`` must still be canonicalized to ``val``.
    """
    dataset_dir = tmp_path / "clsdir"
    for split, count in (("train", 3), ("valid", 2), ("test", 1)):
        _classes(dataset_dir / split, {"budgie": count})

    layout = _detect(dataset_dir)
    assert layout.split_names == ["train", "val", "test"]

    parser = _plugin(ClassificationDirectoryParser)
    records = _split_records(parser.parse(dataset_dir, layout))

    assert Counter(split_name for split_name, _ in records) == {
        "train": 3,
        "val": 2,
        "test": 1,
    }


def test_clsdir_lists_each_class_directory_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Parsing must walk the class directories exactly once.

    Regression: the file list was built by running the record generator a
    second time, so every class directory was listed and every image path
    resolved twice. Streaming the records is now the only walk a parse
    does, and enumerating the files for a count-based import walks once
    instead of parsing everything twice.
    """
    dataset_dir = _classes(tmp_path / "clsdir", {"budgie": 3, "finch": 2})

    listed = _count_listings(monkeypatch)

    parser = _plugin(ClassificationDirectoryParser)
    layout = _detect(dataset_dir)
    result = parser.parse(dataset_dir, layout)

    assert listed == []
    assert len(_records(result)) == 5
    assert sorted(path.name for path in listed) == ["budgie", "finch"]

    listed.clear()
    enumerated = parser.enumerate_files(dataset_dir, layout)
    assert enumerated is not None
    assert len(enumerated[None]) == 5
    assert sorted(path.name for path in listed) == ["budgie", "finch"]


def test_clsdir_resolves_one_path_per_class_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Resolving must stay hoisted out of the per-image loop.

    Regression: every image path was resolved on its own, and resolving
    walks and stats each shared parent component again per image. A class
    directory without symlinked entries is resolved once and the image
    names are joined onto the result.
    """
    dataset_dir = _classes(tmp_path / "clsdir", {"budgie": 4, "finch": 4})

    resolved: list[Path] = []
    resolve = Path.resolve

    def counting_resolve(self: Path, strict: bool = False) -> Path:
        resolved.append(self)
        return resolve(self, strict)

    monkeypatch.setattr(Path, "resolve", counting_resolve)

    parser = _plugin(ClassificationDirectoryParser)
    records = _records(parser.parse(dataset_dir, _detect(dataset_dir)))

    assert len(records) == 8
    assert sorted(path.name for path in resolved) == ["budgie", "finch"]


def test_clsdir_resolves_symlinks_like_realpath(tmp_path: Path):
    """Symlinked images and class directories keep their real paths.

    Regression: joining an image name onto the resolved class directory
    is only equivalent to resolving the image while the image itself is
    not a symlink. Symlinked entries are resolved individually, or a link
    would be reported under the directory that lists it instead of under
    its target.
    """
    store = tmp_path / "store"
    _image(store / "outside.jpg")
    _image(store / "shared" / "inside.jpg")

    dataset_dir = tmp_path / "clsdir"
    budgie = dataset_dir / "budgie"
    _image(budgie / "own.jpg")
    (budgie / "linked.jpg").symlink_to(store / "outside.jpg")
    (budgie / "relative.jpg").symlink_to(
        Path("..") / ".." / "store" / "outside.jpg"
    )
    (dataset_dir / "finch").symlink_to(
        store / "shared", target_is_directory=True
    )

    parser = _plugin(ClassificationDirectoryParser)
    layout = _detect(dataset_dir)
    records = _records(parser.parse(dataset_dir, layout))

    expected = {
        os.path.realpath(budgie / "own.jpg"),
        os.path.realpath(store / "outside.jpg"),
        os.path.realpath(store / "shared" / "inside.jpg"),
    }
    assert {record["file"] for record in records} == expected

    enumerated = parser.enumerate_files(dataset_dir, layout)
    assert enumerated is not None
    assert {str(file) for file in enumerated[None]} == expected


def test_clsdir_lists_an_image_reached_twice_only_once(tmp_path: Path):
    """Images two entries resolve to must be enumerated once.

    Regression: the file list is deduplicated while the records are not,
    so a class directory holding both an image and a symlink to it yields
    two records but a single file.
    """
    dataset_dir = tmp_path / "clsdir"
    budgie = dataset_dir / "budgie"
    _image(budgie / "own.jpg")
    (budgie / "copy.jpg").symlink_to(budgie / "own.jpg")

    parser = _plugin(ClassificationDirectoryParser)
    layout = _detect(dataset_dir)

    enumerated = parser.enumerate_files(dataset_dir, layout)
    assert enumerated is not None
    assert enumerated[None] == [Path(os.path.realpath(budgie / "own.jpg"))]
    assert len(_records(parser.parse(dataset_dir, layout))) == 2


def test_clsdir_records_stay_lazy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """No record may be produced before the caller asks for one.

    Regression: the file list was eager by contract, and building it from
    the record generator ran that generator to completion inside
    ``parse``. Nothing is listed until the first record is pulled, and
    pulling it lists only the class directory that record came from,
    which is what lets a dataset too large to hold in memory stream.
    """
    dataset_dir = _classes(tmp_path / "clsdir", {"budgie": 2, "finch": 2})

    listed = _count_listings(monkeypatch)

    parser = _plugin(ClassificationDirectoryParser)
    result = parser.parse(dataset_dir, _detect(dataset_dir))

    assert isinstance(result.records, GeneratorType)
    assert inspect.getgeneratorstate(result.records) == inspect.GEN_CREATED
    assert listed == []

    split_name, record = next(result.records)
    assert split_name is None
    assert isinstance(record, dict)
    assert record["annotation"]["class"] in {"budgie", "finch"}
    assert inspect.getgeneratorstate(result.records) == inspect.GEN_SUSPENDED
    assert len(listed) == 1
