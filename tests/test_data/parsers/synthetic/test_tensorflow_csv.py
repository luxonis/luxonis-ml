"""TensorFlow CSV parser."""

import inspect
from collections.abc import Sequence
from pathlib import Path
from types import GeneratorType
from typing import Any

import polars as pl
import pytest

from luxonis_ml.data.parsers import (
    TensorflowCSVParser,
)
from luxonis_ml.data.parsers import tensorflow_csv_parser as parser_module
from luxonis_ml.data.parsers.parser_plugin import ParserPlugin
from luxonis_ml.utils.path import resolve_manifest_path
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
    _split_records,
)

HEADER = "filename,width,height,class,xmin,ymin,xmax,ymax"


def _row(filename: str) -> str:
    return f"{filename},20,10,bird,2,1,12,6"


def _write_annotations(split: Path, rows: Sequence[str]) -> Path:
    annotations = split / "_annotations.csv"
    annotations.write_text("\n".join([HEADER, *rows]) + "\n")
    return annotations


def test_tensorflow_csv_parser(tmp_path: Path):
    parser = _plugin(TensorflowCSVParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    _image(split / "annotated.jpg", size=(20, 10))
    assert parser.validate_split(split) is None

    unannotated = _image(split / "unannotated.jpg")
    annotations = split / "_annotations.csv"
    annotations.write_text(
        "filename,width,height,class,xmin,ymin,xmax,ymax\n"
        "annotated.jpg,20,10,bird,2,1,12,6\n"
        ",20,10,bird,0,0,1,1\n"
    )
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._split_records(**kwargs))

    assert len(records) == 2
    annotated = next(record for record in records if record["annotation"])
    assert annotated["annotation"]["boundingbox"] == {
        "x": 0.1,
        "y": 0.1,
        "w": 0.5,
        "h": 0.5,
    }
    assert (
        next(
            record
            for record in records
            if Path(record["file"]).name == unannotated.name
        )["annotation"]
        is None
    )


def test_tensorflow_csv_file_list_does_not_replay_the_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Enumerating the files must not build every record a second time.

    Regression: ``files`` came from ``_get_added_images(generator())``
    while the returned stream ran a fresh ``generator()``, so the whole
    split was walked twice — once per record dict that was thrown away
    again. Every listed image yields at least one record naming it, so
    the listing handed over by ``validate_split`` already is the file
    list: enumerating it lists no directory and reads no annotation file,
    and parsing walks the split exactly once, lazily.
    """
    split = tmp_path / "train"
    annotated = _image(split / "annotated.jpg", size=(20, 10))
    empty = _image(split / "empty.jpg", size=(20, 10))
    _write_annotations(split, [_row(annotated.name), _row(annotated.name)])

    parser = _plugin(TensorflowCSVParser)
    # Detection reports the layout it recognized, and parsing is handed
    # that layout instead of looking at the source a second time.
    layout = TensorflowCSVParser.detect(tmp_path)
    assert layout is not None
    assert layout.split_names == ["train"]

    listed: list[Path] = []
    list_images = ParserPlugin._list_images

    def counting_list_images(image_dir: Path) -> list[Path]:
        listed.append(image_dir)
        return list_images(image_dir)

    read: list[Path] = []
    read_csv = pl.read_csv

    def counting_read_csv(source: Any, *args: Any, **kwargs: Any) -> Any:
        read.append(Path(source))
        return read_csv(source, *args, **kwargs)

    monkeypatch.setattr(
        TensorflowCSVParser, "_list_images", staticmethod(counting_list_images)
    )
    monkeypatch.setattr(parser_module.pl, "read_csv", counting_read_csv)

    enumerated = parser.enumerate_files(tmp_path, layout)
    assert enumerated is not None
    # Compared unordered: the listing comes straight from `glob`, whose
    # order is the file system's and differs between machines.
    assert {split: sorted(files) for split, files in enumerated.items()} == {
        "train": sorted([annotated.resolve(), empty.resolve()])
    }
    assert listed == []
    assert read == []

    parsed = parser.parse(tmp_path, layout)
    # Nothing is walked until the stream is pulled.
    assert isinstance(parsed.records, GeneratorType)
    assert inspect.getgeneratorstate(parsed.records) == "GEN_CREATED"
    assert read == []

    records = _split_records(parsed)
    # One pass over the split: the CSV is read once and the directory is
    # not listed again.
    assert read == [split / "_annotations.csv"]
    assert listed == []
    assert [
        (split_name, record["file"]) for split_name, record in records
    ] == [
        ("train", str(annotated.resolve())),
        ("train", str(annotated.resolve())),
        ("train", str(empty.resolve())),
    ]


def test_tensorflow_csv_resolves_each_filename_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    r"""Rows naming a listed image must not resolve it again.

    Regression: ``resolve_manifest_path`` ran once per row, so an image
    with ten boxes paid ten ``realpath`` walks for a path the listing had
    already resolved. A filename that reads as the name of a listed image
    — plainly, Windows-flavoured, or as that image's resolved path — is
    that image, and any other spelling is resolved once however many rows
    repeat it.
    """
    split = tmp_path / "train"
    first = _image(split / "first.jpg", size=(20, 10))
    second = _image(split / "second.jpg", size=(20, 10))
    spellings = [
        first.name,
        f".\\{first.name}",
        str(second.resolve()),
        # Reads as neither, so this one is resolved -- once, not twice.
        f"sub/../{second.name}",
    ]
    _write_annotations(
        split, [_row(spelling) for spelling in spellings for _ in range(2)]
    )

    delegated: list[str] = []

    def counting_resolve_manifest_path(base_dir: Path, value: Any) -> Path:
        delegated.append(str(value))
        return resolve_manifest_path(base_dir, value)

    monkeypatch.setattr(
        parser_module,
        "resolve_manifest_path",
        counting_resolve_manifest_path,
    )

    parser = _plugin(TensorflowCSVParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._split_records(**kwargs))

    assert delegated == [spellings[-1]]
    # Each spelling still lands on the image it names.
    assert [record["file"] for record in records] == [
        str(first.resolve()),
        str(first.resolve()),
        str(first.resolve()),
        str(first.resolve()),
        str(second.resolve()),
        str(second.resolve()),
        str(second.resolve()),
        str(second.resolve()),
    ]


def test_tensorflow_csv_resolves_the_split_directory_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Listed images must not be resolved one full path walk each.

    Regression: every image was resolved on its own, so each one restated
    the split directory to ``realpath``, which stats every component of a
    path. The directory is resolved once and only images that really are
    symlinks — the sole component that can still redirect — are resolved
    individually, which must not change the path any of them reports.
    """
    split = tmp_path / "train"
    plain = [
        _image(split / f"img_{index}.jpg", size=(20, 10)) for index in range(4)
    ]
    target = _image(tmp_path / "elsewhere" / "target.jpg", size=(20, 10))
    (split / "link.jpg").symlink_to(target)
    _write_annotations(
        split, [_row(image.name) for image in [*plain, Path("link.jpg")]]
    )

    parser = _plugin(TensorflowCSVParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None

    resolved: list[str] = []
    path_resolve = Path.resolve

    def counting_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        resolved.append(str(self))
        return path_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting_resolve)
    parsed = parser._split_records(**kwargs)
    monkeypatch.undo()

    assert resolved == [str(split), str(split / "link.jpg")]
    files = {record["file"] for record in _records(parsed)}
    assert files == {str(image.resolve()) for image in plain} | {
        str(target.resolve())
    }


def test_tensorflow_csv_streams_the_annotation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The CSV must not be turned into one dictionary per row.

    Regression: ``rows(named=True)`` built a dict of every column for
    every bounding box in the split and held them all at once, which cost
    more than reading the rows in buffered batches and kept a copy of the
    whole annotation file in memory next to the parsed records.
    """
    split = tmp_path / "train"
    image = _image(split / "img.jpg", size=(20, 10))
    _write_annotations(split, [_row(image.name)] * 4)

    named_calls: list[bool] = []
    dataframe_rows = pl.DataFrame.rows

    def recording_rows(self: pl.DataFrame, *, named: bool = False) -> Any:
        named_calls.append(named)
        return dataframe_rows(self, named=named)

    monkeypatch.setattr(pl.DataFrame, "rows", recording_rows)

    parser = _plugin(TensorflowCSVParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._split_records(**kwargs))
    monkeypatch.undo()

    assert True not in named_calls
    assert len(records) == 4
