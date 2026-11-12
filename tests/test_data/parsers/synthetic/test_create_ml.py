"""CreateML parser."""

import inspect
import json
from itertools import islice
from pathlib import Path
from types import GeneratorType
from typing import IO, Any

import pytest

from luxonis_ml.data.parsers import (
    CreateMLParser,
)
from luxonis_ml.data.parsers import create_ml_parser as parser_module
from luxonis_ml.utils.path import resolve_manifest_path
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
    _split_records,
)


def _box(label: str = "bird") -> dict[str, Any]:
    return {
        "label": label,
        "coordinates": {"x": 10, "y": 5, "width": 10, "height": 4},
    }


def _write_manifest(split: Path, entries: list[dict[str, Any]]) -> Path:
    annotations = split / "_annotations.createml.json"
    annotations.write_text(json.dumps(entries))
    return annotations


def _mixed_split(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Write a split covering every frame role the manifest can hold.

    Returns:
        The split directory, the annotated first frame, the frame without
        boxes, and the frame the manifest names twice.

    """
    split = tmp_path / "train"
    first = _image(split / "first.jpg")
    empty = _image(split / "empty.jpg")
    second = _image(split / "second.jpg")
    _write_manifest(
        split,
        [
            {"image": first.name, "annotations": [_box(), _box("cat")]},
            # No box at all, so this frame yields no record and must stay
            # out of the file list.
            {"image": empty.name, "annotations": []},
            {"image": "missing.jpg", "annotations": [_box()]},
            {"image": second.name, "annotations": [_box()]},
            # Named twice by the manifest, but still one file.
            {"image": second.name, "annotations": [_box("cat")]},
        ],
    )
    return split, first, empty, second


def _count_image_opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the path of every image the parser decodes."""
    opened: list[str] = []
    original_open = parser_module.Image.open

    def counting_open(fp: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(fp))
        return original_open(fp, *args, **kwargs)

    monkeypatch.setattr(parser_module.Image, "open", counting_open)
    return opened


def test_create_ml_parser_with_missing_and_valid_images(tmp_path: Path):
    parser = _plugin(CreateMLParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    annotations = split / "_annotations.createml.json"
    annotations.write_text("[]")
    assert parser.validate_split(split) is None

    image = _image(split / "image.jpg")
    annotations.write_text(
        json.dumps(
            [
                {
                    "image": image.name,
                    "annotations": [
                        {
                            "label": "bird",
                            "coordinates": {
                                "x": 10,
                                "y": 5,
                                "width": 10,
                                "height": 4,
                            },
                        }
                    ],
                },
                {"image": "missing.jpg", "annotations": []},
            ]
        )
    )
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._split_records(**kwargs))
    assert parser._split_files(**kwargs) == [image.resolve()]
    assert records[0]["annotation"] == {
        "class": "bird",
        "boundingbox": {"x": 0.25, "y": 0.3, "w": 0.5, "h": 0.4},
    }
    assert len(parser._issues.messages) == 1


def test_create_ml_parse_tags_records_with_their_split(tmp_path: Path):
    """Detection finds the splits and every record is tagged with one.

    ``valid`` is the name Roboflow writes and ``val`` the name LDF uses,
    so the canonical name has to reach both the records and the file
    enumeration.
    """
    for split_name in ("train", "valid"):
        split = tmp_path / split_name
        image = _image(split / f"{split_name}.jpg")
        _write_manifest(
            split, [{"image": image.name, "annotations": [_box()]}]
        )

    layout = CreateMLParser.detect(tmp_path)
    assert layout is not None
    assert layout.split_names == ["train", "val"]

    parser = _plugin(CreateMLParser)
    result = parser.parse(tmp_path, layout)
    tagged = _split_records(result)

    assert [split_name for split_name, _ in tagged] == ["train", "val"]
    assert [Path(record["file"]).name for _, record in tagged] == [
        "train.jpg",
        "valid.jpg",
    ]
    assert result.skeletons == {}
    assert parser.enumerate_files(tmp_path, layout) == {
        "train": [(tmp_path / "train" / "train.jpg").resolve()],
        "val": [(tmp_path / "valid" / "valid.jpg").resolve()],
    }


def test_create_ml_records_stream_lazily_in_one_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The manifest is read once, and only once a record is pulled.

    Replaces the guard on the old contract, where the file list had to be
    complete before the records were consumed and was built by replaying
    the record generator, so every record dict was built twice. The
    records now come straight out of the single pass over the manifest:
    nothing is read before the first record is pulled, the manifest is
    loaded exactly once, and a frame is decoded when it is reached rather
    than up front.
    """
    split, first, empty, second = _mixed_split(tmp_path)

    loaded: list[str] = []
    original_load = parser_module.json.load

    def counting_load(fp: IO[str], *args: Any, **kwargs: Any) -> Any:
        loaded.append(getattr(fp, "name", ""))
        return original_load(fp, *args, **kwargs)

    monkeypatch.setattr(parser_module.json, "load", counting_load)
    opened = _count_image_opens(monkeypatch)

    parser = _plugin(CreateMLParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None

    records = parser._split_records(**kwargs)
    assert isinstance(records, GeneratorType)
    assert inspect.getgeneratorstate(records) == "GEN_CREATED"
    assert loaded == []
    assert opened == []

    # Only the frame the first record belongs to may have been touched.
    assert [record["file"] for record in _records(islice(records, 1))] == [
        str(first.resolve())
    ]
    assert len(loaded) == 1
    assert opened == [str(first.resolve())]

    assert [record["file"] for record in _records(records)] == [
        str(first.resolve()),
        str(second.resolve()),
        str(second.resolve()),
    ]
    # One pass over the manifest, and one decode per frame it names that
    # exists — the frame without boxes included, because an unreadable
    # image has to fail the parse whether or not it carries annotations.
    assert len(loaded) == 1
    assert opened == [
        str(first.resolve()),
        str(empty.resolve()),
        str(second.resolve()),
        str(second.resolve()),
    ]


def test_create_ml_file_list_comes_from_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Enumerating the files must not build the records at all.

    Regression: the file list came from draining a second run of the
    record generator, so every record dict — and a ``Path`` per record —
    was built twice for a list the manifest already spells out. It must
    still hold exactly the frames that yield a record, in manifest order
    and deduplicated, and now costs no image decode whatsoever.
    """
    split, first, _empty, second = _mixed_split(tmp_path)
    opened = _count_image_opens(monkeypatch)

    parser = _plugin(CreateMLParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None

    assert parser._split_files(**kwargs) == [first.resolve(), second.resolve()]
    assert opened == []
    # Enumerating is not a parse: the image the manifest names but does
    # not have is reported by the parse itself, not here.
    assert parser._issues.messages == []


def test_create_ml_resolves_the_split_directory_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The split directory must not be re-resolved per manifest entry.

    Regression: ``image_dir.absolute().resolve()`` sat inside the entry
    loop, so every entry paid a second ``realpath`` walk on top of the
    one resolving the image itself. Exactly one resolve per entry, plus
    the single one for the split directory, is expected.
    """
    split = tmp_path / "train"
    entries = []
    for index in range(4):
        image = _image(split / f"img_{index}.jpg")
        entries.append({"image": image.name, "annotations": [_box()]})
    _write_manifest(split, entries)

    parser = _plugin(CreateMLParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None

    resolved: list[str] = []
    original_resolve = Path.resolve

    def counting_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        resolved.append(str(self))
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting_resolve)
    # Counted across the whole stream, since the records are what walks
    # the manifest now.
    records = _records(parser._split_records(**kwargs))
    monkeypatch.undo()

    assert len(records) == 4
    assert len(resolved) == len(entries) + 1


def test_create_ml_plain_image_names_skip_manifest_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    r"""Separator-free references take a shortcut that must still agree.

    Regression: a reference holding neither ``/`` nor ``\\`` can be
    neither absolute nor Windows-flavoured, so ``resolve_manifest_path``
    provably reduces to joining it onto the split directory; the parser
    takes that shortcut instead of building three throwaway path objects
    per entry. Only references that do carry a separator may reach
    ``resolve_manifest_path``, and every form must resolve to exactly
    what ``resolve_manifest_path`` returns.
    """
    split = tmp_path / "train"
    plain = _image(split / "plain.jpg")
    nested = _image(split / "nested" / "posix.jpg")
    windows = _image(split / "nested" / "windows.jpg")
    absolute = _image(split / "absolute.jpg")
    references = [
        plain.name,
        f"nested/{nested.name}",
        f"nested\\{windows.name}",
        str(absolute.resolve()),
    ]
    _write_manifest(
        split,
        [
            {"image": reference, "annotations": [_box()]}
            for reference in references
        ],
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

    parser = _plugin(CreateMLParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._split_records(**kwargs))

    base_dir = split.absolute().resolve()
    assert [record["file"] for record in records] == [
        str(resolve_manifest_path(base_dir, reference))
        for reference in references
    ]
    assert delegated == references[1:]
    # The enumeration resolves the same references the same way, so a
    # count-based import selects the files the records name.
    assert parser._split_files(**kwargs) == [
        resolve_manifest_path(base_dir, reference) for reference in references
    ]


def test_create_ml_reads_each_image_header_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The image size is per frame, so it must be read per frame.

    Guards the box loop against pulling the image open back inside it:
    the width and height a box is normalized by depend only on the
    frame, and reading them per box would open a frame with ten boxes
    ten times.
    """
    split = tmp_path / "train"
    first = _image(split / "first.jpg")
    second = _image(split / "second.jpg")
    _write_manifest(
        split,
        [
            {
                "image": image.name,
                "annotations": [_box(), _box("cat"), _box("dog")],
            }
            for image in (first, second)
        ],
    )

    opened = _count_image_opens(monkeypatch)

    parser = _plugin(CreateMLParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    assert len(_records(parser._split_records(**kwargs))) == 6
    assert opened == [str(first.resolve()), str(second.resolve())]
