"""Native (LDF) parser."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from luxonis_ml.data.parsers import (
    NativeParser,
)
from luxonis_ml.data.parsers import native_parser as native_parser_module
from luxonis_ml.typing import PathType
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
    _split_records,
)
from tests.test_data.utils import create_image


def _write_split(split: Path, entries: list[dict[str, Any]]) -> Path:
    """Write ``entries`` as the ``annotations.json`` of ``split``."""
    split.mkdir(parents=True, exist_ok=True)
    annotations = split / "annotations.json"
    annotations.write_text(json.dumps(entries))
    return annotations


def test_native_parser_resolves_files_and_masks(tmp_path: Path):
    parser = _plugin(NativeParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    image = _image(split / "image.jpg")
    depth = split / "depth.bin"
    depth.write_bytes(b"depth")
    mask = split / "mask.png"
    Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(mask)
    annotations = split / "annotations.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "file": image.name,
                    "annotation": {
                        "segmentation": {"mask": mask.name},
                        "instance_segmentation": {"mask": 123},
                    },
                },
                {
                    "files": {
                        "image": image.name,
                        "depth": depth.name,
                    },
                    "annotation": None,
                },
                {"annotation": None},
                {"file": image.name, "annotation": {"segmentation": {}}},
            ]
        )
    )

    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._split_records(**kwargs))
    assert records[0]["file"] == image.resolve()
    assert records[0]["annotation"]["segmentation"]["mask"] == mask.resolve()
    assert records[0]["annotation"]["instance_segmentation"]["mask"] == 123
    assert records[1]["files"] == {
        "image": image.resolve(),
        "depth": depth.resolve(),
    }
    assert records[3]["annotation"] == {"segmentation": {}}


def test_native_parser_resolves_every_path_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each distinct path of a split is resolved exactly once.

    Regression: the file list was built by re-running the record
    generator, so every reference was resolved twice; on top of that each
    annotation of an image resolved that image anew, so a file carrying
    ``n`` annotations cost ``2 * n`` ``realpath`` walks and a mask the
    same. The file list is now collected from the records as they stream,
    leaving the resolution cache as the only thing between the eight
    references below and the three paths they name -- undoing it makes
    the recorded calls outnumber the distinct paths again.
    """
    split = tmp_path / "train"
    image = _image(split / "images" / "img.jpg")
    depth = split / "depth" / "img.png"
    depth.parent.mkdir(parents=True)
    depth.write_bytes(b"depth")
    mask = _image(split / "masks" / "img.png")

    annotated = {
        "file": "images/img.jpg",
        "annotation": {
            "class": "class0",
            "segmentation": {"mask": "masks/img.png"},
        },
    }
    _write_split(
        split,
        [
            *(annotated for _ in range(3)),
            {
                "files": {
                    "image": "images/img.jpg",
                    "depth": "depth/img.png",
                },
                "annotation": None,
            },
        ],
    )

    resolved: list[str] = []
    original_resolve = native_parser_module.resolve_manifest_path

    def counting_resolve(base_dir: Path, value: PathType) -> Path:
        resolved.append(str(value))
        return original_resolve(base_dir, value)

    monkeypatch.setattr(
        native_parser_module, "resolve_manifest_path", counting_resolve
    )

    parser = _plugin(NativeParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._split_records(**kwargs))
    assert len(records) == 4

    # Eight references -- three images, three masks and the two members
    # of the multi-source record -- name only three distinct paths.
    assert sorted(resolved) == [
        "depth/img.png",
        "images/img.jpg",
        "masks/img.png",
    ]

    # The files the importer collects as the records go past, which is
    # what the parser used to have to publish up front.
    assert records[0]["file"] == image.resolve()
    assert records[0]["annotation"]["segmentation"]["mask"] == mask.resolve()
    assert records[3]["files"] == {
        "image": image.resolve(),
        "depth": depth.resolve(),
    }


def test_native_parser_records_stay_lazy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Records are rewritten one at a time, and each split read once.

    A parse walks the whole JSON document to rewrite its paths, which
    makes it tempting to walk it up front and hand the walked list back
    as ``records`` -- the pre-pass the old contract needed to publish a
    file list. `ParseResult` instead wants a single-consumption iterator
    that starts no work until a record is pulled and then does only that
    record's; this pins both, and that every split is read exactly once.
    """
    train_images = [
        _image(tmp_path / "train" / "images" / f"img_{index}.jpg")
        for index in range(3)
    ]
    val_image = _image(tmp_path / "val" / "images" / "img.jpg")
    _write_split(
        tmp_path / "train",
        [
            {"file": f"images/img_{index}.jpg", "annotation": None}
            for index in range(3)
        ],
    )
    _write_split(
        tmp_path / "val",
        [{"file": "images/img.jpg", "annotation": None}],
    )

    reads: list[Path] = []
    resolved: list[str] = []
    original_read_text = Path.read_text
    original_resolve = native_parser_module.resolve_manifest_path

    def counting_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "annotations.json":
            reads.append(self)
        return original_read_text(self, *args, **kwargs)

    def counting_resolve(base_dir: Path, value: PathType) -> Path:
        resolved.append(str(value))
        return original_resolve(base_dir, value)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    monkeypatch.setattr(
        native_parser_module, "resolve_manifest_path", counting_resolve
    )

    layout = NativeParser.detect(tmp_path)
    assert layout is not None
    assert layout.split_names == ["train", "val"]
    assert reads == [], "detection must not read the annotations"

    train_records = _plugin(NativeParser)._split_records(
        **layout.splits["train"]
    )
    assert reads == [], "the document is read only once a record is pulled"
    first = next(train_records)
    assert isinstance(first, dict)
    assert first["file"] == train_images[0].resolve()
    assert resolved == ["images/img_0.jpg"], (
        "a record is rewritten as it is yielded, not in a pass over the "
        "whole document"
    )

    reads.clear()
    parsed = _plugin(NativeParser).parse(tmp_path, layout)
    assert reads == [], "no split may be read before the records are pulled"
    assert iter(parsed.records) is parsed.records

    streamed = _split_records(parsed)
    assert [split_name for split_name, _ in streamed] == [
        "train",
        "train",
        "train",
        "val",
    ]
    assert reads == [
        tmp_path / "train" / "annotations.json",
        tmp_path / "val" / "annotations.json",
    ]
    assert [record["file"] for _, record in streamed] == [
        *(image.resolve() for image in train_images),
        val_image.resolve(),
    ]
    assert list(parsed.records) == []


def test_native_parser_accepts_windows_style_file_paths(tempdir: Path):
    image_path = create_image(0, tempdir)
    split_dir = tempdir / "train"
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True)
    copied_image = image_dir / image_path.name
    copied_image.write_bytes(image_path.read_bytes())

    annotations_path = split_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "file": f"images\\{image_path.name}",
                    "task_name": "task",
                    "annotation": {
                        "class": "class0",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.2,
                            "w": 0.3,
                            "h": 0.4,
                        },
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # A bare split directory carries no split information of its own.
    layout = NativeParser.detect(split_dir)
    assert layout is not None
    assert layout.split_names == []

    parsed = _plugin(NativeParser).parse(split_dir, layout)
    streamed = _split_records(parsed)
    assert len(streamed) == 1
    split_name, parsed_record = streamed[0]
    assert split_name is None
    parsed_file = (
        parsed_record["file"]
        if isinstance(parsed_record, dict)
        else parsed_record.file
    )
    assert parsed_file == copied_image.resolve()
