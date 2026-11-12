"""Segmentation-mask-directory parser."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from luxonis_ml.data.parsers import (
    SegmentationMaskDirectoryParser,
)
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
    _symlink,
)

CLASSES = ("background", "bird", "cat", "dog")


def _write_classes(split: Path, classes: Sequence[str] = CLASSES) -> Path:
    rows = "".join(f"{value},{name}\n" for value, name in enumerate(classes))
    path = split / "_classes.csv"
    path.write_text(f"Pixel Value, Class\n{rows}")
    return path


def _write_mask(path: Path, mask: np.ndarray) -> Path:
    Image.fromarray(mask).save(path)
    return path


def _write_split(split: Path, masks: int) -> Path:
    """Write a split with ``masks`` images, each with a two-class mask."""
    split.mkdir(parents=True, exist_ok=True)
    _write_classes(split)
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    for index in range(masks):
        _image(split / f"img_{index}.jpg", size=(2, 2))
        _write_mask(split / f"img_{index}_mask.png", mask)
    return split


def _count_calls(monkeypatch: pytest.MonkeyPatch, method: str) -> list[Path]:
    """Record the paths ``Path.<method>`` is called on."""
    calls: list[Path] = []
    original = getattr(Path, method)

    def counting(self: Path, *args: Any, **kwargs: Any) -> Any:
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, method, counting)
    return calls


def _count_imreads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the paths ``cv2.imread`` is called on."""
    calls: list[str] = []
    original = cv2.imread

    def counting(path: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imread", counting)
    return calls


def test_segmentation_mask_directory_parser(tmp_path: Path):
    parser = _plugin(SegmentationMaskDirectoryParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    classes = split / "_classes.csv"
    classes.write_text("Pixel Value, Class\n0,background\n1,bird\n")
    orphan_mask = split / "orphan_mask.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(orphan_mask)
    assert parser.validate_split(split) is None
    orphan_mask.unlink()

    image = _image(split / "sample.jpg", size=(2, 2))
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    Image.fromarray(mask).save(split / "sample_mask.png")
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._split_records(**kwargs))

    assert len(records) == 2
    assert {record["annotation"]["class"] for record in records} == {
        "background",
        "bird",
    }
    assert {Path(record["file"]) for record in records} == {image.resolve()}

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "_classes.csv").write_text("Pixel Value, Class\n0,bird\n")
    _image(broken / "image.jpg")
    (broken / "image_mask.png").write_text("not an image")
    # No mask is decoded to start streaming, but an undecodable one must
    # still fail the parse itself: failing later would leave a
    # half-imported dataset behind.
    with pytest.raises(ValueError, match="Failed to read mask"):
        parser._split_records(broken, broken, broken / "_classes.csv")


def test_png_images_validate_like_they_parse(tmp_path: Path):
    """A split whose images are not `.jpg` must still be recognized.

    Regression: `validate_split` looked every image up as `{stem}.jpg`
    while the pairing accepts any suffix through the prefix rule, so a
    split of `.png` images failed validation, `detect` returned `None`,
    and the layout was reported as "not in expected format" even though
    the parse handles it.
    """
    parser = _plugin(SegmentationMaskDirectoryParser)
    split = tmp_path / "train"
    split.mkdir()
    _write_classes(split)
    image = _image(split / "sample.png", size=(2, 2))
    _write_mask(
        split / "sample_mask.png",
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
    )

    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._split_records(**kwargs))
    assert {Path(record["file"]) for record in records} == {image.resolve()}


def test_every_mask_is_decoded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard against decoding a mask more than once per parse.

    The file list used to be built by draining a first copy of the
    record generator, which decoded every mask twice. It is now derived
    from the mask-to-image pairing alone, so listing the files decodes
    nothing at all and a full parse reads each mask exactly once.
    """
    split = _write_split(tmp_path / "train", 6)
    parser = _plugin(SegmentationMaskDirectoryParser)
    calls = _count_imreads(monkeypatch)

    files = parser._split_files(split, split, split / "_classes.csv")
    assert len(files) == 6
    assert calls == []

    records = _records(
        parser._split_records(split, split, split / "_classes.csv")
    )
    assert len(records) == 12
    assert len(calls) == 6


def test_records_are_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Guard against decoding masks before the records are consumed.

    The records are a lazy iterator, so no mask may be decoded until a
    record is actually pulled, and pulling one record may only decode
    one mask.
    """
    split = _write_split(tmp_path / "train", 6)
    parser = _plugin(SegmentationMaskDirectoryParser)
    calls = _count_imreads(monkeypatch)

    records = parser._split_records(split, split, split / "_classes.csv")
    assert calls == []

    next(records)
    assert len(calls) == 1


def test_parse_streams_the_detected_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard against walking the source twice to tag records.

    ``detect`` recognizes the splits once and hands the layout to
    ``parse``, which streams every split's records tagged with the split
    they came from. Nothing is decoded before the records are pulled,
    and the whole source is walked a single time.
    """
    for name in ("train", "valid", "test"):
        _write_split(tmp_path / name, 2)
    layout = SegmentationMaskDirectoryParser.detect(tmp_path)
    assert layout is not None
    assert layout.split_names == ["train", "val", "test"]

    parser = _plugin(SegmentationMaskDirectoryParser)
    calls = _count_imreads(monkeypatch)
    result = parser.parse(tmp_path, layout)
    assert calls == []

    tagged = [next(result.records)]
    assert len(calls) == 1

    tagged.extend(result.records)
    assert Counter(split for split, _ in tagged) == {
        "train": 4,
        "val": 4,
        "test": 4,
    }
    assert len(calls) == 6
    assert result.skeletons == {}


def test_files_are_enumerated_without_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard against paying a parse to answer count-based ratios.

    Count-based ``split_ratios`` need each split's files before anything
    is imported. The masks name their images, so the parser answers
    that from the pairing, per split and without decoding a mask - a
    parser returning ``None`` here would be parsed a second time.
    """
    for name in ("train", "valid"):
        _write_split(tmp_path / name, 3)
    layout = SegmentationMaskDirectoryParser.detect(tmp_path)
    assert layout is not None

    parser = _plugin(SegmentationMaskDirectoryParser)
    calls = _count_imreads(monkeypatch)
    enumerated = parser.enumerate_files(tmp_path, layout)

    assert calls == []
    assert enumerated is not None
    assert {name: set(files) for name, files in enumerated.items()} == {
        canonical: {
            (tmp_path / directory / f"img_{index}.jpg").resolve()
            for index in range(3)
        }
        for canonical, directory in (("train", "train"), ("val", "valid"))
    }


@pytest.mark.parametrize("method", ["glob", "resolve"])
def test_directory_work_does_not_scale_with_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
):
    """Guard against per-mask globbing and per-mask path resolving.

    Every mask used to look its image up with its own
    ``image_dir.glob`` and resolve the match with its own ``realpath``.
    Both are now answered by a single listing and a single resolve of
    the image directory, so their number stays the same however many
    masks the split holds.
    """
    counts = []
    for masks in (2, 12):
        split = _write_split(tmp_path / f"split_{masks}", masks)
        parser = _plugin(SegmentationMaskDirectoryParser)
        with monkeypatch.context() as patched:
            calls = _count_calls(patched, method)
            records = _records(
                parser._split_records(split, split, split / "_classes.csv")
            )
            assert len(records) == 2 * masks
            counts.append(len(calls))

    assert counts[0] == counts[1]


def test_validation_does_not_stat_every_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard against checking each mask's image with its own stat.

    ``validate_split`` used to call ``Path.exists`` once per mask; one
    listing of the split answers all of them.
    """
    counts = []
    for masks in (2, 12):
        split = _write_split(tmp_path / f"split_{masks}", masks)
        parser = _plugin(SegmentationMaskDirectoryParser)
        with monkeypatch.context() as patched:
            calls = _count_calls(patched, "exists")
            assert parser.validate_split(split) is not None
            counts.append(len(calls))

    assert counts[0] == counts[1]


def test_mask_annotations_match_the_reference_construction(tmp_path: Path):
    """Guard against changing what the emitted masks contain.

    The per-class masks are built by casting a comparison instead of
    indexing a zero-filled array, and the pixel values present are found
    by counting bins instead of sorting. Both must reproduce, exactly,
    what the array-filling version produced: one ``uint8`` 0/1 array per
    value present, in ascending value order.
    """
    split = tmp_path / "train"
    split.mkdir()
    _write_classes(split)
    _image(split / "sample.jpg", size=(4, 4))
    raw = np.array(
        [[3, 3, 0, 0], [1, 3, 0, 1], [0, 0, 0, 3], [1, 1, 3, 0]],
        dtype=np.uint8,
    )
    _write_mask(split / "sample_mask.png", raw)

    parser = _plugin(SegmentationMaskDirectoryParser)
    records = _records(
        parser._split_records(split, split, split / "_classes.csv")
    )

    values = np.unique(raw)
    assert [record["annotation"]["class"] for record in records] == [
        CLASSES[value] for value in values
    ]
    for record, value in zip(records, values, strict=True):
        expected = np.zeros_like(raw)
        expected[raw == value] = 1
        mask = record["annotation"]["segmentation"]["mask"]
        assert mask.dtype == expected.dtype
        np.testing.assert_array_equal(mask, expected)


def test_per_class_masks_avoid_full_mask_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard against sorting the mask and zero-filling per class.

    The values present are counted rather than sorted out of the whole
    mask, and each per-class mask is cast from the comparison instead
    of being written twice: once as zeros, once through a boolean
    index. Both replaced calls are made unavailable here.
    """

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the whole mask is scanned again")

    split = tmp_path / "train"
    split.mkdir()
    _write_classes(split)
    _image(split / "sample.jpg", size=(2, 2))
    _write_mask(
        split / "sample_mask.png", np.array([[0, 2], [2, 0]], dtype=np.uint8)
    )

    parser = _plugin(SegmentationMaskDirectoryParser)
    monkeypatch.setattr(np, "unique", unavailable)
    monkeypatch.setattr(np, "zeros_like", unavailable)
    records = _records(
        parser._split_records(split, split, split / "_classes.csv")
    )

    assert [record["annotation"]["class"] for record in records] == [
        "background",
        "cat",
    ]


def test_mask_without_an_image_raises(tmp_path: Path):
    """Guard against changing how an unpaired mask fails.

    The image lookup runs while the masks are paired up front, and an
    exhausted lookup raises ``StopIteration`` there. Building the
    pairing inside a generator is what turns that into the
    ``RuntimeError`` callers saw when the lookup lived in the record
    generator - for listing the files as much as for parsing them.
    """
    split = tmp_path / "train"
    split.mkdir()
    _write_classes(split)
    _write_mask(split / "lonely_mask.png", np.zeros((2, 2), dtype=np.uint8))

    parser = _plugin(SegmentationMaskDirectoryParser)
    with pytest.raises(RuntimeError, match="generator raised StopIteration"):
        parser._split_records(split, split, split / "_classes.csv")
    with pytest.raises(RuntimeError, match="generator raised StopIteration"):
        parser._split_files(split, split, split / "_classes.csv")


def test_symlinked_images_are_resolved(tmp_path: Path):
    """Guard against skipping ``realpath`` for a symlinked image.

    Only a name that is not itself a symlink can be appended to the
    resolved image directory; a symlinked image still has to be
    resolved on its own, or records would point at the link.
    """
    target = _image(tmp_path / "elsewhere" / "real.jpg", size=(2, 2))
    split = tmp_path / "train"
    split.mkdir()
    _write_classes(split)
    _symlink(split / "sample.jpg", target)
    _write_mask(split / "sample_mask.png", np.zeros((2, 2), dtype=np.uint8))

    parser = _plugin(SegmentationMaskDirectoryParser)
    records = _records(
        parser._split_records(split, split, split / "_classes.csv")
    )

    assert [Path(record["file"]) for record in records] == [target.resolve()]


def test_dangling_image_symlink_is_not_a_valid_split(tmp_path: Path):
    """Guard against treating a listed name as an existing image.

    ``validate_split`` reads the split's directory entries instead of
    stating each image, and a dangling symlink is listed but does not
    exist.
    """
    split = tmp_path / "train"
    split.mkdir()
    _write_classes(split)
    _symlink(split / "sample.jpg", split / "missing.jpg")
    _write_mask(split / "sample_mask.png", np.zeros((2, 2), dtype=np.uint8))

    parser = _plugin(SegmentationMaskDirectoryParser)
    assert parser.validate_split(split) is None


def test_glob_metacharacters_keep_glob_semantics(tmp_path: Path):
    """Guard against looking a pattern-like stem up as a literal name.

    ``a[b]_mask.png`` used to look its image up by globbing
    ``a[b].*``, which matches ``ab.jpg`` and not ``a[b].jpg``. A stem
    holding glob metacharacters therefore may not be resolved through
    the name index.
    """
    split = tmp_path / "train"
    split.mkdir()
    _write_classes(split)
    _image(split / "ab.jpg", size=(2, 2))
    _image(split / "a[b].jpg", size=(2, 2))
    _write_mask(split / "a[b]_mask.png", np.zeros((2, 2), dtype=np.uint8))

    parser = _plugin(SegmentationMaskDirectoryParser)
    records = _records(
        parser._split_records(split, split, split / "_classes.csv")
    )

    assert [Path(record["file"]) for record in records] == [
        (split / "ab.jpg").resolve()
    ]
