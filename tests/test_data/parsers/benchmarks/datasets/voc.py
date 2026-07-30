"""Benchmark datasets for the voc parser.

The parser expects one XML annotation next to every image, so a split
directory holds ``img_0.jpg`` beside ``img_0.xml``. The generated images
fall into five deterministic groups that together cover everything the
parser reads:

* ``detection`` - every ``<object>`` carries a ``<bndbox>`` and becomes
  one record.
* ``mixed`` - only the first half of the objects are boxed; the rest are
  classification-style objects that the parser reads but never turns
  into a record.
* ``classification`` - no object is boxed, so the image yields a single
  record with ``annotation=None``.
* ``empty`` - no ``<object>`` at all, the second way to reach the
  ``annotation=None`` branch.
* ``missing`` - ``<filename>`` names an image that was never written,
  which is the parser's only skip path
  (``ParserIssue.MISSING_IMAGE``).

``<filename>`` values rotate between a bare name, a Windows-style
relative name and an absolute path so that `resolve_manifest_path` is
exercised in all three of its modes.

The parser also accepts a single flat directory with no split
subdirectories, because `SplitParserPlugin.parse` falls back to
validating ``source`` itself. That layout is mutually exclusive with the
split layout, so only the split layout - the one the parser documents -
is generated here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    bbox,
    benchmark_dataset,
    class_names,
    write_images,
)

SPLITS = ("train", "valid", "test")

Category = Literal["detection", "mixed", "classification", "empty", "missing"]


@dataclass(frozen=True)
class _Totals:
    """What one written split is expected to produce."""

    images: int
    records: int
    files: int
    issues: int


def _category(index: int) -> Category:
    """Return the annotation flavour written for image ``index``."""
    if index % 25 == 24:
        return "missing"
    if index % 10 == 5:
        return "empty"
    if index % 10 == 7:
        return "classification"
    if index % 10 == 3:
        return "mixed"
    return "detection"


def _boxed_objects(category: Category, scale: Scale) -> int:
    """Return how many objects of ``category`` carry a bounding box."""
    if category in {"classification", "empty"}:
        return 0
    if category == "mixed":
        return (scale.annotations_per_image + 1) // 2
    return scale.annotations_per_image


def _objects_xml(scale: Scale, boxed: int, names: list[str]) -> str:
    """Return the ``<object>`` elements shared by every image.

    Args:
        scale: Scale describing how many objects to emit.
        boxed: Number of leading objects that get a ``<bndbox>``.
        names: Class names to cycle through.

    Returns:
        The concatenated ``<object>`` elements.

    """
    size = scale.image_size
    parts: list[str] = []
    for index in range(scale.annotations_per_image):
        parts.append(
            "<object>"
            f"<name>{names[index % scale.classes]}</name>"
            "<pose>Unspecified</pose>"
            "<truncated>0</truncated>"
            "<difficult>0</difficult>"
        )
        if index < boxed:
            x, y, w, h = bbox(index)
            parts.append(
                "<bndbox>"
                f"<xmin>{round(x * size)}</xmin>"
                f"<ymin>{round(y * size)}</ymin>"
                f"<xmax>{round((x + w) * size)}</xmax>"
                f"<ymax>{round((y + h) * size)}</ymax>"
                "</bndbox>"
            )
        parts.append("</object>")
    return "".join(parts)


def _annotation_xml(split: str, filename: str, size: int, objects: str) -> str:
    """Return one VOC annotation document."""
    return (
        "<annotation>"
        f"<folder>{split}</folder>"
        f"<filename>{filename}</filename>"
        "<source><database>luxonis-benchmark</database></source>"
        f"<size><width>{size}</width><height>{size}</height>"
        "<depth>3</depth></size>"
        "<segmented>0</segmented>"
        f"{objects}"
        "</annotation>"
    )


def _reference(split_dir: Path, name: str, index: int) -> str:
    """Return the ``<filename>`` value pointing at ``name``.

    Real exports mix bare names with paths written on another platform,
    so the three forms `resolve_manifest_path` understands are rotated
    over the images.
    """
    if index % 20 == 11:
        return f".\\{name}.jpg"
    if index % 20 == 17:
        return (split_dir / f"{name}.jpg").resolve().as_posix()
    return f"{name}.jpg"


def _write_split(
    root: Path,
    split: str,
    scale: Scale,
    objects: dict[Category, str],
) -> _Totals:
    """Write one split of images with their XML annotations."""
    split_dir = root / split
    names = [f"img_{index}" for index in range(scale.images)]
    write_images(split_dir, names, scale.image_size)

    records = 0
    files = 0
    issues = 0
    for index, name in enumerate(names):
        category = _category(index)
        if category == "missing":
            filename = f"missing_{index}.jpg"
            issues += 1
        else:
            filename = _reference(split_dir, name, index)
            files += 1
            records += max(1, _boxed_objects(category, scale))
        (split_dir / f"{name}.xml").write_text(
            _annotation_xml(
                split, filename, scale.image_size, objects[category]
            )
        )
    return _Totals(len(names), records, files, issues)


@benchmark_dataset("voc")
def build_voc(root: Path, scale: Scale) -> BenchmarkCase:
    """Per-image VOC XML annotations across three splits.

    Kept at the full scale: an annotation is a single small XML document
    written from a prebuilt string, so generation stays cheap even with
    one file per image.
    """
    names = class_names(scale.classes)
    detection = _objects_xml(scale, _boxed_objects("detection", scale), names)
    objects: dict[Category, str] = {
        "detection": detection,
        "mixed": _objects_xml(scale, _boxed_objects("mixed", scale), names),
        "classification": _objects_xml(scale, 0, names),
        "empty": "",
        "missing": detection,
    }

    totals = [_write_split(root, split, scale, objects) for split in SPLITS]

    return BenchmarkCase(
        dataset_type="voc",
        source=root,
        features=(
            "detection",
            "classification-only objects",
            "images without annotations",
            "manifest path resolution",
            "missing images",
            "multi-split",
        ),
        images=sum(total.images for total in totals),
        expected_records=sum(total.records for total in totals),
        expected_files=sum(total.files for total in totals),
        expected_issues=sum(total.issues for total in totals),
    )
