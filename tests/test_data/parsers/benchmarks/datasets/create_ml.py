"""Benchmark datasets for the CreateML parser.

The CreateML export Roboflow produces is one directory per split, each
holding its images next to a single ``_annotations.createml.json``
manifest. The generated dataset uses that multi-split layout because it
is the one the parser documents. The parser also accepts a single
unsplit directory -- `SplitParserPlugin.parse` falls back to validating
the source itself when no split directory matches -- but the two layouts
are mutually exclusive for one source, so only the multi-split one is
benchmarked here.

Bounding boxes are the only annotation the format carries. The parser
collects the labels of a frame into a ``classes`` list as well, but that
list never reaches a record; the class name travels with the box
instead. Everything else the parser can do is exercised: images with an
empty annotation list, images present on disk but absent from the
manifest, manifest entries pointing into a subdirectory with both POSIX
and Windows separators, and entries referencing an image that does not
exist, which the parser reports as a skipped annotation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    bbox,
    benchmark_dataset,
    class_names,
    write_images,
)

SPLITS = ("train", "valid", "test")
NESTED_DIR = "nested"

# Roles are assigned by index modulo this period, so a split stays
# overwhelmingly ordinary -- as a real export is -- while still covering
# every path the parser can take, at any image count.
ROLE_PERIOD = 50
ROLE_EMPTY = 0
ROLE_UNREFERENCED = 1
ROLE_NESTED_POSIX = 2
ROLE_NESTED_WINDOWS = 3
ROLE_MISSING = 4


def _bbox_annotations(scale: Scale) -> list[dict[str, Any]]:
    """Return the annotations shared by every labelled image.

    CreateML stores the pixel coordinates of a box centre, so the
    normalized boxes from the harness are scaled back up to the size of
    the generated images.
    """
    labels = class_names(scale.classes)
    annotations: list[dict[str, Any]] = []
    for index in range(scale.annotations_per_image):
        x, y, w, h = bbox(index)
        annotations.append(
            {
                "label": labels[index % scale.classes],
                "coordinates": {
                    "x": (x + w / 2) * scale.image_size,
                    "y": (y + h / 2) * scale.image_size,
                    "width": w * scale.image_size,
                    "height": h * scale.image_size,
                },
            }
        )
    return annotations


@dataclass
class _Split:
    """The contents of one generated split.

    Attributes:
        root_images: Image stems written directly into the split.
        nested_images: Image stems written into the subdirectory.
        entries: Manifest entries, in image order.
        labelled: Images the parser yields records for.
        missing: Entries whose image does not exist.

    """

    root_images: list[str] = field(default_factory=list)
    nested_images: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    labelled: int = 0
    missing: int = 0


def _plan_split(scale: Scale, annotations: list[dict[str, Any]]) -> _Split:
    """Give every image index a role and build the manifest for it."""
    split = _Split()
    for index in range(scale.images):
        name = f"img_{index}"
        role = index % ROLE_PERIOD

        if role == ROLE_UNREFERENCED:
            split.root_images.append(name)
            continue

        if role == ROLE_MISSING:
            split.missing += 1
            split.entries.append(
                {"image": f"{name}.jpg", "annotations": annotations}
            )
            continue

        if role == ROLE_EMPTY:
            split.root_images.append(name)
            split.entries.append({"image": f"{name}.jpg", "annotations": []})
            continue

        if role == ROLE_NESTED_POSIX:
            split.nested_images.append(name)
            reference = f"{NESTED_DIR}/{name}.jpg"
        elif role == ROLE_NESTED_WINDOWS:
            split.nested_images.append(name)
            reference = f"{NESTED_DIR}\\{name}.jpg"
        else:
            split.root_images.append(name)
            reference = f"{name}.jpg"

        split.labelled += 1
        split.entries.append({"image": reference, "annotations": annotations})
    return split


def _write_split(
    root: Path, split_name: str, scale: Scale, split: _Split
) -> None:
    """Write one split's images and its CreateML manifest."""
    split_dir = root / split_name
    write_images(split_dir, split.root_images, scale.image_size)
    write_images(split_dir / NESTED_DIR, split.nested_images, scale.image_size)
    (split_dir / "_annotations.createml.json").write_text(
        json.dumps(split.entries)
    )


@benchmark_dataset("createml")
def build_create_ml(root: Path, scale: Scale) -> BenchmarkCase:
    """Build detection manifests for the three Roboflow splits."""
    annotations = _bbox_annotations(scale)
    split = _plan_split(scale, annotations)
    for split_name in SPLITS:
        _write_split(root, split_name, scale, split)

    splits = len(SPLITS)
    images = splits * (len(split.root_images) + len(split.nested_images))
    labelled = splits * split.labelled

    return BenchmarkCase(
        dataset_type="createml",
        source=root,
        features=(
            "detection",
            "multi-split",
            "roboflow-layout",
            "nested image paths",
            "windows-style paths",
            "unannotated images",
            "unreferenced images",
            "missing image reporting",
        ),
        images=images,
        expected_records=labelled * scale.annotations_per_image,
        expected_files=labelled,
        expected_issues=splits * split.missing,
    )
