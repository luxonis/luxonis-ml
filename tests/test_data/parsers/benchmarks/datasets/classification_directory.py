"""Benchmark datasets for the classification directory parser.

The parser accepts two mutually exclusive layouts at the same root: the
split layout (``train``/``valid``/``test``, each holding one directory
per class) and the flat layout (class directories directly in the root,
split randomly at parse time). Only one of them can be present, because
`SplitParserPlugin.parse` falls back to the flat layout only when no
split directory is recognized. The benchmark writes the split layout, as
that is the one Roboflow generates and the one that makes the parser
walk three separate trees; the flat layout runs the exact same
`_split_records` on a single directory and is covered by the functional
tests.

Everything else the parser distinguishes is exercised: the optional
``info.json`` metadata file that a split directory is allowed to hold,
the reserved directory names that are never treated as classes, the
non-recursive image listing, the case-insensitive suffix filter, and a
class directory holding no images at all.

Two parser behaviours have nothing to exercise here. The parser takes no
optional parse arguments -- `_split_records` accepts only the class
directory that `validate_split` hands it -- and it cannot emit an image
without an annotation, since an image is annotated by the directory
holding it and a loose file next to the class directories makes
`validate_split` reject the split outright. Classification is likewise
the only annotation type it produces, so `Scale.annotations_per_image`
does not apply: every image carries exactly one class.
"""

import json
from pathlib import Path

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    benchmark_dataset,
    class_names,
    write_images,
)

#: ``valid`` is the only spelling the parser probes for the validation
#: split; it is canonicalized to ``val`` in the combined output.
SPLITS = ("train", "valid", "test")

#: Cycled over the class directories. The parser matches suffixes
#: case-insensitively, so ``.JPG`` must be picked up just like ``.jpg``.
IMAGE_SUFFIXES = (".jpg", ".png", ".JPG")

#: Directory names the parser reserves for other layouts. Images below
#: them must never become records, however many there are.
RESERVED_DIRS = ("images", "labels", "masks")

#: Images written into each ignored directory. Kept small: they only
#: have to prove the directory is skipped, not to be walked quickly.
IGNORED_IMAGES = 4

#: Class directory that stays empty, standing in for a class declared by
#: the taxonomy but not represented in a split.
EMPTY_CLASS = "class_empty"


def _class_image_counts(scale: Scale) -> list[int]:
    """Return how many images each class directory holds.

    The images of a split are spread as evenly as possible over the
    class directories, so a split holds exactly ``scale.images`` images
    however the two numbers divide.
    """
    base, remainder = divmod(scale.images, scale.classes)
    return [base + int(index < remainder) for index in range(scale.classes)]


def _write_split(root: Path, split: str, scale: Scale) -> int:
    """Write one class-per-directory split and return its image count."""
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # The one file a split directory may hold. Any other file next to
    # the class directories makes the parser reject the split.
    (split_dir / "info.json").write_text(
        json.dumps({"split": split, "classes": scale.classes})
    )

    names = class_names(scale.classes)
    counts = _class_image_counts(scale)
    written = 0
    for index, (name, count) in enumerate(zip(names, counts, strict=True)):
        class_dir = split_dir / name
        write_images(
            class_dir,
            (f"{split}_{name}_{image}" for image in range(count)),
            scale.image_size,
            suffix=IMAGE_SUFFIXES[index % len(IMAGE_SUFFIXES)],
        )
        written += count
        # Filtered out by the suffix check, so it must not be counted.
        (class_dir / "notes.txt").write_text(f"{name}\n")

    # Listed but never recursed into, so these images are invisible.
    write_images(
        split_dir / names[0] / "nested",
        (f"nested_{image}" for image in range(IGNORED_IMAGES)),
        scale.image_size,
    )

    (split_dir / EMPTY_CLASS).mkdir(exist_ok=True)

    for reserved in RESERVED_DIRS:
        write_images(
            split_dir / reserved,
            (f"{reserved}_{image}" for image in range(IGNORED_IMAGES)),
            scale.image_size,
        )

    return written


@benchmark_dataset("clsdir")
def build_classification_directory(root: Path, scale: Scale) -> BenchmarkCase:
    """Class-per-directory images across three splits.

    The format is one file per image and no annotation files at all, so
    it is written at the full scale: generating it costs little more
    than the images every other benchmark writes anyway.
    """
    images = sum(_write_split(root, split, scale) for split in SPLITS)

    return BenchmarkCase(
        dataset_type="clsdir",
        source=root,
        # `images` counts the images the parser is expected to read; the
        # ignored decoys are deliberately left out of the throughput.
        images=images,
        features=(
            "classification",
            "multi-split",
            "class-per-directory",
            "info.json metadata",
            "reserved directory names",
            "non-recursive listing",
            "mixed image suffixes",
            "empty class directory",
        ),
        expected_records=images,
        expected_files=images,
    )
