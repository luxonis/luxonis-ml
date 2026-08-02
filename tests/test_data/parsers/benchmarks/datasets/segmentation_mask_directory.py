"""Benchmark dataset for the segmentation mask directory parser.

The parser is mask driven: it globs ``*_mask.*`` inside a split, reads
every match as a grayscale image and yields one record per pixel value
present, resolving the class name through ``_classes.csv``. The dataset
built here therefore exercises

* the Roboflow ``train``/``valid``/``test`` split layout,
* the ``_classes.csv`` pixel value to class name mapping, including the
  column name the parser expects to be prefixed with a space,
* segmentation masks paired with a class name, the two annotation kinds
  this parser emits,
* masks holding every class at once next to masks holding only the
  background class, the closest this format gets to an image without
  annotations, and
* the extension agnostic ``*_mask.*`` glob, by storing the ``test``
  split's masks as BMP while the other two use PNG.

The parser also accepts a bare split directory as its source, without
``train``/``valid``/``test`` subdirectories. That layout is mutually
exclusive with the one built here; the multi-split layout is both the
more representative one and the one that additionally exercises split
discovery and merging, so it is the one benchmarked.

Images without a matching mask are not written. The parser never lists
them - it walks masks, not images - so they would add nothing but noise
to the reported throughput.
"""

from collections.abc import Sequence
from pathlib import Path

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    benchmark_dataset,
    class_names,
    write_images,
    write_masks,
)

SPLITS = ("train", "valid", "test")

MASK_SUFFIXES = {"train": ".png", "valid": ".png", "test": ".bmp"}

# Generation is cheap here (0.5s for the 12000 files the common scale
# asks for), but the parse is not: every mask is decoded twice - once to
# collect the files, once to yield the records - and each pixel value
# present costs a full-size mask allocation, which puts one full-scale
# parse at ~27s against ~0.3s for the text-label parsers. Since the
# benchmark parses a case three times, a quarter of the common image
# count is what keeps this parser in the same order of magnitude as the
# rest while still writing 1500 images and yielding ~27000 records.
SHRINK_FACTOR = 4


def _mask_values(scale: Scale) -> list[int]:
    """Return the pixel values a fully populated mask contains.

    The values double as indices into ``_classes.csv``, so they start at
    the background class. They are capped by the image height because
    every value occupies one horizontal band, and a band has to be at
    least one row tall to survive.
    """
    return list(range(min(scale.classes, scale.image_size)))


def _background_only_images(scale: Scale) -> int:
    """Return how many images of a split get a background-only mask."""
    return max(1, scale.images // 10)


def _write_split(
    root: Path,
    split: str,
    scale: Scale,
    values: Sequence[int],
) -> int:
    """Write one split and return the records the parser yields for it.

    Args:
        root: Root of the generated dataset.
        split: Name of the split directory to write.
        scale: Size of the dataset to generate.
        values: Pixel values a fully populated mask contains.

    Returns:
        Number of records the parser yields for this split.

    """
    split_dir = root / split
    names = [f"img_{index}" for index in range(scale.images)]
    write_images(split_dir, names, scale.image_size)

    background = _background_only_images(scale)
    suffix = MASK_SUFFIXES[split]
    write_masks(
        split_dir,
        [f"{name}_mask" for name in names[background:]],
        scale.image_size,
        [(value, value, value) for value in values],
        suffix=suffix,
    )
    write_masks(
        split_dir,
        [f"{name}_mask" for name in names[:background]],
        scale.image_size,
        [(0, 0, 0)],
        suffix=suffix,
    )

    rows = "\n".join(
        f"{value},{name}"
        for value, name in enumerate(class_names(scale.classes))
    )
    # NOTE: the parser looks the class column up as " Class", space
    # included, the way Roboflow writes it.
    (split_dir / "_classes.csv").write_text(f"Pixel Value, Class\n{rows}\n")

    return (len(names) - background) * len(values) + background


@benchmark_dataset("segmask")
def build_segmentation_masks(root: Path, scale: Scale) -> BenchmarkCase:
    """Roboflow segmentation masks across three splits."""
    scale = scale.shrunk(SHRINK_FACTOR)
    values = _mask_values(scale)
    records = sum(_write_split(root, split, scale, values) for split in SPLITS)
    images = scale.images * len(SPLITS)

    return BenchmarkCase(
        dataset_type="segmask",
        source=root,
        features=(
            "segmentation",
            "classification",
            "multi-split",
            "mask decode",
            "class csv",
        ),
        images=images,
        expected_records=records,
        expected_files=images,
    )
