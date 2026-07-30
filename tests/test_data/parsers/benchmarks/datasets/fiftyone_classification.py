"""Benchmark datasets for the FiftyOne classification parser.

The parser accepts two mutually exclusive layouts: a split layout with
``train``/``validation``/``test`` subdirectories, and a flat layout with a
single ``data`` directory whose labels are additionally run through
`clean_imagenet_annotations`. One benchmark case has one source, so the
split layout is the one generated here: it is the representative FiftyOne
export shape and it additionally exercises split discovery, the
``validation`` to ``val`` canonicalization and the per-split output merge.
The flat layout, and with it the ImageNet label cleanup, is covered by the
functional tests instead.

The parser takes no optional parse arguments; everything it does is driven
by the layout, so the generated dataset covers its remaining behaviour
through content: mixed image extensions resolved by stem, images present in
``data`` that no label refers to, and labels referring to a stem that has no
image file, which is the parser's only reported issue.
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

SPLITS = ("train", "validation", "test")


def _write_split(root: Path, split: str, scale: Scale) -> tuple[int, int]:
    """Write one FiftyOne split directory.

    Args:
        root: Dataset root the split directory is created under.
        split: Name of the split directory.
        scale: Size of the generated split.

    Returns:
        The number of labels resolving to an image present in the split,
        and the number of labels referring to a missing image stem.

    """
    data_dir = root / split / "data"
    stems = [f"{split}_{index}" for index in range(scale.images)]

    # A tenth of the images use a different extension, so the parser has to
    # match labels to files by stem rather than by name.
    png_images = max(1, scale.images // 10)
    write_images(data_dir, stems[:png_images], scale.image_size, suffix=".png")
    write_images(data_dir, stems[png_images:], scale.image_size)
    # Dropped by the parser's extension filter.
    (data_dir / "export_manifest.txt").write_text(f"{split}\n")

    # Images the labels never mention are silently ignored by the parser.
    unlabeled = min(max(1, scale.images // 100), scale.images - 1)
    labeled = scale.images - unlabeled

    # Labels pointing at a stem with no image file, spread through the
    # label map so the generator skips them mid-iteration.
    dangling = max(1, scale.images // 200)
    stride = max(1, labeled // dangling)

    labels: dict[str, int] = {}
    written = 0
    for index in range(labeled):
        labels[stems[index]] = index % scale.classes
        if written < dangling and (index + 1) % stride == 0:
            labels[f"{split}_dangling_{written}"] = written % scale.classes
            written += 1

    (root / split / "labels.json").write_text(
        json.dumps({"classes": class_names(scale.classes), "labels": labels})
    )
    return labeled, written


@benchmark_dataset("fiftyone-classification")
def build_classification(root: Path, scale: Scale) -> BenchmarkCase:
    """Classification labels across the three FiftyOne splits."""
    counts = [_write_split(root, split, scale) for split in SPLITS]
    records = sum(labeled for labeled, _ in counts)
    dangling = sum(missing for _, missing in counts)

    return BenchmarkCase(
        dataset_type="fiftyone-classification",
        source=root,
        features=(
            "classification",
            "multi-split",
            "mixed image extensions",
            "unreferenced images",
            "dangling label stems",
        ),
        images=len(SPLITS) * scale.images,
        expected_records=records,
        expected_files=records,
        expected_issues=dangling,
    )
