"""Benchmark datasets for the Darknet parser.

The parser accepts two mutually exclusive layouts: a split layout with
``train``/``valid``/``test`` subdirectories, and a flat layout where the
dataset root is itself a single split. Only the split layout is built
here because it is what Roboflow exports and because
`SplitParserPlugin.parse` only falls back to the flat layout when no
split directory is recognized, so a single dataset cannot exercise both.

Darknet keeps one small text file per image next to the image, so the
format is light enough to generate at the full benchmark scale without
shrinking it.
"""

from pathlib import Path

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    bbox,
    benchmark_dataset,
    class_names,
    write_images,
)

SPLITS = ("train", "valid", "test")

# Every twentieth image gets no label file at all, and the one after it
# gets an empty label file. Both make the parser emit a single record
# with no annotation, which is the only branch besides detection.
UNLABELLED_STRIDE = 20


def _detection_payload(scale: Scale) -> str:
    """Return the Darknet label file shared by every annotated image."""
    lines = []
    for index in range(scale.annotations_per_image):
        x, y, w, h = bbox(index)
        class_id = index % scale.classes
        lines.append(f"{class_id} {x + w / 2} {y + h / 2} {w} {h}")
    return "\n".join(lines) + "\n"


def _write_split(
    root: Path,
    split: str,
    scale: Scale,
    payload: str,
) -> tuple[int, int]:
    """Write one Darknet split.

    Args:
        root: Root of the generated dataset.
        split: Name of the split directory to create.
        scale: How much data to write.
        payload: Label file contents used for annotated images.

    Returns:
        The number of images written and the number of records the
        parser is expected to yield for them.

    """
    directory = root / split
    names = [f"img_{index}" for index in range(scale.images)]

    # A slice of the images is stored as PNG so that the parser's image
    # suffix filter is exercised beyond the usual `.jpg`.
    jpg = [
        name
        for index, name in enumerate(names)
        if index % UNLABELLED_STRIDE != 2
    ]
    png = [
        name
        for index, name in enumerate(names)
        if index % UNLABELLED_STRIDE == 2
    ]
    write_images(directory, jpg, scale.image_size)
    write_images(directory, png, scale.image_size, suffix=".png")

    (directory / "_darknet.labels").write_text(
        "\n".join(class_names(scale.classes)) + "\n"
    )

    unlabelled = 0
    empty = 0
    for index, name in enumerate(names):
        kind = index % UNLABELLED_STRIDE
        if kind == 0:
            unlabelled += 1
            continue
        if kind == 1:
            empty += 1
            (directory / f"{name}.txt").write_text("")
            continue
        (directory / f"{name}.txt").write_text(payload)

    annotated = len(names) - unlabelled - empty
    records = annotated * scale.annotations_per_image + unlabelled + empty
    return len(names), records


@benchmark_dataset("darknet")
def build_detection(root: Path, scale: Scale) -> BenchmarkCase:
    """Darknet detection labels across three splits."""
    payload = _detection_payload(scale)
    totals = [_write_split(root, split, scale, payload) for split in SPLITS]
    images = sum(count for count, _ in totals)
    records = sum(count for _, count in totals)

    return BenchmarkCase(
        dataset_type="darknet",
        source=root,
        features=(
            "detection",
            "multi-split",
            "images and labels side by side",
            "_darknet.labels class file",
            "missing label files",
            "empty label files",
            "mixed image suffixes",
        ),
        images=images,
        expected_records=records,
        expected_files=images,
    )
