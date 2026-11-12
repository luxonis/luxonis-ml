"""Benchmark datasets for the TensorFlow CSV parser.

The parser accepts two mutually exclusive layouts: a directory holding
``train``/``valid``/``test`` splits, and a single split directory used as
the dataset root. Only the multi-split layout is generated here since it
is the one Roboflow exports and the one the parser documents; the
single-split layout runs the very same ``_split_records`` code, just
once.

Every split therefore exercises the full feature set the parser offers:
bounding boxes normalized by the per-row ``width``/``height`` columns,
the ``class`` column naming each box, images that no CSV row mentions,
image suffixes other than ``.jpg``, and the three filename spellings
``resolve_manifest_path`` understands (bare, Windows-relative and
absolute). Rows with an empty ``filename`` are dropped by the parser
without being reported, so they cost no expected issues.

The format is light per image -- one small JPEG plus one CSV line per
box -- so the datasets are generated at the full benchmark scale.
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

CSV_HEADER = "filename,width,height,class,xmin,ymin,xmax,ymax"


def _split_plan(scale: Scale) -> tuple[list[str], list[str], list[str]]:
    """Return the JPEG, PNG and unannotated image stems of one split.

    The PNG images prove the parser lists every image suffix OpenCV
    supports, and the unannotated ones make it emit a record with no
    annotation.
    """
    unannotated = max(1, scale.images // 10)
    annotated = max(1, scale.images - unannotated)
    png = max(1, annotated // 10)
    return (
        [f"img_{index}" for index in range(annotated - png)],
        [f"img_{index}" for index in range(annotated - png, annotated)],
        [f"empty_{index}" for index in range(unannotated)],
    )


def _row_suffixes(scale: Scale) -> list[list[str]]:
    """Return the CSV columns following the filename, one row per box.

    A box depends only on its index and on the class offset of the image
    holding it, so the columns are built once per offset and reused
    across images rather than formatted anew for every row. Rotating the
    offset per image puts every class name in the CSV.
    """
    size = scale.image_size
    names = class_names(scale.classes)
    variants = []
    for offset in range(scale.classes):
        rows = []
        for index in range(scale.annotations_per_image):
            x, y, w, h = bbox(index)
            rows.append(
                f",{size},{size},{names[(index + offset) % scale.classes]},"
                f"{round(x * size)},{round(y * size)},"
                f"{round((x + w) * size)},{round((y + h) * size)}"
            )
        variants.append(rows)
    return variants


def _reference(split: str, image: Path) -> str:
    """Return how ``image`` is spelled in the CSV of ``split``.

    Each split uses a different spelling so that one parse covers every
    branch of the manifest path resolution.
    """
    if split == "train":
        return image.name
    if split == "valid":
        return f".\\{image.name}"
    return str(image.resolve())


def _write_split(root: Path, split: str, scale: Scale) -> tuple[int, int]:
    """Write one split and return its image and record counts."""
    split_dir = root / split
    jpg_names, png_names, empty_names = _split_plan(scale)
    annotated = write_images(split_dir, jpg_names, scale.image_size)
    annotated += write_images(
        split_dir, png_names, scale.image_size, suffix=".png"
    )
    unannotated = write_images(split_dir, empty_names, scale.image_size)

    variants = _row_suffixes(scale)
    lines = [CSV_HEADER]
    for index, image in enumerate(annotated):
        reference = _reference(split, image)
        lines.extend(
            reference + suffix for suffix in variants[index % len(variants)]
        )
    # A row whose filename is empty; the parser drops it silently, so it
    # counts towards neither the records nor the reported issues.
    lines.append(variants[0][0])
    (split_dir / "_annotations.csv").write_text("\n".join(lines) + "\n")

    records = len(annotated) * scale.annotations_per_image
    return len(annotated) + len(unannotated), records + len(unannotated)


@benchmark_dataset("tfcsv")
def build_detection(root: Path, scale: Scale) -> BenchmarkCase:
    """Build detection labels in ``_annotations.csv``, one per split."""
    images = 0
    records = 0
    for split in SPLITS:
        split_images, split_records = _write_split(root, split, scale)
        images += split_images
        records += split_records

    return BenchmarkCase(
        dataset_type="tfcsv",
        source=root,
        features=(
            "detection",
            "multi-split",
            "unannotated images",
            "mixed image suffixes",
            "manifest path resolution",
        ),
        images=images,
        expected_records=records,
        expected_files=images,
    )
