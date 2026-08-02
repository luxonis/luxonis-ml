"""Benchmark datasets for the Ultralytics NDJSON parser.

Every dataset is a single NDJSON manifest describing three splits, so the
builders share `_write_dataset` and differ only in the header and in the
annotation payload attached to each image record.

Between them the three datasets cover the parser's whole surface: both
source layouts (a directory holding exactly one manifest, and the
manifest file itself), both `class_names` spellings (list and mapping),
all three annotation families, records without annotations, repeated
image paths, non-image lines, blank lines, a byte-order mark, all three
`file` spellings (relative POSIX, relative Windows, absolute), every
split name the parser normalizes, images referenced by `file://` URL and
a missing image reporting one issue.

Two features are deliberately left out:

- `kpt_shape = [n, 2]`, where the parser fills the visibility in itself.
  `kpt_shape` is a header field, so it cannot coexist with the 3-value
  layout in one manifest; the keypoint dataset carries the explicit
  `[n, 3]` shape an Ultralytics pose export writes, and the mixed
  dataset omits `kpt_shape` entirely so the inference branch runs too.
- `http(s)`, `gs` and `s3` image URLs, and therefore the `reuse_cached`
  keyword that only guards the remote download directory. Those need
  the network; the `file://` records exercise the same download path.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    bbox,
    benchmark_dataset,
    class_names,
    keypoint_names,
    keypoints,
    polygon,
    write_images,
)

SPLITS = ("train", "valid", "test")
"""Split names written to the manifest; ``valid`` normalizes to ``val``."""

EMPTY_EVERY = 10
"""One in this many images is written without any annotation."""

REMOTE_IMAGES = 3
"""Image records fetched through a ``file://`` URL."""

Annotations = dict[str, list[list[float]]]


class _Counts(NamedTuple):
    """What a generated manifest makes the parser report.

    Attributes:
        images: Number of images written to disk.
        records: Number of records the parser yields.
        files: Number of files the parser reports.
        issues: Number of skipped annotations the parser reports.

    """

    images: int
    records: int
    files: int
    issues: int


def _boxes(scale: Scale) -> list[list[float]]:
    """Return one ``boxes`` payload in Ultralytics center form."""
    payload = []
    for index in range(scale.annotations_per_image):
        x, y, w, h = bbox(index)
        payload.append([index % scale.classes, x + w / 2, y + h / 2, w, h])
    return payload


def _segments(scale: Scale) -> list[list[float]]:
    """Return one ``segments`` payload of flattened polygons."""
    payload = []
    for index in range(scale.annotations_per_image):
        points = polygon(index, scale.polygon_points)
        flat = [value for point in points for value in point]
        payload.append([index % scale.classes, *flat])
    return payload


def _pose(scale: Scale) -> list[list[float]]:
    """Return one ``pose`` payload of boxes with visible keypoints."""
    payload = []
    for index in range(scale.annotations_per_image):
        x, y, w, h = bbox(index)
        flat = [
            value
            for keypoint in keypoints(index, scale.keypoints)
            for value in keypoint
        ]
        payload.append(
            [index % scale.classes, x + w / 2, y + h / 2, w, h, *flat]
        )
    return payload


def _manifest_file(root: Path, split: str, name: str, index: int) -> str:
    """Return a ``file`` value, rotating over the accepted spellings.

    The parser resolves relative POSIX paths, relative Windows paths and
    absolute paths to the same image, so every third record uses each.
    """
    relative = f"images/{split}/{name}.jpg"
    variant = index % 3
    if variant == 0:
        return relative
    if variant == 1:
        return relative.replace("/", "\\")
    return str((root / relative).resolve())


def _image_record(
    file: str,
    split: str | None,
    scale: Scale,
    annotations: Annotations | None,
) -> dict[str, Any]:
    """Build one image record.

    Args:
        file: Value of the record's ``file`` field.
        split: Split name, or None to omit the key so that the parser
            falls back to ``"train"``.
        scale: Scale the record belongs to, giving the image size.
        annotations: Annotation payload, or None to omit the key.

    Returns:
        The record, ready to be serialized as one manifest line.

    """
    record: dict[str, Any] = {
        "type": "image",
        "file": file,
        "width": scale.image_size,
        "height": scale.image_size,
    }
    if split is not None:
        record["split"] = split
    if annotations is not None:
        record["annotations"] = annotations
    return record


def _write_dataset(
    root: Path,
    scale: Scale,
    *,
    header: dict[str, Any],
    annotations: Annotations,
    remote_images: int = 0,
    encoding: str = "utf-8",
) -> _Counts:
    """Write the images and the manifest of one benchmark dataset.

    Args:
        root: Directory the dataset is written to.
        scale: Size of each of the three splits.
        header: Fields of the leading ``dataset`` record.
        annotations: Payload attached to every annotated image.
        remote_images: How many extra records point at a ``file://``
            URL instead of a path on disk.
        encoding: Manifest encoding; ``utf-8-sig`` prepends a BOM.

    Returns:
        The counts the parser must report for the written manifest.

    """
    names = [f"img_{index}" for index in range(scale.images)]
    per_image = sum(len(family) for family in annotations.values())
    remote = min(remote_images, scale.images)

    # A blank line and a non-image line: both are skipped, and the parser
    # must not let them shift the records that follow.
    lines = ["", json.dumps({"type": "dataset", **header})]
    lines.append(json.dumps({"type": "metadata", "note": "not an image"}))

    for split in SPLITS:
        write_images(root / "images" / split, names, scale.image_size)
        for index, name in enumerate(names):
            empty = index % EMPTY_EVERY == EMPTY_EVERY - 1
            lines.append(
                json.dumps(
                    _image_record(
                        _manifest_file(root, split, name, index),
                        split,
                        scale,
                        {} if empty else annotations,
                    )
                )
            )

    first = f"images/{SPLITS[0]}/{names[0]}.jpg"
    extras = [
        # A repeated image path under every remaining split spelling:
        # both records are yielded, but the image is reported once and
        # only in the split it was first seen in. An absent split and an
        # unknown one both fall back to `train`.
        _image_record(first, "validation", scale, annotations),
        _image_record(first, None, scale, annotations),
        _image_record(first, "unknown", scale, {}),
        _image_record(first, "val", scale, None),
        # The only skipped record, and the only reported issue.
        _image_record(
            f"images/{SPLITS[0]}/img_missing.jpg", "train", scale, annotations
        ),
    ]
    for index in range(remote):
        source = root / "images" / SPLITS[0] / f"{names[index]}.jpg"
        record = _image_record(
            f"remote_{index}.jpg", "train", scale, annotations
        )
        record["url"] = source.resolve().as_uri()
        extras.append(record)

    lines.append("")
    lines.extend(json.dumps(record) for record in extras)
    lines.append(json.dumps({"type": "annotation-definitions"}))
    (root / "dataset.ndjson").write_text(
        "\n".join(lines) + "\n", encoding=encoding
    )

    empty_images = scale.images // EMPTY_EVERY
    annotated = scale.images - empty_images
    return _Counts(
        images=len(SPLITS) * scale.images,
        records=(
            len(SPLITS) * (annotated * per_image + empty_images)
            + (2 + remote) * per_image
            + 2
        ),
        files=len(SPLITS) * scale.images + remote,
        issues=1,
    )


@benchmark_dataset("ultralytics-ndjson")
def build_mixed(root: Path, scale: Scale) -> BenchmarkCase:
    """Every annotation family in one manifest, parsed from a directory.

    Boxes, polygons and poses on every image make this the heaviest of
    the three manifests, and the only one whose images are also reached
    through a URL. The full scale still generates in about a second, so
    it is not shrunk.
    """
    counts = _write_dataset(
        root,
        scale,
        # No `kpt_shape`: the parser infers it from the 3-value layout.
        header={"class_names": class_names(scale.classes)},
        annotations={
            "boxes": _boxes(scale),
            "segments": _segments(scale),
            "pose": _pose(scale),
        },
        remote_images=REMOTE_IMAGES,
        encoding="utf-8-sig",
    )

    return BenchmarkCase(
        dataset_type="ultralytics-ndjson",
        source=root,
        features=(
            "detection",
            "instance segmentation",
            "keypoints",
            "inferred kpt_shape",
            "file:// images",
            "directory source",
            "byte-order mark",
        ),
        images=counts.images,
        expected_records=counts.records,
        expected_files=counts.files,
        expected_issues=counts.issues,
    )


@benchmark_dataset("ultralytics-ndjson-instancesegmentation")
def build_instance_segmentation(root: Path, scale: Scale) -> BenchmarkCase:
    """Polygon payloads, parsed from the manifest file itself."""
    counts = _write_dataset(
        root,
        scale,
        header={
            "class_names": {
                str(index): name
                for index, name in enumerate(class_names(scale.classes))
            }
        },
        annotations={"segments": _segments(scale)},
    )

    return BenchmarkCase(
        dataset_type="ultralytics-ndjson-instancesegmentation",
        source=root / "dataset.ndjson",
        features=(
            "instance segmentation",
            "fitted bounding boxes",
            "mapping class names",
            "manifest file source",
        ),
        images=counts.images,
        expected_records=counts.records,
        expected_files=counts.files,
        expected_issues=counts.issues,
    )


@benchmark_dataset("ultralytics-ndjson-keypoints")
def build_keypoints(root: Path, scale: Scale) -> BenchmarkCase:
    """Pose payloads with an explicit ``kpt_shape``."""
    counts = _write_dataset(
        root,
        scale,
        header={
            "class_names": class_names(scale.classes),
            "kpt_shape": [scale.keypoints, 3],
            "keypoints": keypoint_names(scale.keypoints),
        },
        annotations={"pose": _pose(scale)},
    )

    return BenchmarkCase(
        dataset_type="ultralytics-ndjson-keypoints",
        source=root,
        features=(
            "keypoints",
            "visibility flags",
            "explicit kpt_shape",
            "directory source",
        ),
        images=counts.images,
        expected_records=counts.records,
        expected_files=counts.files,
        expected_issues=counts.issues,
    )
