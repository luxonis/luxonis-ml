"""Benchmark dataset for the COCO parser.

The generated dataset uses the FiftyOne layout — ``<split>/data`` holding
the images next to a ``<split>/labels.json`` annotation file, with the
splits named ``train`` / ``validation`` / ``test``. That layout is the
one the parser's optional arguments apply to, so it is the more
representative of the two. The Roboflow layout the same parser accepts
(``_annotations.coco.json`` sitting directly next to the images, with
``valid`` instead of ``validation``) is detected from the very same
directory names and is therefore mutually exclusive with it; only one
builder may be registered per dataset type, so the Roboflow layout and
the single-split source are left to the functional tests.

Two ``parse`` options are deliberately left at their defaults for the
same reason: ``use_keypoint_ann`` with ``keypoint_ann_paths`` replaces
every split's annotation file with one from the official COCO ``raw/``
directory, and ``split_val_to_test`` only changes anything when the
``test`` split contributes no images at all. Both would take data away
from the parse the benchmark is meant to time.

Everything else the parser understands is exercised: polygon and
multi-polygon segmentation, compressed RLE segmentation, keypoints with
category skeletons, bbox-only instances, annotations without an ``id``
(the parser assigns fallback instance ids), images that carry no
annotation at all, an annotation-free split whose categories live under
``info``, ``iscrowd`` annotations and dangling image references (both
reported as issues), and the hard-coded image list that
``clean_annotations`` strips from the train split.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pycocotools.mask as mask_util

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

#: One of the file names ``COCOParser.clean_annotations`` drops from the
#: train split, so the cleaning pass really rewrites the annotations.
CLEANED_STEM = "000000341448"

#: Every tenth validation image is written without any annotation, which
#: is the only situation in which the parser emits a record whose
#: ``annotation`` is ``None``.
EMPTY_EVERY = 10


class _SplitSummary(NamedTuple):
    """Counts describing one generated split.

    Attributes:
        images: Number of image files written to disk.
        records: Number of records the parser yields for the split.
        files: Number of distinct files the parser reports.
        issues: Number of skipped items the parser reports.

    """

    images: int
    records: int
    files: int
    issues: int


def _rle_mask(size: int) -> dict[str, Any]:
    """Return a compressed RLE mask, encoded a single time."""
    mask = np.zeros((size, size), dtype=np.uint8, order="F")
    mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
    counts = mask_util.encode(mask)["counts"]
    return {
        "size": [size, size],
        # `pycocotools` hands back raw bytes, which JSON cannot hold.
        "counts": counts.decode() if isinstance(counts, bytes) else counts,
    }


def _pixel_bbox(index: int, scale: Scale) -> list[float]:
    """Return a deterministic ``(x, y, w, h)`` box in pixels."""
    return [round(value * scale.image_size, 2) for value in bbox(index)]


def _flat_polygon(index: int, scale: Scale) -> list[float]:
    """Return one polygon flattened to pixel coordinates."""
    return [
        round(value * scale.image_size, 2)
        for point in polygon(index, scale.polygon_points)
        for value in point
    ]


def _flat_keypoints(index: int, scale: Scale) -> list[int]:
    """Return keypoints flattened to the COCO ``x, y, v`` triples."""
    return [
        value
        for x, y, visibility in keypoints(index, scale.keypoints)
        for value in (
            round(x * scale.image_size),
            round(y * scale.image_size),
            visibility,
        )
    ]


def _segmentation(
    index: int, scale: Scale, rle: dict[str, Any]
) -> list[list[float]] | dict[str, Any] | None:
    """Return one of the segmentation shapes the parser accepts.

    The variants cycle through a single polygon, a multi-polygon that
    forces the parser to merge several RLEs, a pre-encoded RLE mask and
    no segmentation at all.
    """
    variant = index % 4
    if variant == 0:
        return [_flat_polygon(index, scale)]
    if variant == 1:
        return [_flat_polygon(index, scale), _flat_polygon(index + 3, scale)]
    if variant == 2:
        return rle
    return None


def _annotation(
    index: int,
    scale: Scale,
    rle: dict[str, Any],
    *,
    image_id: int,
    annotation_id: int | None,
    with_keypoints: bool,
) -> dict[str, Any]:
    """Return one COCO instance annotation."""
    x, y, width, height = _pixel_bbox(index, scale)
    annotation: dict[str, Any] = {
        "image_id": image_id,
        "category_id": index % scale.classes + 1,
        "bbox": [x, y, width, height],
        "area": round(width * height, 2),
        "iscrowd": 0,
    }
    if annotation_id is not None:
        annotation["id"] = annotation_id

    segmentation = _segmentation(index, scale, rle)
    if segmentation is not None:
        annotation["segmentation"] = segmentation

    if with_keypoints:
        annotation["keypoints"] = _flat_keypoints(index, scale)
        annotation["num_keypoints"] = scale.keypoints

    return annotation


def _image_entry(
    image_id: int, file_name: str, scale: Scale
) -> dict[str, Any]:
    """Return one COCO image entry."""
    return {
        "id": image_id,
        "file_name": file_name,
        "height": scale.image_size,
        "width": scale.image_size,
    }


def _categories(scale: Scale, *, with_skeleton: bool) -> list[dict[str, Any]]:
    """Return the category list, optionally carrying a skeleton.

    A category with both ``keypoints`` and ``skeleton`` makes the parser
    build skeleton metadata — and, as a side effect, drop images that
    have no annotations.
    """
    names = keypoint_names(scale.keypoints)
    skeleton = [[edge + 1, edge + 2] for edge in range(scale.keypoints - 1)]
    categories = []
    for category_id, name in enumerate(class_names(scale.classes), start=1):
        category: dict[str, Any] = {
            "id": category_id,
            "name": name,
            "supercategory": "object",
        }
        if with_skeleton:
            category["keypoints"] = names
            category["skeleton"] = skeleton
        categories.append(category)
    return categories


def _write_labels(split_dir: Path, payload: dict[str, Any]) -> None:
    """Write one ``labels.json`` annotation file."""
    with open(split_dir / "labels.json", "w") as file:
        json.dump(payload, file)


def _write_train_split(
    root: Path, scale: Scale, rle: dict[str, Any]
) -> _SplitSummary:
    """Write the train split: every annotation type plus skeletons.

    The split also holds one image the parser's ``clean_annotations``
    pass removes by file name, one ``iscrowd`` annotation and one image
    entry pointing at a file that was never written.
    """
    stems = [f"img_{index}" for index in range(scale.images)]
    write_images(
        root / "train" / "data", [*stems, CLEANED_STEM], scale.image_size
    )

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1_000_000

    for image_id, stem in enumerate(stems, start=1):
        images.append(_image_entry(image_id, f"{stem}.jpg", scale))
        for index in range(scale.annotations_per_image):
            annotations.append(
                _annotation(
                    index,
                    scale,
                    rle,
                    image_id=image_id,
                    annotation_id=annotation_id,
                    with_keypoints=index % 2 == 0,
                )
            )
            annotation_id += 1

    crowd = _annotation(
        0,
        scale,
        rle,
        image_id=1,
        annotation_id=annotation_id,
        with_keypoints=False,
    )
    crowd["iscrowd"] = 1
    annotations.append(crowd)
    annotation_id += 1

    cleaned_id = len(stems) + 1
    images.append(_image_entry(cleaned_id, f"{CLEANED_STEM}.jpg", scale))
    for index in range(scale.annotations_per_image):
        annotations.append(
            _annotation(
                index,
                scale,
                rle,
                image_id=cleaned_id,
                annotation_id=annotation_id,
                with_keypoints=False,
            )
        )
        annotation_id += 1

    images.append(_image_entry(len(stems) + 2, "missing_train.jpg", scale))

    _write_labels(
        root / "train",
        {
            "info": {"description": "benchmark train split"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": _categories(scale, with_skeleton=True),
        },
    )

    return _SplitSummary(
        images=len(stems) + 1,
        records=len(stems) * scale.annotations_per_image,
        files=len(stems),
        issues=2,
    )


def _write_val_split(
    root: Path, scale: Scale, rle: dict[str, Any]
) -> _SplitSummary:
    """Write the validation split: no skeletons, some empty images.

    Without skeleton metadata the parser keeps images that carry no
    annotation and yields a record with ``annotation=None`` for them.
    Every fifth annotation is written without an ``id`` so the parser has
    to hand out fallback instance ids.
    """
    stems = [f"img_{index}" for index in range(scale.images)]
    write_images(root / "validation" / "data", stems, scale.image_size)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 2_000_000
    empty = 0

    for offset, stem in enumerate(stems):
        image_id = 100_000 + offset
        images.append(_image_entry(image_id, f"{stem}.jpg", scale))
        if offset % EMPTY_EVERY == EMPTY_EVERY - 1:
            empty += 1
            continue
        for index in range(scale.annotations_per_image):
            has_id = index % 5 != 4
            annotations.append(
                _annotation(
                    index,
                    scale,
                    rle,
                    image_id=image_id,
                    annotation_id=annotation_id if has_id else None,
                    with_keypoints=index % 3 == 0,
                )
            )
            if has_id:
                annotation_id += 1

    crowd = _annotation(
        1,
        scale,
        rle,
        image_id=100_000,
        annotation_id=annotation_id,
        with_keypoints=False,
    )
    crowd["iscrowd"] = 1
    annotations.append(crowd)

    images.append(_image_entry(100_000 + len(stems), "missing_val.jpg", scale))

    _write_labels(
        root / "validation",
        {
            "info": {"description": "benchmark validation split"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": _categories(scale, with_skeleton=False),
        },
    )

    annotated = len(stems) - empty
    return _SplitSummary(
        images=len(stems),
        records=annotated * scale.annotations_per_image + empty,
        files=len(stems),
        issues=2,
    )


def _write_test_split(root: Path, scale: Scale) -> _SplitSummary:
    """Write the test split: images only, categories nested in ``info``.

    Public COCO test sets ship without annotations, which the parser
    accepts by falling back to an empty annotation list and by looking
    for the categories inside ``info`` when they are not at the top
    level.
    """
    stems = [f"img_{index}" for index in range(scale.images)]
    write_images(root / "test" / "data", stems, scale.image_size)

    images = [
        _image_entry(200_000 + offset, f"{stem}.jpg", scale)
        for offset, stem in enumerate(stems)
    ]
    images.append(
        _image_entry(200_000 + len(stems), "missing_test.jpg", scale)
    )

    _write_labels(
        root / "test",
        {
            "info": {
                "description": "benchmark test split",
                "categories": _categories(scale, with_skeleton=False),
            },
            "licenses": [],
            "images": images,
        },
    )

    return _SplitSummary(
        images=len(stems),
        records=len(stems),
        files=len(stems),
        issues=1,
    )


@benchmark_dataset("coco")
def build_coco(root: Path, scale: Scale) -> BenchmarkCase:
    """Build a three-split COCO dataset in the FiftyOne layout.

    The full scale is used: COCO stores one JSON per split rather than
    one file per image, so generation stays cheap even though every
    annotation carries a segmentation and half of them carry keypoints.
    """
    rle = _rle_mask(scale.image_size)
    summaries = (
        _write_train_split(root, scale, rle),
        _write_val_split(root, scale, rle),
        _write_test_split(root, scale),
    )

    return BenchmarkCase(
        dataset_type="coco",
        source=root,
        features=(
            "fiftyone-layout",
            "multi-split",
            "detection",
            "polygon segmentation",
            "multi-polygon merge",
            "rle segmentation",
            "keypoints",
            "skeletons",
            "fallback instance ids",
            "images without annotations",
            "annotation-free split",
            "categories nested in info",
            "iscrowd skips",
            "missing images",
            "annotation cleaning",
        ),
        images=sum(summary.images for summary in summaries),
        expected_records=sum(summary.records for summary in summaries),
        expected_files=sum(summary.files for summary in summaries),
        expected_issues=sum(summary.issues for summary in summaries),
    )
