import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.exporters import PreparedLDF
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.enums import DatasetType


def _export_and_reimport(
    url: str,
    dataset_type: DatasetType,
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
) -> tuple[LuxonisDataset, LuxonisDataset]:
    """Parse -> export -> re-import and return (original_dataset,
    reimported_dataset) to compare the two."""
    url = f"{storage_url}/{url}"
    dataset = LuxonisParser(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    export_dir = tempdir / "exported"
    dataset.export(
        output_path=export_dir,
        zip_output=True,
        dataset_type=dataset_type,
    )

    zip_files = sorted(export_dir.glob("*.zip"))
    exported_zip = zip_files[0]

    parser = LuxonisParser(
        str(exported_zip),
        dataset_name=f"{dataset_name}_reimported",
        delete_local=True,
        save_dir=tempdir,
    )
    parse_kwargs = {}
    if dataset_type == DatasetType.COCO:
        parse_kwargs["split_val_to_test"] = False

    new_dataset = parser.parse(**parse_kwargs)
    return dataset, new_dataset


def _multiset_equal_with_tolerance(
    prev_map: dict[tuple[str], Counter],
    new_map: dict[tuple[str], Counter],
    tol: float,
) -> None:
    """Assert that two dict[key -> multiset of tuples] are equal up to a
    per-coordinate absolute tolerance `tol`.

    Keys are image hashes; values are Counters of bbox tuples (x, y, w,
    h).
    """
    assert prev_map.keys() == new_map.keys(), (
        f"Different image sets:\nprev-only={set(prev_map) - set(new_map)}\n"
        f"new-only={set(new_map) - set(prev_map)}"
    )

    for key, prev_counter in prev_map.items():
        new_counter = new_map[key]

        if prev_counter == new_counter:
            continue

        prev_list = list(prev_counter.elements())
        new_list = list(new_counter.elements())

        assert len(prev_list) == len(new_list), (
            f"Different number of boxes for {key}: "
            f"{len(prev_list)} vs {len(new_list)}"
        )

        used = [False] * len(new_list)

        def within_tol(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
            return all(
                abs(aa - bb) <= tol for aa, bb in zip(a, b, strict=True)
            )

        for _i, box in enumerate(prev_list):
            found = False
            for j, cand in enumerate(new_list):
                if not used[j] and within_tol(box, cand):
                    used[j] = True
                    found = True
                    break
            assert found, (
                f"No match within tol={tol} for box {box} in image {key}. "
                f"Unmatched candidates: "
                f"{[c for u, c in zip(used, new_list, strict=True) if not u]}"
            )

        assert all(used), (
            f"Extra unmatched boxes in new for {key}: "
            f"{[c for u, c in zip(used, new_list, strict=True) if not u]}"
        )


def _assert_equivalence(
    dataset: LuxonisDataset, new_dataset: LuxonisDataset, collector: Callable
) -> None:
    """Compare two datasets using a given collector (e.g., bbox or
    keypoints)."""
    previous_ldf = PreparedLDF.from_dataset(dataset)
    new_ldf = PreparedLDF.from_dataset(new_dataset)
    prev = collector(previous_ldf)
    new = collector(new_ldf)

    if collector.__name__ == "collect_bbox_multiset":
        # account for precision loss in exporting then parsing
        _multiset_equal_with_tolerance(prev, new, tol=0.02)
    else:
        assert prev == new


def file_sha256(path: Path) -> str:
    """The image's hash is used to order the bounding boxes to survive
    renaming during export."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_bbox_multiset(prepared_ldf: PreparedLDF):
    out: dict[tuple[str], Counter] = {}
    g = prepared_ldf.processed_df.groupby(
        ["file", "group_id"], maintain_order=True
    )
    for (file_path, _gid), df in g:
        key = (file_sha256(Path(file_path)),)
        boxes = []
        for row in df.iter_rows(named=True):
            if row["task_type"] == "boundingbox":
                d = json.loads(row["annotation"])
                boxes.append(
                    (
                        round(d["x"], 2),
                        round(d["y"], 2),
                        round(d["w"], 2),
                        round(d["h"], 2),
                    )
                )
        if boxes:
            out.setdefault(key, Counter()).update(boxes)
    return out


def collect_keypoint_multiset(prepared_ldf: PreparedLDF):
    out: dict[tuple[str], Counter] = {}
    g = prepared_ldf.processed_df.groupby(
        ["file", "group_id"], maintain_order=True
    )
    for (file_path, _gid), df in g:
        key = (file_sha256(Path(file_path)),)
        keypoints = []
        for row in df.iter_rows(named=True):
            if row["task_type"] == "keypoints":
                d = json.loads(row["annotation"])
                keypoints.extend(tuple(kp) for kp in d["keypoints"])
        if keypoints:
            out.setdefault(key, Counter()).update(keypoints)
    return out


def collect_instance_segmentation_multiset(prepared_ldf: PreparedLDF):
    out: dict[tuple[str], Counter] = {}
    g = prepared_ldf.processed_df.groupby(
        ["file", "group_id"], maintain_order=True
    )
    for (file_path, _gid), df in g:
        key = (file_sha256(Path(file_path)),)
        counts = []
        for row in df.iter_rows(named=True):
            if row["task_type"] == "instance_segmentation":
                d = json.loads(row["annotation"])
                counts.extend(d["counts"])
        if counts:
            out.setdefault(key, Counter()).update(counts)
    return out


def collect_classification_multiset(prepared_ldf: PreparedLDF):
    out: dict[tuple[str], Counter] = {}
    g = prepared_ldf.processed_df.groupby(
        ["file", "group_id"], maintain_order=True
    )
    for (file_path, _gid), df in g:
        key = (file_sha256(Path(file_path)),)
        classes = []
        for row in df.iter_rows(named=True):
            if (
                row["task_type"] == "classification"
                and row["instance_id"] == -1
            ):
                classes.append(row["class_name"])
        if classes:
            out.setdefault(key, Counter()).update(classes)
    return out


# Which (DatasetType, collector) pairs to run for each logical annotation type
ANNOTATION_REGISTRY: dict[str, list[tuple[DatasetType, Callable]]] = {
    # Bounding boxes
    "boundingbox": [
        (DatasetType.YOLOV4, collect_bbox_multiset),
        (DatasetType.YOLOV6, collect_bbox_multiset),
        (DatasetType.YOLOV8, collect_bbox_multiset),
        (DatasetType.COCO, collect_bbox_multiset),
        (DatasetType.DARKNET, collect_bbox_multiset),
        (DatasetType.VOC, collect_bbox_multiset),
        (DatasetType.NATIVE, collect_bbox_multiset),
        (DatasetType.CREATEML, collect_bbox_multiset),
        (DatasetType.TFCSV, collect_bbox_multiset),
    ],
    "instance_segmentation": [
        (DatasetType.NATIVE, collect_instance_segmentation_multiset),
        (DatasetType.COCO, collect_instance_segmentation_multiset),
        (DatasetType.YOLOV8, collect_instance_segmentation_multiset),
    ],
    "keypoints": [
        (DatasetType.COCO, collect_keypoint_multiset),
        (DatasetType.NATIVE, collect_keypoint_multiset),
    ],
    "classification": [
        (DatasetType.NATIVE, collect_classification_multiset),
        (DatasetType.CLSDIR, collect_classification_multiset),
    ],
}

DATASETS = [
    {
        "url": "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
        "types": ["boundingbox"],
    },
    {"url": "D2_Tile.png-mask-semantic.zip", "types": ["segmentation"]},
    {
        "url": "COCO_people_subset.zip",
        "types": ["instance_segmentation", "boundingbox", "keypoints"],
    },
]


def build_params():
    """Expand DATASETS x supported annotation types x registry combos
    into pytest params."""
    params = []
    for ds in DATASETS:
        url = ds["url"]
        for anno_type in ds["types"]:
            combos = ANNOTATION_REGISTRY.get(anno_type, [])
            for dataset_type, collector in combos:
                params.append(
                    pytest.param(
                        url,
                        dataset_type,
                        collector,
                        id=f"{url}::{anno_type}::{dataset_type.name}",
                    )
                )
    return params


@pytest.mark.parametrize(("url", "dataset_type", "collector"), build_params())
def test_export_import_equivalence(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url: str,
    dataset_type: DatasetType,
    collector: Callable,
):
    original, reimported = _export_and_reimport(
        url=url,
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        storage_url=storage_url,
        tempdir=tempdir,
    )
    _assert_equivalence(original, reimported, collector)
