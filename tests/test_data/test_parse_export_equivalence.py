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
    assert len(zip_files) == 1
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


def _assert_equivalence(
    dataset: LuxonisDataset, new_dataset: LuxonisDataset, collector: Callable
) -> None:
    """Compare two datasets using a given collector (e.g., bbox or
    keypoints)."""
    previous_ldf = PreparedLDF.from_dataset(dataset)
    new_ldf = PreparedLDF.from_dataset(new_dataset)
    prev = collector(previous_ldf)
    new = collector(new_ldf)
    assert prev == new


def file_sha256(
    path: Path,
) -> str:
    """The image's hash is used to order the bounding boxes to survive
    the renaming process during export."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_bbox_multiset(prepared_ldf: PreparedLDF):
    out = {}
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
    out = {}
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
    out = {}
    g = prepared_ldf.processed_df.groupby(
        ["file", "group_id"], maintain_order=True
    )
    for (file_path, _gid), df in g:
        key = (file_sha256(Path(file_path)),)
        keypoints = []
        for row in df.iter_rows(named=True):
            if row["task_type"] == "instance_segmentation":
                d = json.loads(row["annotation"])
                keypoints.extend(d["counts"])
        if keypoints:
            out.setdefault(key, Counter()).update(keypoints)
    return out


def collect_classification_multiset(prepared_ldf: PreparedLDF):
    out = {}
    g = prepared_ldf.processed_df.groupby(
        ["file", "group_id"], maintain_order=True
    )
    for (file_path, _gid), df in g:
        key = (file_sha256(Path(file_path)),)
        classification_entries = []
        for row in df.iter_rows(named=True):
            if row["task_type"] == "classification":
                d = json.loads(row["annotation"])
                classification_entries.extend(d["class_name"])
        if classification_entries:
            out.setdefault(key, Counter()).update(classification_entries)
    return out


test_matrix = [
    # Bounding boxes
    pytest.param(DatasetType.YOLOV4, collect_bbox_multiset),
    pytest.param(DatasetType.YOLOV6, collect_bbox_multiset),
    pytest.param(DatasetType.YOLOV8, collect_bbox_multiset),
    pytest.param(DatasetType.COCO, collect_bbox_multiset),
    pytest.param(DatasetType.DARKNET, collect_bbox_multiset),
    pytest.param(DatasetType.VOC, collect_bbox_multiset),
    pytest.param(DatasetType.NATIVE, collect_bbox_multiset),
]


@pytest.mark.parametrize("url", ["horse_pose.v8i.yolov8.zip"])
@pytest.mark.parametrize(("dataset_type", "collector"), test_matrix)
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
