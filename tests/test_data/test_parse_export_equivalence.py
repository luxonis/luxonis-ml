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
    reimported_dataset)."""
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
    prev = collector(previous_ldf, round_ndigits=2)
    new = collector(new_ldf, round_ndigits=2)
    assert prev == new


def file_sha256(
    path: Path,
) -> str:
    """The image's hash is used to order the bounding boxes to survive
    the renaming process during export."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_bbox_multiset(
    prepared_ldf: PreparedLDF, *, round_ndigits: int = 2
):
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
                        round(d["x"], round_ndigits),
                        round(d["y"], round_ndigits),
                        round(d["w"], round_ndigits),
                        round(d["h"], round_ndigits),
                    )
                )
        if boxes:
            out.setdefault(key, Counter()).update(boxes)
    return out


def collect_keypoint_multiset(
    prepared_ldf: PreparedLDF, round_ndigits: int = 2
):
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


@pytest.mark.parametrize("url", ["COCO_people_subset.zip"])
@pytest.mark.parametrize(
    ("dataset_type", "collector"),
    [
        # TODO: Do we need it to be exact? Because it's maybe not feasible to save bounding boxes to .12f
        # Bounding boxes
        pytest.param(
            DatasetType.COCO,
            collect_bbox_multiset,
        ),
        pytest.param(
            DatasetType.DARKNET,
            collect_bbox_multiset,
        ),
        pytest.param(
            DatasetType.YOLOV4,
            collect_bbox_multiset,
        ),
        pytest.param(
            DatasetType.YOLOV6,
            collect_bbox_multiset,
        ),
        pytest.param(
            DatasetType.YOLOV8,
            collect_bbox_multiset,
        ),
        pytest.param(
            DatasetType.NATIVE,
            collect_bbox_multiset,
        ),
        # TODO: VOC and maybe createML and
        # Keypoints
        pytest.param(
            DatasetType.COCO,
            collect_keypoint_multiset,
        ),
        pytest.param(
            DatasetType.NATIVE,
            collect_keypoint_multiset,
        ),
        # Instance segmentation
        # TODO: COCO, NATIVE, YOLOV8, VOC
        # Classification
        # TODO: NATIVE, CLS
        # Semantic Segmentation
        # TODO:
    ],
)
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
