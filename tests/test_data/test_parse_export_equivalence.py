import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from luxonis_ml.data.exporters import PreparedLDF
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.enums import DatasetType


@pytest.mark.parametrize("url", ["COCO_people_subset.zip"])
@pytest.mark.parametrize(
    "dataset_type",
    [
        DatasetType.COCO,
        DatasetType.YOLOV4,
        DatasetType.YOLOV6,
        DatasetType.YOLOV8,
        DatasetType.DARKNET,
    ],
)
def test_export_no_partition(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url: str,
    dataset_type: DatasetType,
):
    url = f"{storage_url}/{url}"
    dataset = LuxonisParser(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    original_splits = dataset.get_splits()
    assert original_splits is not None

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

    if dataset_type == DatasetType.COCO:
        new_dataset = parser.parse(split_val_to_test=False)
    else:
        new_dataset = parser.parse()

    new_splits = new_dataset.get_splits()
    assert new_splits is not None
    assert set(original_splits.keys()) == set(new_splits.keys())

    for split in original_splits:
        original_split = sorted(original_splits[split])
        new_split = sorted(new_splits[split])
        assert original_split == new_split


@pytest.mark.parametrize("url", ["COCO_people_subset.zip"])
@pytest.mark.parametrize(
    "dataset_type",
    [
        DatasetType.COCO,
        DatasetType.YOLOV4,
        DatasetType.YOLOV6,
        DatasetType.YOLOV8,
        DatasetType.DARKNET,
    ],
)
def test_bounding_box_equivalence(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url: str,
    dataset_type: DatasetType,
):
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

    if dataset_type == DatasetType.COCO:
        new_dataset = parser.parse(split_val_to_test=False)
    else:
        new_dataset = parser.parse()

    previous_ldf = PreparedLDF.from_dataset(dataset)
    new_ldf = PreparedLDF.from_dataset(new_dataset)
    prev = collect_bbox_multiset(previous_ldf, round_ndigits=2)
    new = collect_bbox_multiset(new_ldf, round_ndigits=2)
    assert (
        prev == new
    )  # maybe we can assert that the difference is small? Because right now being saved as .12f


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
