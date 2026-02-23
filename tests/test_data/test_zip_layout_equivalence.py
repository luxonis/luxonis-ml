import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.utils import LuxonisFileSystem

from .utils import LDFEquivalence


def _resolve_extracted_root(unzip_dir: Path) -> Path:
    ignored_entries = {"__MACOSX", "Thumbs.db", "desktop.ini"}
    current = unzip_dir
    while True:
        visible_entries = [
            entry
            for entry in current.iterdir()
            if entry.name not in ignored_entries
            and not entry.name.startswith(".")
        ]
        if len(visible_entries) == 1 and visible_entries[0].is_dir():
            current = visible_entries[0]
            continue
        return current


def _make_wrapped_zip(dataset_root: Path, wrapped_zip_path: Path) -> None:
    wrapper_dir_name = f"{dataset_root.name}-wrapped"
    with zipfile.ZipFile(
        wrapped_zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for path in dataset_root.rglob("*"):
            arcname = Path(wrapper_dir_name) / path.relative_to(dataset_root)
            zf.write(path, arcname)


@pytest.mark.parametrize(
    ("url", "collectors"),
    [
        (
            "COCO_people_subset.zip",
            [
                LDFEquivalence.collect_bbox_multiset,
                LDFEquivalence.collect_keypoint_multiset,
                LDFEquivalence.collect_instance_segmentation_multiset,
                LDFEquivalence.collect_classification_multiset,
            ],
        ),
        (
            "D2_ParkingLot.zip",
            [
                LDFEquivalence.collect_bbox_multiset,
                LDFEquivalence.collect_keypoint_multiset,
                LDFEquivalence.collect_segmentation_multiset,
                LDFEquivalence.collect_classification_multiset,
            ],
        ),
        (
            "crack-seg.zip",
            [
                LDFEquivalence.collect_bbox_multiset,
                LDFEquivalence.collect_instance_segmentation_multiset,
                LDFEquivalence.collect_classification_multiset,
            ],
        ),
        (
            "imagenet-sample.zip",
            [
                LDFEquivalence.collect_classification_multiset,
            ],
        ),
    ],
)
def test_zip_layout_equivalence(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url: str,
    collectors: list[Callable],
):
    source_url = f"{storage_url}/{url}"
    downloaded_zip = LuxonisFileSystem.download(source_url, tempdir / url)

    extracted_dir = tempdir / f"{downloaded_zip.stem}_extracted"
    with zipfile.ZipFile(downloaded_zip, "r") as zf:
        zf.extractall(extracted_dir)
    dataset_root = _resolve_extracted_root(extracted_dir)

    wrapped_zip = tempdir / f"{downloaded_zip.stem}_wrapped.zip"
    _make_wrapped_zip(dataset_root, wrapped_zip)

    datasets: list[LuxonisDataset] = []
    try:
        # 1) Parse original zip from bucket
        original = LuxonisParser(
            source_url,
            dataset_name=f"{dataset_name}_orig",
            delete_local=True,
            save_dir=tempdir,
        ).parse()
        datasets.append(original)

        # 2) Parse zip with added top-level wrapper directory
        wrapped = LuxonisParser(
            wrapped_zip.as_posix(),
            dataset_name=f"{dataset_name}_wrapped_zip",
            delete_local=True,
            save_dir=tempdir,
        ).parse()
        datasets.append(wrapped)

        # 3) Parse flat unzipped dataset directory
        flat = LuxonisParser(
            dataset_root.as_posix(),
            dataset_name=f"{dataset_name}_flat_dir",
            delete_local=True,
            save_dir=tempdir,
        ).parse()
        datasets.append(flat)

        assert len(original) == len(wrapped) == len(flat)
        for collector in collectors:
            LDFEquivalence.assert_equivalence(original, wrapped, collector)
            LDFEquivalence.assert_equivalence(original, flat, collector)
    finally:
        for dataset in datasets:
            dataset.delete_dataset(delete_local=True)
