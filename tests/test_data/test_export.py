from pathlib import Path

import polars as pl
import pytest
from pytest_subtests import SubTests

from luxonis_ml.data import LuxonisLoader, LuxonisParser
from luxonis_ml.enums.enums import DatasetType


@pytest.mark.parametrize("url", ["COCO_people_subset.zip"])
def test_dir_parser(
    dataset_name: str,
    url: str,
    storage_url: str,
    tempdir: Path,
):
    url = f"{storage_url}/{url}"
    dataset = LuxonisParser(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    metadata = dataset._metadata.model_dump()
    del metadata["tasks"]
    del metadata["skeletons"]
    df = dataset._load_df_offline(raise_when_empty=True)
    anns = (
        df.filter(pl.col("task_type").is_in(["keypoints", "boundingbox"]))
        .select(
            [
                "task_name",
                "class_name",
                "instance_id",
                "task_type",
                "annotation",
            ]
        )
        .to_dict(as_series=False)
    )
    anns = {k: sorted(v) for k, v in anns.items()}
    splits = dataset.get_splits()
    assert splits is not None
    splits = {split: sorted(files) for split, files in splits.items()}

    zip_result = dataset.export(tempdir / "exported")
    zip_path = zip_result[0] if isinstance(zip_result, list) else zip_result
    exported_dataset = LuxonisParser(
        str(zip_path / dataset_name),
        dataset_type=DatasetType.NATIVE,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()
    imported_metadata = exported_dataset._metadata.model_dump()
    imported_anns = (
        exported_dataset._load_df_offline(raise_when_empty=True)
        .filter(pl.col("task_type").is_in(["keypoints", "boundingbox"]))
        .select(
            [
                "task_name",
                "class_name",
                "instance_id",
                "task_type",
                "annotation",
            ]
        )
        .to_dict(as_series=False)
    )
    imported_anns = {k: sorted(v) for k, v in imported_anns.items()}
    imported_splits = exported_dataset.get_splits()
    imported_splits = {
        split: sorted(files) for split, files in imported_splits.items()
    }
    assert imported_splits == splits
    del imported_metadata["tasks"]
    del imported_metadata["skeletons"]
    assert imported_metadata == metadata
    assert imported_anns == anns


@pytest.mark.parametrize("url", ["COCO_people_subset.zip"])
def test_export_edge_cases(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url: str,
    subtests: SubTests,
):
    url = f"{storage_url}/{url}"
    dataset = LuxonisParser(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    dataset.make_splits(ratios=(1, 0, 0), replace_old_splits=True)

    loader = LuxonisLoader(dataset, view="train")
    original_data = [img for img, _ in loader]

    with subtests.test("Export with max_zip_size_gb=0.003"):
        dataset.export(
            output_path=tempdir / "exported",
            max_partition_size_gb=0.003,
            zip_output=True,
        )
        zip_files = sorted((tempdir / "exported").glob("*.zip"))
        assert len(zip_files) == 2

    with subtests.test(
        "Parse zip dirs into single dataset, having some splits empty"
    ):
        for i, zip_file in enumerate(zip_files):
            dataset = LuxonisParser(
                str(zip_file),
                dataset_name=dataset_name,
                delete_local=(i == 0),
                save_dir=tempdir,
            ).parse()
        dataset.make_splits(ratios=(1, 0, 0), replace_old_splits=True)
        loader = LuxonisLoader(dataset, view="train")
        new_data = [img for img, _ in loader]
        assert len(new_data) == len(original_data)


@pytest.mark.parametrize("url", ["COCO_people_subset.zip"])
def test_export_regular_splits(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url: str,
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

    dataset.export(
        output_path=tempdir / "exported",
        max_partition_size_gb=0.003,
        zip_output=True,
    )
    zip_files = sorted((tempdir / "exported").glob("*.zip"))

    assert len(zip_files) == 2

    for i, zip_file in enumerate(zip_files):
        dataset = LuxonisParser(
            str(zip_file),
            dataset_name=dataset_name,
            delete_local=(i == 0),
            save_dir=tempdir,
        ).parse()

    new_splits = dataset.get_splits()
    assert new_splits is not None
    assert len(new_splits) == len(original_splits)

    for split in original_splits:
        original_split = sorted(original_splits[split])
        new_split = sorted(new_splits[split])
        assert original_split == new_split
