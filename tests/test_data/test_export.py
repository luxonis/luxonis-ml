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
        dataset = LuxonisParser(
            str(zip_files[0]),
            dataset_name=dataset_name,
            delete_local=True,
            save_dir=tempdir,
        ).parse()

        dataset = LuxonisParser(
            str(zip_files[1]),
            dataset_name=dataset_name,
            delete_local=False,
            save_dir=tempdir,
        ).parse()

        dataset.make_splits(ratios=(1, 0, 0), replace_old_splits=True)

        loader = LuxonisLoader(dataset, view="train")

        new_data = [img for img, _ in loader]
        assert len(new_data) == len(original_data)
