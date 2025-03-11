from pathlib import Path

import polars as pl
import pytest

from luxonis_ml.data import LuxonisParser


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
        delete_existing=True,
        save_dir=tempdir,
    ).parse()

    metadata = dataset._metadata.model_dump()
    del metadata["tasks"]
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

    zip = dataset.export(tempdir / "exported")
    exported_dataset = LuxonisParser(
        str(zip),
        dataset_name=dataset_name,
        delete_existing=True,
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
    assert imported_metadata == metadata
    assert imported_anns == anns
