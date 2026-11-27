import json
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data import __main__ as data_cli
from luxonis_ml.utils import environ

from .utils import create_dataset, create_image


@pytest.fixture
def datasets(
    tempdir: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[LuxonisDataset, LuxonisDataset]:
    monkeypatch.setattr(environ, "LUXONISML_BASE_PATH", tempdir / "datasets")
    images = [create_image(i, tempdir) for i in range(3)]
    target = create_dataset(
        "dataset_1",
        iter(
            [
                {
                    "file": images[0],
                    "sample_metadata": {"record_id": 0, "nested": {"x": [1]}},
                    "annotation": {
                        "class": "object",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.5,
                            "h": 0.5,
                        },
                    },
                }
            ]
        ),
        splits={"train": 1.0},
    )
    source = create_dataset(
        "dataset_2",
        iter(
            [
                {
                    "file": images[0],
                    "sample_metadata": {"record_id": 99},
                },
                {
                    "file": images[1],
                    "sample_metadata": {
                        "record_id": 1,
                        "source_dataset": "original_dataset",
                    },
                },
                {
                    "file": images[2],
                    "sample_metadata": {"record_id": 2, "payload": "00123-é"},
                },
            ]
        ),
        splits=False,
    )
    source.make_splits({"train": images[:2], "val": images[2:]})
    return target, source


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("preserve_source", [False, True])
def test_merge_source_metadata(
    datasets: tuple[LuxonisDataset, LuxonisDataset],
    *,
    inplace: bool,
    preserve_source: bool,
):
    target, source = datasets
    target_before = target._load_df_offline(raise_when_empty=True)
    source_before = source._load_df_offline(raise_when_empty=True)
    merged = target.merge_with(
        source,
        inplace=inplace,
        new_dataset_name="merged",
        preserve_source=preserve_source,
    )

    expected = {
        0: {"record_id": 0, "nested": {"x": [1]}},
        1: {"record_id": 1, "source_dataset": "original_dataset"},
        2: {"record_id": 2, "payload": "00123-é"},
    }
    if preserve_source:
        expected[0]["source_dataset"] = "dataset_1"
        expected[2]["source_dataset"] = "dataset_2"

    # Check every annotation row, including duplicate and nested metadata.
    df = merged._load_df_offline(raise_when_empty=True)
    for value in df["sample_metadata"]:
        metadata = json.loads(value)
        assert metadata == expected[metadata["record_id"]]

    loader = LuxonisLoader(
        merged, view=["train", "val"], autopopulate_metadata=False
    )
    assert len(loader) == 3
    assert {
        sample.metadata["record_id"]: sample.metadata for sample in loader
    } == expected

    assert source._load_df_offline(raise_when_empty=True).equals(source_before)
    if not inplace:
        assert target._load_df_offline(raise_when_empty=True).equals(
            target_before
        )


def test_merge_preserves_original_source_across_merges(
    datasets: tuple[LuxonisDataset, LuxonisDataset], tempdir: Path
):
    target, source = datasets
    merged = target.merge_with(
        source, inplace=False, new_dataset_name="merged", preserve_source=True
    )
    third = create_dataset(
        "dataset_3",
        iter([{"file": create_image(3, tempdir)}]),
        splits={"train": 1.0},
    )
    result = third.merge_with(merged, preserve_source=True)
    assert {
        cast("str", sample.metadata["source_dataset"])
        for sample in LuxonisLoader(result, view=["train", "val"])
    } == {"dataset_1", "dataset_2", "dataset_3", "original_dataset"}


def test_merge_source_metadata_defaults_off(
    datasets: tuple[LuxonisDataset, LuxonisDataset],
):
    target, source = datasets
    target.merge_with(source)
    sample = LuxonisLoader(target, view="val")[0]
    assert "source_dataset" not in sample.metadata


def test_merge_source_metadata_specific_split(
    datasets: tuple[LuxonisDataset, LuxonisDataset],
):
    target, source = datasets
    merged = target.merge_with(
        source, splits_to_merge=["val"], preserve_source=True
    )
    assert {
        sample.metadata["record_id"]: sample.metadata["source_dataset"]
        for sample in LuxonisLoader(merged, view=["train", "val"])
    } == {0: "dataset_1", 2: "dataset_2"}


@pytest.mark.parametrize("legacy_metadata", ["missing", "null", "empty"])
def test_merge_source_metadata_legacy_records(
    datasets: tuple[LuxonisDataset, LuxonisDataset], legacy_metadata: str
):
    target, source = datasets
    for shard in target._annotations_path.glob("*.parquet"):
        df = pl.read_parquet(shard)
        if legacy_metadata == "missing":
            df = df.drop("sample_metadata")
        else:
            df = df.with_columns(
                pl.lit(
                    None if legacy_metadata == "null" else "", dtype=pl.String
                ).alias("sample_metadata")
            )
        df.write_parquet(shard)

    target.merge_with(source, preserve_source=True)
    metadata = [
        sample.metadata
        for sample in LuxonisLoader(target, autopopulate_metadata=False)
    ]
    assert {"source_dataset": "dataset_1"} in metadata


@pytest.mark.parametrize(
    ("flags", "preserve_source"),
    [
        ((), False),
        (("--preserve-source",), True),
        (("--no-preserve-source",), False),
    ],
)
def test_merge_cli_source_metadata(
    datasets: tuple[LuxonisDataset, LuxonisDataset],
    flags: tuple[str, ...],
    *,
    preserve_source: bool,
):
    target, source = datasets
    data_cli.app(
        [
            "merge",
            source.identifier,
            target.identifier,
            "--new-name",
            "merged",
            *flags,
        ],
        exit_on_error=False,
        result_action="return_value",
    )
    merged = LuxonisDataset("merged")
    metadata = [
        sample.metadata for sample in LuxonisLoader(merged, view="val")
    ]
    assert len(metadata) == 1
    assert metadata[0].get("source_dataset") == (
        "dataset_2" if preserve_source else None
    )
