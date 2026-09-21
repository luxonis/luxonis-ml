from collections.abc import Iterable
from pathlib import Path
from typing import TypeAlias

import numpy as np
import polars as pl
import pytest

from luxonis_ml.data import DatasetIterator, LuxonisParser
from luxonis_ml.data.utils.data_utils import (
    ClassDistributionRow,
    ClassHeatmapRow,
    HeatmapRow,
)

from .utils import create_dataset, create_image

ClassDistributionsByType: TypeAlias = dict[str, list[ClassDistributionRow]]


@pytest.mark.parametrize("url", ["COCO_people_subset.zip"])
def test_dataset_health(
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

    statistics = dataset.get_statistics()

    class_dists = statistics["class_distributions"][""]
    tasks = [
        "segmentation",
        "keypoints",
        "instance_segmentation",
        "boundingbox",
    ]
    for task in tasks:
        task_classes = class_dists[task]
        assert len(task_classes) == 1
        assert task_classes[0]["class_name"] == "person"
        assert task_classes[0]["count"] == 145

    heatmaps = statistics["heatmaps"][""]
    annotation_types = [
        "boundingbox",
        "keypoints",
        "segmentation",
        "instance_segmentation",
    ]
    for ann_type in annotation_types:
        grid = heatmaps[ann_type]
        assert len(grid) == 15
        for row in grid:
            assert len(row) == 15

    assert heatmaps["segmentation"] == heatmaps["instance_segmentation"]

    assert (
        sum(sum(row) for row in statistics["heatmaps"][""]["keypoints"])
        == 1169
    )
    assert (
        sum(sum(row) for row in statistics["heatmaps"][""]["boundingbox"])
        == 145
    )
    assert (
        abs(
            sum(sum(row) for row in statistics["heatmaps"][""]["segmentation"])
            - 64819
        )
        <= 2
    )
    assert (
        abs(
            sum(
                sum(row)
                for row in statistics["heatmaps"][""]["instance_segmentation"]
            )
            - 64819
        )
        <= 2
    )


def test_dataset_sanitize(
    dataset_name: str,
    tempdir: Path,
):
    def generator() -> DatasetIterator:
        for i in range(5):
            img = create_image(i, tempdir)
            img_copy_path = tempdir / f"img_{i}_copy.jpg"
            img_copy_path.write_bytes(img.read_bytes())
            # Original image with annotations
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.8,
                        "h": 0.8,
                    },
                },
            }
            # Duplicate image with same UUID
            yield {
                "file": img_copy_path,
                "annotation": {
                    "class": "person",
                    "boundingbox": {
                        "x": 0.11,
                        "y": 0.11,
                        "w": 0.78,
                        "h": 0.78,
                    },
                },
            }
            # Duplicate annotations
            yield {
                "file": img,
                "annotation": {
                    "class": "person",
                    "boundingbox": {
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.8,
                        "h": 0.8,
                    },
                },
            }

    dataset = create_dataset(dataset_name, generator())

    stats_before = dataset.get_statistics()
    assert len(stats_before["duplicates"]["duplicate_uuids"]) == 5
    assert len(stats_before["duplicates"]["duplicate_annotations"]) == 5

    dataset.remove_duplicates()

    stats_after = dataset.get_statistics()
    assert len(stats_after["duplicates"]["duplicate_uuids"]) == 0
    assert len(stats_after["duplicates"]["duplicate_annotations"]) == 0


def test_per_class_heatmaps(
    dataset_name: str,
    tempdir: Path,
) -> None:
    """``per_class_heatmaps`` splits the density by class and still sums back."""

    def generator() -> DatasetIterator:
        for i in range(6):
            img = create_image(i, tempdir)
            class_name = "person" if i % 2 == 0 else "car"
            yield {
                "file": img,
                "annotation": {
                    "class": class_name,
                    "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
                },
            }

    dataset = create_dataset(dataset_name, generator())

    # The default output does not carry per-class heatmaps.
    assert "class_heatmaps" not in dataset.get_statistics()

    stats = dataset.get_statistics(per_class_heatmaps=True)
    per_class = stats["class_heatmaps"][""]["boundingbox"]
    assert set(per_class) == {"person", "car"}
    for grid in per_class.values():
        assert len(grid) == 15
        assert all(len(row) == 15 for row in grid)

    # Per-class grids partition the combined heatmap exactly.
    combined_total = sum(
        sum(row) for row in stats["heatmaps"][""]["boundingbox"]
    )
    per_class_total = sum(
        sum(sum(row) for row in grid) for grid in per_class.values()
    )
    assert combined_total == per_class_total == 6


def test_per_class_heatmaps_share_sample(
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined and per-class heatmaps are built from one sampled row set."""

    def generator() -> DatasetIterator:
        for i in range(8):
            yield {
                "file": create_image(i, tempdir),
                "annotation": {
                    "class": "person" if i % 2 == 0 else "car",
                    "boundingbox": {
                        "x": i * 0.1,
                        "y": i * 0.1,
                        "w": 0.05,
                        "h": 0.05,
                    },
                },
            }

    dataset = create_dataset(
        f"heatmap_shared_sample_{tempdir.name}", generator()
    )

    from luxonis_ml.data.utils import data_utils

    original_heatmap_rows = data_utils._heatmap_rows
    sample_calls = 0

    def tracked_heatmap_rows(
        df: pl.LazyFrame,
        sample_size: int | None,
        *,
        with_class: bool,
    ) -> Iterable[HeatmapRow | ClassHeatmapRow]:
        nonlocal sample_calls
        sample_calls += 1
        return original_heatmap_rows(df, sample_size, with_class=with_class)

    monkeypatch.setattr(data_utils, "_heatmap_rows", tracked_heatmap_rows)

    stats = dataset.get_statistics(sample_size=3, per_class_heatmaps=True)

    assert sample_calls == 1
    combined = np.asarray(stats["heatmaps"][""]["boundingbox"])
    per_class = stats["class_heatmaps"][""]["boundingbox"]
    class_sum = sum(np.asarray(grid) for grid in per_class.values())
    np.testing.assert_array_equal(combined, class_sum)
