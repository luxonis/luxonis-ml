"""Native records must stay intact when their rows span parquet shards."""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.enums import DatasetType

from .utils import create_image


def test_native_round_trip_preserves_rows_across_shards(
    dataset_name: str, tempdir: Path, monkeypatch: pytest.MonkeyPatch
):
    image = create_image(0, tempdir)
    dataset = LuxonisDataset(dataset_name, delete_local=True).add(
        (
            {
                "file": image,
                "sample_metadata": {"frame": 1},
                "annotation": {
                    "class": "cat",
                    "instance_id": i,
                    "boundingbox": {
                        "x": 0.1 * (i + 1),
                        "y": 0.2,
                        "w": 0.3,
                        "h": 0.4,
                    },
                    "metadata": {"text": str(i)},
                },
            }
            for i in range(3)
        ),
        batch_size=2,
    )
    dataset.make_splits((1, 0, 0))
    columns = [
        "task_name",
        "class_name",
        "instance_id",
        "task_type",
        "annotation",
        "sample_metadata",
    ]
    expected = Counter(
        dataset._load_df_offline(raise_when_empty=True)
        .select(columns)
        .iter_rows()
    )
    annotations_path = dataset._annotations_path
    original_glob = Path.glob

    def reverse_shards(path: Path, pattern: str) -> Iterator[Path]:
        paths = original_glob(path, pattern)
        if path == annotations_path and pattern == "*.parquet":
            return iter(sorted(paths, reverse=True))
        return paths

    # glob order is arbitrary. Force a deterministic order that separates
    # each detection's labels from its final classification row.
    monkeypatch.setattr(Path, "glob", reverse_shards)

    for round_trip in range(2):
        output = tempdir / f"export_{round_trip}"
        dataset.export(output, dataset_type=DatasetType.NATIVE)
        dataset = LuxonisParser(
            str(output / dataset.identifier),
            dataset_type=DatasetType.NATIVE,
            dataset_name=f"{dataset_name}_imported_{round_trip}",
            delete_local=True,
            save_dir=tempdir,
        ).parse()
        actual = Counter(
            dataset._load_df_offline(raise_when_empty=True)
            .select(columns)
            .iter_rows()
        )
        assert actual == expected


@pytest.mark.parametrize("separate_records", [False, True])
def test_loader_keeps_unidentified_objects_aligned_across_shards(
    dataset_name: str,
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    separate_records: bool,
):
    image = create_image(0, tempdir)
    masks = []
    records: list[dict[str, Any]] = []
    centers = [0.25, 0.5, 0.75]
    for slot, center in enumerate(centers):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[:, int((center - 0.125) * 512) : int((center + 0.125) * 512)] = 1
        masks.append(mask)
        records.append(
            {
                "file": image,
                "annotation": {
                    "class": "object",
                    "boundingbox": {
                        "x": center - 0.125,
                        "y": 0.0,
                        "w": 0.25,
                        "h": 1.0,
                    },
                    "keypoints": {"keypoints": [[center, 0.5, 2]]},
                    "instance_segmentation": {"mask": mask},
                    "metadata": {"slot": slot},
                },
            }
        )
    if separate_records:
        records = [
            {
                "file": record["file"],
                "annotation": {"class": "object", field: value},
            }
            for record in records
            for field, value in record["annotation"].items()
            if field != "class"
        ]
    # Six-row shards cut through objects in both input layouts.
    dataset = LuxonisDataset(dataset_name, delete_local=True).add(
        iter(records), batch_size=6
    )
    dataset.make_splits((1, 0, 0))
    original_glob = Path.glob

    def reverse_shards(path: Path, pattern: str) -> Iterator[Path]:
        paths = original_glob(path, pattern)
        if path == dataset._annotations_path and pattern == "*.parquet":
            return iter(sorted(paths, reverse=True))
        return paths

    monkeypatch.setattr(Path, "glob", reverse_shards)
    for cycle in range(2):
        labels = LuxonisLoader(dataset)[0].labels
        boxes = labels["/boundingbox"]
        box_centers = (boxes[:, 1] + boxes[:, 3] / 2).tolist()
        assert labels["/keypoints"][:, 0].tolist() == box_centers
        assert box_centers == centers
        np.testing.assert_array_equal(labels["/instance_segmentation"], masks)
        assert labels["/metadata/slot"].tolist() == [0, 1, 2]

        if cycle == 0:
            output = tempdir / "export"
            dataset.export(output, dataset_type=DatasetType.NATIVE)
            dataset = LuxonisParser(
                str(output / dataset.identifier),
                dataset_type=DatasetType.NATIVE,
                dataset_name=f"{dataset_name}_imported",
                delete_local=True,
                save_dir=tempdir,
            ).parse()
