from pathlib import Path

import pytest

from luxonis_ml.data import LuxonisParser
from luxonis_ml.data.utils.plot_utils import (
    _prepare_class_data,
    _prepare_heatmap_data,
)


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
        delete_existing=True,
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
        _prepare_class_data(task_classes)

    heatmaps = statistics["heatmaps"][""]
    annotation_types = [
        "boundingbox",
        "keypoints",
        "segmentation",
        "instance_segmentation",
    ]
    for ann_type in annotation_types:
        grid = heatmaps[ann_type]
        _prepare_heatmap_data(grid)
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
            - 66953
        )
        <= 2
    )
    assert (
        abs(
            sum(
                sum(row)
                for row in statistics["heatmaps"][""]["instance_segmentation"]
            )
            - 66953
        )
        <= 2
    )
