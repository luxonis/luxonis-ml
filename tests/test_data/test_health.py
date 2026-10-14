from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pytest

from luxonis_ml.data import DatasetIterator, LuxonisParser

from .utils import create_dataset, create_image

ClassDistributionsByType: TypeAlias = dict[str, list[dict[str, Any]]]


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
                "media": img,
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
                "media": img_copy_path,
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
                "media": img,
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


def test_build_health_grid_renders() -> None:
    """The vizlab health grid renders class-distribution and heatmap panels."""
    pytest.importorskip("luxonis_ml.vizlab")
    from luxonis_ml.data.utils.health_plots import build_health_grid

    class_dist: ClassDistributionsByType = {
        "boundingbox": [
            {"class_name": "person", "count": 1240},
            {"class_name": "car", "count": 712},
            {"class_name": "dog", "count": 143},
        ],
        "keypoints": [],  # exercises the empty-distribution placeholder
    }
    heatmaps = {
        "boundingbox": [[i + j for j in range(15)] for i in range(15)],
        "keypoints": None,  # exercises the missing-heatmap placeholder
    }
    image = build_health_grid(class_dist, heatmaps)
    rendered = image.render()
    assert rendered.ndim == 3
    assert rendered.shape[2] == 4
    assert rendered.shape[0] > 0
    assert rendered.shape[1] > 0
    assert rendered[..., 3].max() > 0


def test_build_health_grid_keeps_a_spatial_task_type_without_a_heatmap() -> (
    None
):
    """A spatial task with no heatmap points keeps its class distribution.

    Heatmaps only exist for annotations that yielded points, so a keypoints task
    whose joints are all invisible produces none — but its class counts are
    still worth plotting. Driving the panels off the heatmaps dropped them.
    """
    pytest.importorskip("luxonis_ml.vizlab")
    from luxonis_ml.data.utils.health_plots import build_health_grid

    class_dist: ClassDistributionsByType = {
        "keypoints": [{"class_name": "person", "count": 12}]
    }
    rendered = build_health_grid(class_dist, {}).render()
    assert rendered.shape[2] == 4
    assert rendered[..., 3].max() > 0
    # Same panels as when the heatmap key is present but empty.
    explicit = build_health_grid(class_dist, {"keypoints": None}).render()
    np.testing.assert_array_equal(rendered, explicit)


def test_build_health_grid_omits_metadata_and_task_name_from_titles() -> None:
    """Only spatial types render, and subplot titles do not repeat the task."""
    pytest.importorskip("luxonis_ml.vizlab")
    from luxonis_ml.data.utils import health_plots

    spatial_class_dist: ClassDistributionsByType = {
        "boundingbox": [{"class_name": "person", "count": 1}]
    }
    heatmaps = {"boundingbox": [[1] * 15 for _ in range(15)]}
    with_metadata = health_plots.build_health_grid(
        {
            **spatial_class_dist,
            "metadata": [{"class_name": "sunny", "count": 1}],
        },
        heatmaps,
    ).render()
    without_metadata = health_plots.build_health_grid(
        spatial_class_dist, heatmaps
    ).render()

    np.testing.assert_array_equal(with_metadata, without_metadata)
    assert health_plots._panel_title("boundingbox", "Spatial density") == (
        "<b>Spatial density</b>\n<code>boundingbox</code>"
    )


def test_many_task_types_use_a_wide_layout() -> None:
    """Several task types pack two pairs per row, so the grid is wide, not tall.

    A tall single-column grid would be shrunk hard to fit the screen (shrinking
    its titles too); the wide layout keeps it closer to screen aspect.
    """
    pytest.importorskip("luxonis_ml.vizlab")
    from luxonis_ml.data.utils.health_plots import build_health_grid

    cd: list[dict[str, Any]] = [{"class_name": "person", "count": 10}]
    hm = [[1] * 15 for _ in range(15)]
    types = [
        "boundingbox",
        "keypoints",
        "segmentation",
        "instance_segmentation",
    ]
    wide = build_health_grid(
        dict.fromkeys(types, cd), dict.fromkeys(types, hm)
    ).render()
    # Four task types -> four columns -> wider than tall.
    assert wide.shape[1] > wide.shape[0]

    # Two task types keep the two-column layout, so the four-type grid is packed
    # into more columns (wider) rather than more rows.
    narrow = build_health_grid(
        dict.fromkeys(types[:2], cd),
        dict.fromkeys(types[:2], hm),
    ).render()
    assert wide.shape[1] > narrow.shape[1]


def test_build_health_grid_theme_style_options() -> None:
    """The light theme, a gradient, a mode, and a scale all thread through."""
    pytest.importorskip("luxonis_ml.vizlab")
    from luxonis_ml.data.utils.health_plots import build_health_grid
    from luxonis_ml.vizlab import LIGHT_THEME

    class_dist: ClassDistributionsByType = {
        "boundingbox": [
            {"class_name": "person", "count": 1240},
            {"class_name": "car", "count": 712},
        ]
    }
    heatmaps = {"boundingbox": [[i + j for j in range(15)] for i in range(15)]}
    image = build_health_grid(
        class_dist,
        heatmaps,
        theme=LIGHT_THEME,
        gradient="turbo",
        mode="stacked",
        scale=1.25,
    )
    rendered = image.render()
    assert rendered.shape[0] > 0
    assert rendered.shape[1] > 0
    # The light background is present in the composited grid.
    bg = LIGHT_THEME.background
    matches = (
        (rendered[..., 0] == bg.r)
        & (rendered[..., 1] == bg.g)
        & (rendered[..., 2] == bg.b)
    )
    assert matches.any()

    # A larger scale yields a larger grid.
    smaller = build_health_grid(
        class_dist, heatmaps, theme=LIGHT_THEME, scale=0.6
    ).render()
    assert rendered.shape[0] > smaller.shape[0]
