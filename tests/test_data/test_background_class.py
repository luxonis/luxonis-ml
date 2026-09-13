"""The background class of a segmentation task."""

import json
from pathlib import Path

import numpy as np
import pycocotools.mask
import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.ldf.annotation import SegmentationAnnotation

from .utils import create_dataset, create_image

SIZE = 512  # `create_image` writes a 512x512 image.


@pytest.fixture
def half_covered(dataset_name: str, tempdir: Path) -> LuxonisDataset:
    """Build a dataset whose masks cover half of each annotated image.

    The dataset needs two classes; the loader adds no background to a
    single-channel mask. The third sample carries no mask.
    """
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[: SIZE // 2] = 1

    def generator() -> DatasetIterator:
        for i, class_name in enumerate(["dog", "cat"]):
            yield {
                "file": create_image(i, tempdir),
                "annotation": {
                    "class": class_name,
                    "segmentation": {"mask": mask},
                },
            }
        yield {"file": create_image(2, tempdir)}

    return create_dataset(dataset_name, generator(), splits={"train": 1.0})


@pytest.fixture
def overlapped(dataset_name: str, tempdir: Path) -> LuxonisDataset:
    """Build a dataset whose two masks overlap over one image.

    The masks claim 300 rows each, of the 512 an image holds, so their
    areas add up to more than the image. Their union stops at row 400.
    """
    first = np.zeros((SIZE, SIZE), dtype=np.uint8)
    first[:300] = 1
    second = np.zeros((SIZE, SIZE), dtype=np.uint8)
    second[100:400] = 1

    def generator() -> DatasetIterator:
        path = create_image(0, tempdir)
        for class_name, mask in [("dog", first), ("cat", second)]:
            yield {
                "file": path,
                "annotation": {
                    "class": class_name,
                    "segmentation": {"mask": mask},
                },
            }

    return create_dataset(dataset_name, generator(), splits={"train": 1.0})


def test_a_sample_with_no_mask_is_all_background(half_covered: LuxonisDataset):
    """The background class takes the pixels that no class claims.

    Every pixel then has exactly one class, so a per-pixel loss always
    has a target.
    """
    loader = LuxonisLoader(half_covered, view="train")
    classes = loader._classes[""]
    background = classes["background"]

    masks = [sample.labels["/segmentation"] for sample in loader]

    for mask in masks:
        assert mask.shape == (len(classes), SIZE, SIZE)
        assert np.array_equal(mask.sum(axis=0), np.ones((SIZE, SIZE)))
    assert sum(mask[background].all() for mask in masks) == 1


def test_the_background_class_ignores_the_split_order(
    half_covered: LuxonisDataset, tempdir: Path
):
    """A task keeps its class map, whichever sample the split puts first."""
    class_maps = []
    for unannotated_first in (True, False):
        _reorder_split(half_covered, unannotated_first, tempdir)
        class_maps.append(
            LuxonisLoader(half_covered, view="train")._classes[""]
        )

    assert class_maps[0] == class_maps[1]
    assert "background" in class_maps[0]


def test_overlapping_masks_do_not_hide_uncovered_pixels(
    overlapped: LuxonisDataset,
):
    """Two masks that overlap cover less than their areas add up to.

    Here they add up to more than the image holds, and the bottom of
    the image still keeps no class.
    """
    loader = LuxonisLoader(overlapped, view="train")
    classes = loader._classes[""]

    assert "background" in classes
    mask = loader[0].labels["/segmentation"]
    assert mask.shape == (len(classes), SIZE, SIZE)
    assert np.array_equal(mask.sum(axis=0), np.ones((SIZE, SIZE)))


def test_a_polyline_mask_needs_no_image_to_measure():
    """An LDF 1.0 polyline carries no mask size, only normalized points."""
    left_half = json.dumps(
        {"points": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]}
    )

    union = pycocotools.mask.merge(LuxonisLoader._sample_masks([left_half]))

    height, width = union["size"]
    assert pycocotools.mask.area(union) == height * width // 2


def test_a_polyline_takes_the_size_of_the_mask_beside_it():
    """A polyline beside an encoded mask takes that mask's size."""
    height, width = 8, 16
    top_half = np.zeros((height, width), dtype=np.uint8)
    top_half[: height // 2] = 1
    mask = SegmentationAnnotation.model_validate(
        {"mask": top_half}
    ).model_dump_json()
    bottom_half = json.dumps(
        {"points": [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]}
    )

    union = pycocotools.mask.merge(
        LuxonisLoader._sample_masks([mask, bottom_half])
    )

    assert union["size"] == [height, width]
    assert pycocotools.mask.area(union) == height * width


def _reorder_split(
    dataset: LuxonisDataset, unannotated_first: bool, tempdir: Path
) -> None:
    """Put the unannotated sample first or last in the train split."""
    unannotated = create_image(2, tempdir).absolute()
    df = dataset._load_df_offline()
    assert df is not None
    group = next(
        row["group_id"]
        for row in df.iter_rows(named=True)
        if Path(row["file"]) == unannotated
    )
    splits_path = dataset._metadata_path / "splits.json"
    splits = json.loads(splits_path.read_text())
    rest = [other for other in splits["train"] if other != group]
    splits["train"] = [group, *rest] if unannotated_first else [*rest, group]
    splits_path.write_text(json.dumps(splits))
