"""The background class covers a sample that carries no mask."""

import json
from pathlib import Path

import numpy as np

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator

from .utils import create_dataset, create_image


def test_a_sample_with_no_mask_is_all_background(
    dataset_name: str, tempdir: Path
):
    """Both kinds of sample use the same convention.

    The loader assigns every pixel of an annotated sample; the
    background class takes the pixels that no class claimed. A sample
    with no mask came back all-zero before the fix, so a per-pixel
    loss had no target.
    """
    dataset = _half_covered_dataset(dataset_name, tempdir)
    loader = LuxonisLoader(dataset, view="train")
    classes = loader._classes[""]
    background = classes["background"]

    masks = [sample.labels["/segmentation"] for sample in loader]

    for mask in masks:
        assert mask.shape == (len(classes), 512, 512)
        assert np.array_equal(mask.sum(axis=0), np.ones((512, 512)))
    # Only the sample with no mask is all background.
    assert [mask[background].sum() for mask in masks].count(512 * 512) == 1


def test_the_background_class_ignores_the_split_order(
    dataset_name: str, tempdir: Path
):
    """Every sample decides, not the one the split put first.

    The old check read sample 0 alone. When that sample carried no
    mask, the task got no background class. The channel count then
    followed the split order.
    """
    dataset = _half_covered_dataset(dataset_name, tempdir)

    class_maps = []
    for unannotated_first in (True, False):
        _reorder_split(dataset, unannotated_first, tempdir)
        class_maps.append(LuxonisLoader(dataset, view="train")._classes[""])

    assert class_maps[0] == class_maps[1]
    assert "background" in class_maps[0]


def _half_covered_dataset(dataset_name: str, tempdir: Path) -> LuxonisDataset:
    """Build a dataset whose masks cover half of each annotated image.

    The dataset needs two classes; the loader adds no background to a
    single-channel mask. The third sample carries no mask.
    """
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:256] = 1

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
