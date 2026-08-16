"""The background class covers a sample that carries no mask."""

import json
from pathlib import Path

import numpy as np

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator

from .utils import create_dataset, create_image


def _dataset_with_an_unannotated_sample(
    dataset_name: str, tempdir: Path
) -> LuxonisDataset:
    """Build a dataset whose masks cover only half of every image.

    Two classes are needed, or a single-channel mask needs no background at
    all. The first sample of the split has to be an annotated one, because
    that is the sample the loader reads to decide.
    """
    # The mask covers the top half, so half the pixels belong to no class.
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

    dataset = create_dataset(dataset_name, generator(), splits={"train": 1.0})

    unannotated = create_image(2, tempdir).absolute()
    df = dataset._load_df_offline()
    assert df is not None
    unannotated_group = next(
        row["group_id"]
        for row in df.iter_rows(named=True)
        if Path(row["file"]) == unannotated
    )
    # The splits are shuffled, so the unannotated sample is moved last by
    # hand. The loader decides from the first one.
    splits_path = dataset._metadata_path / "splits.json"
    splits = json.loads(splits_path.read_text())
    splits["train"] = [
        group for group in splits["train"] if group != unannotated_group
    ] + [unannotated_group]
    splits_path.write_text(json.dumps(splits))
    return dataset


def test_a_sample_with_no_mask_is_all_background(
    dataset_name: str, tempdir: Path
):
    """Both kinds of sample use the same convention.

    A sample with a mask has every pixel assigned, the background class
    taking the ones no class claimed. A sample with no mask used to come
    back all-zero instead, so no pixel had a class at all and the target
    had nothing for a per-pixel loss to point at.
    """
    dataset = _dataset_with_an_unannotated_sample(dataset_name, tempdir)
    loader = LuxonisLoader(dataset, view="train")
    classes = loader._classes[""]
    background = classes["background"]

    masks = [sample.labels["/segmentation"] for sample in loader]

    assert "background" in classes
    for mask in masks:
        assert mask.shape == (len(classes), 512, 512)
        # Every pixel belongs to exactly one class.
        assert mask.sum() == 512 * 512
    # The unannotated sample is the one whose pixels are all background.
    assert [mask[background].sum() for mask in masks].count(512 * 512) == 1
