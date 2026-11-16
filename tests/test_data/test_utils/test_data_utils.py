import json
import uuid
from typing import cast

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from pycocotools import mask as mask_utils

from luxonis_ml.data.utils import merge_uuids
from luxonis_ml.data.utils.data_utils import get_heatmaps

uuid_lists = st.lists(st.uuids().map(str), min_size=1, max_size=8)


@given(uuids=uuid_lists, data=st.data())
def test_merging_does_not_depend_on_order(
    uuids: list[str], data: st.DataObject
):
    shuffled = data.draw(st.permutations(uuids))

    assert merge_uuids(uuids) == merge_uuids(shuffled)


@given(uuids=uuid_lists)
def test_merging_is_deterministic(uuids: list[str]):
    merged = merge_uuids(uuids)

    assert isinstance(merged, uuid.UUID)
    assert merged == merge_uuids(uuids)
    assert merged == merge_uuids(iter(uuids))


def test_heatmaps_count_only_the_sampled_rows():
    box = json.dumps({"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2})
    df = pl.LazyFrame(
        {
            "task_name": ["cars"] * 3,
            "task_type": ["boundingbox"] * 3,
            "annotation": [box] * 3,
        }
    )

    heatmaps = get_heatmaps(df, sample_size=2)

    assert sum(map(sum, heatmaps["cars"]["boundingbox"])) == 2


def test_heatmaps_skip_a_mask_without_foreground():
    rle = mask_utils.encode(np.asfortranarray(np.zeros((8, 8), np.uint8)))
    counts = cast("bytes", rle["counts"]).decode()
    mask = json.dumps({"height": 8, "width": 8, "counts": counts})
    df = pl.LazyFrame(
        {
            "task_name": ["cars"],
            "task_type": ["segmentation"],
            "annotation": [mask],
        }
    )

    assert get_heatmaps(df) == {}
