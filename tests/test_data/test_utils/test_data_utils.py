import json
from typing import cast

import numpy as np
import polars as pl
from pycocotools import mask as mask_utils

from luxonis_ml.data.utils.data_utils import get_heatmaps


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
