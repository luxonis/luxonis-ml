import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import typer
from typing_extensions import Annotated

from luxonis_ml.data import BucketStorage, LuxonisDataset, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.typing import Params

name = "benchmark"

DATA_DIR = Path("data/test_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generator(size: int) -> DatasetIterator:
    for i in range(size):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(DATA_DIR / f"img_{i}.jpg"), img)
        for j in range(10):
            offset = j / 50
            yield {
                "file": str(DATA_DIR / f"img_{i}.jpg"),
                "annotation": {
                    "class": "test",
                    "boundingbox": {
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.1 + offset,
                        "h": 0.1 + offset,
                    },
                    "segmentation": {
                        "points": [
                            (0.1 + offset, 0.1),
                            (0.2 + offset, 0.2),
                            (0.3 + offset, 0.3),
                        ],
                        "width": 256,
                        "height": 256,
                    },
                    "keypoints": {
                        "keypoints": [
                            (0.1 + offset, 0.1, 1),
                            (0.2 + offset, 0.2, 2),
                            (0.3 + offset, 0.3, 1),
                        ],
                    },
                },
            }


app = typer.Typer()

normal_config: List[Params] = [
    {"name": "Defocus", "params": {"p": 1}},
    {"name": "Sharpen", "params": {"p": 1}},
    {"name": "Affine", "params": {"p": 1}},
]
batched_config: List[Params] = [
    {
        "name": "MixUp",
        "params": {"p": 1, "alpha": 0.5},
    }
]


def main(
    repeat: Annotated[int, typer.Option(..., "-r", "--repeat")] = 1,
    size: Annotated[int, typer.Option(..., "-s", "--size")] = 10_000,
    no_write: Annotated[bool, typer.Option(..., "-nw", "--no-write")] = False,
    no_read: Annotated[bool, typer.Option(..., "-nr", "--no-read")] = False,
    augment: Annotated[bool, typer.Option(..., "-a", "--augment")] = False,
    batched: Annotated[bool, typer.Option(..., "-b", "--batched")] = False,
) -> None:
    if not no_write:
        avg = 0
        for _ in range(repeat):
            dataset = LuxonisDataset(
                name, delete_existing=True, bucket_storage=BucketStorage.LOCAL
            )
            t = time.time()
            dataset.add(generator(size))
            dataset.make_splits()
            avg += time.time() - t
        typer.echo(f"Time to write: {avg / repeat:.2f}s")

    if not no_read:
        avg = 0
        aug_config = None
        if augment:
            aug_config = batched_config if batched else normal_config
        for _ in range(repeat):
            dataset = LuxonisDataset(name, bucket_storage=BucketStorage.LOCAL)
            loader = LuxonisLoader(
                dataset,
                height=256 if augment else None,
                width=256 if augment else None,
                augmentation_config=aug_config,
            )
            t = time.time()
            for _, _ in loader:
                pass

            avg += time.time() - t
        typer.echo(f"Time to read: {avg / repeat:.2f}s")


if __name__ == "__main__":
    typer.run(main)
