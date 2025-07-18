import random

import albumentations as A
import numpy as np
from albumentations.core.composition import TransformsSeqType
from typing_extensions import override

from .batch_transform import BatchTransform
from .utils import yield_batches


class BatchCompose(A.Compose):
    transforms: list[BatchTransform]

    def __init__(self, transforms: TransformsSeqType, **kwargs):
        """Compose transforms and handle all transformations regarding
        bounding boxes.

        @param transforms: List of transformations to compose
        @type transforms: TransformsSeqType
        @param kwargs: Additional arguments to pass to A.Compose
        """
        super().__init__(transforms, is_check_shapes=False, **kwargs)

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.batch_size = 1
        for transform in self.transforms:
            self.batch_size *= transform.batch_size

    @override
    def __call__(
        self, data_batch: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        if len(data_batch) != self.batch_size:
            raise ValueError(
                f"Batch size must be equal to {self.batch_size}, "
                f"but got {len(data_batch)}."
            )

        if not self.transforms:
            return data_batch[0]

        for data in data_batch:
            original_image_key = data.pop("_original_image_key", None)
            self.preprocess(data)

        for transform in self.transforms:
            new_batch = []
            for batch in yield_batches(data_batch, transform.batch_size):
                data = transform(**batch)  # type: ignore

                if isinstance(next(iter(data.values())), list):
                    data = {key: value[0] for key, value in batch.items()}

                data = self.check_data_post_transform(data)
                new_batch.append(data)
            data_batch = new_batch

        assert len(data_batch) == 1
        data = data_batch[0]

        data = self.make_contiguous(data)

        data = self.postprocess(data)

        data["_original_image_key"] = original_image_key

        return data

    @staticmethod
    def make_contiguous(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            data[key] = value
        return data
