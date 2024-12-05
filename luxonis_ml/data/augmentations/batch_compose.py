from typing import Any, Dict, List

import albumentations as A
import numpy as np
from albumentations.core.composition import TransformsSeqType
from typing_extensions import override

from .batch_transform import BatchBasedTransform
from .batch_utils import yield_batches


class BatchCompose(A.Compose):
    def __init__(self, transforms: TransformsSeqType, **kwargs):
        """Compose transforms and handle all transformations regarding
        bounding boxes.

        @param transforms: List of transformations to compose
        @type transforms: TransformsSeqType
        @param kwargs: Additional arguments to pass to A.Compose
        """
        super().__init__(transforms, **kwargs)

        self.batch_size = 1
        for transform in self.transforms:
            if isinstance(transform, BatchBasedTransform):
                self.batch_size *= transform.batch_size

    @override
    def __call__(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch_data) != self.batch_size:
            raise ValueError(
                f"Batch size must be equal to {self.batch_size}, "
                f"but got {len(batch_data)}."
            )

        for data in batch_data:
            self.preprocess(data)

        for transform in self.transforms:
            assert isinstance(transform, BatchBasedTransform)
            new_batch = []
            for data in yield_batches(batch_data, transform.batch_size):
                data = transform(**data, force_apply=False)

                data = self.check_data_post_transform(data)
                new_batch.append(data)
            batch_data = new_batch

        assert len(batch_data) == 1
        data = batch_data[0]

        self._make_targets_contiguous(data)

        return self.postprocess(data)

    @staticmethod
    def _make_targets_contiguous(data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            data[key] = value
